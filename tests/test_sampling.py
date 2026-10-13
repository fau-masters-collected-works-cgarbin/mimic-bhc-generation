"""Tests for experiment cohort sampling logic."""

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from mimic_data_prep.models import MODEL_KEYS
from mimic_data_prep.sampling import (
    assign_los_quartiles,
    assign_token_stratum,
    sample_stratified,
    write_cohort_file,
    write_filtered_pool_file,
)

from .conftest import per_model


def _rtc(total: int) -> dict[str, dict[str, int]]:
    """Build a record_token_counts per-model dict with the total in 'notes'."""
    return per_model(
        {
            "notes": total,
            "labs_all": 0,
            "labs_abnormal": 0,
            "prescriptions": 0,
            "demographics": 0,
        }
    )


def _temporal_density() -> dict:
    return {
        "first_note_offset_hours": -1.0,
        "last_note_offset_hours": -0.5,
        "clinical_span_hours": 72.0,
        "gap_hours_between_notes": {"p50": 4.0, "p75": 6.0, "p90": 12.0, "max": 24.0},
        "notes_per_day": {"p50": 2.0, "p75": 3.0, "p90": 4.0, "max": 5.0},
    }


def _timestamp_completeness() -> dict:
    return {
        "null_charttime_note_count": 1,
        "null_charttime_note_categories": {"ECG": 1},
    }


def _source_diversity() -> dict:
    return {
        "distinct_lab_items": 25,
        "distinct_drugs": 8,
        "distinct_services_touching": 1,
    }


def _pre_admission_notes() -> dict:
    return {
        "count": 0,
        "token_counts": per_model(0),
        "categories": {},
    }


class TestAssignLosQuartiles:
    """Tests for assign_los_quartiles."""

    def test_assigns_four_bins(self) -> None:
        """Each admission gets a quartile label Q1-Q4."""
        df = pd.DataFrame({"los_days": [3, 4, 5, 7, 8, 10, 14, 15, 20, 25, 30, 40]})
        result, edges = assign_los_quartiles(df)
        assert set(result["los_quartile"].unique()) == {"Q1", "Q2", "Q3", "Q4"}
        assert len(edges) == 3  # p25, p50, p75

    def test_boundary_values_land_in_correct_bin(self) -> None:
        """Values exactly at bin edges go into the higher bin (left-inclusive)."""
        # 20 values: 3-7 are Q1, 8-12 are Q2, 13-17 are Q3, 18-22 are Q4
        df = pd.DataFrame({"los_days": list(range(3, 23))})
        result, edges = assign_los_quartiles(df)
        p25, p50, p75 = edges
        for _, row in result.iterrows():
            los = row["los_days"]
            q = row["los_quartile"]
            if q == "Q1":
                assert los < p25, f"los={los} should be < {p25} for Q1"
            elif q == "Q2":
                assert p25 <= los < p50, f"los={los} should be in [{p25}, {p50}) for Q2"
            elif q == "Q3":
                assert p50 <= los < p75, f"los={los} should be in [{p50}, {p75}) for Q3"
            elif q == "Q4":
                assert los >= p75, f"los={los} should be >= {p75} for Q4"


class TestAssignTokenStratum:
    """Tests for assign_token_stratum.

    Bin convention is left-open and right-closed: an admission with
    max_total_tokens of exactly 4,000 goes into the 4K stratum, while
    4,001 goes into the 8K stratum. The pool filter caps admissions at
    112K, so the helper rejects anything above that.
    """

    def test_lower_edge_of_first_bin_is_zero(self) -> None:
        """Zero tokens fall into the smallest stratum."""
        df = pd.DataFrame({"max_total_tokens": [0]})
        result = assign_token_stratum(df)
        assert result["token_stratum"].iloc[0] == "4K"

    def test_upper_edge_inclusive(self) -> None:
        """An admission at exactly 4,000 tokens falls in the 4K stratum.

        The 4K bin is (-inf, 4000], closed at the upper edge, so the boundary
        value falls in 4K and the 8K bin starts at 4,001.
        """
        edges = [4_000, 8_000, 16_000, 32_000, 64_000, 96_000, 112_000]
        df = pd.DataFrame({"max_total_tokens": edges})
        result = assign_token_stratum(df)
        expected = ["4K", "8K", "16K", "32K", "64K", "96K", "112K"]
        assert result["token_stratum"].tolist() == expected

    def test_one_above_edge_goes_to_next_bin(self) -> None:
        """4,001 tokens cross into the 8K stratum, 8,001 into 16K, and so on."""
        df = pd.DataFrame({"max_total_tokens": [4_001, 8_001, 16_001, 32_001, 64_001, 96_001]})
        result = assign_token_stratum(df)
        assert result["token_stratum"].tolist() == ["8K", "16K", "32K", "64K", "96K", "112K"]

    def test_above_pool_budget_raises(self) -> None:
        """An admission above 112K should never reach the cohort sampler.

        A count above 112K raises. The 112K bin stays closed at its edge.
        """
        df = pd.DataFrame({"max_total_tokens": [112_001]})
        with pytest.raises(ValueError, match="exceed"):
            assign_token_stratum(df)


_AXES = ["bhc_category", "los_quartile", "note_category_availability"]


def _make_grid_rows(per_cell: int) -> list[dict]:
    """Build a 2x4x2 grid of synthetic admissions, per_cell per cell."""
    rows = []
    for fmt in ["narrative", "domain_organized"]:
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            for availability in ["clinical_narrative", "imaging_only"]:
                for _ in range(per_cell):
                    rows.append(
                        {
                            "hadm_id": len(rows),
                            "bhc_category": fmt,
                            "los_quartile": q,
                            "note_category_availability": availability,
                        }
                    )
    return rows


class TestSampleStratified:
    """Tests for sample_stratified."""

    def test_returns_correct_total(self) -> None:
        """With enough data, returns samples_per_group * num_cells admissions."""
        # 2 formats x 4 quartiles x 2 availability, 10 per cell = 160 total
        df = pd.DataFrame(_make_grid_rows(per_cell=20))
        result = sample_stratified(df, grouping_columns=_AXES, samples_per_group=10, seed=42)
        assert len(result) == 160

    def test_equal_per_group(self) -> None:
        """Each cell gets exactly samples_per_group admissions."""
        df = pd.DataFrame(_make_grid_rows(per_cell=50))
        result = sample_stratified(df, grouping_columns=_AXES, samples_per_group=25, seed=42)
        counts = result.groupby(_AXES).size()
        assert all(c == 25 for c in counts)
        # All 16 cells must be present.
        assert len(counts) == 16

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces identical samples."""
        df = pd.DataFrame(_make_grid_rows(per_cell=50))
        r1 = sample_stratified(df, grouping_columns=_AXES, samples_per_group=25, seed=42)
        r2 = sample_stratified(df, grouping_columns=_AXES, samples_per_group=25, seed=42)
        assert r1["hadm_id"].tolist() == r2["hadm_id"].tolist()

    def test_warns_on_small_group(self, caplog: pytest.LogCaptureFixture) -> None:
        """Logs a warning when a cell has fewer than samples_per_group rows."""
        rows = []
        for fmt in ["narrative", "domain_organized"]:
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                for availability in ["clinical_narrative", "imaging_only"]:
                    is_small = (
                        fmt == "narrative" and q == "Q1" and availability == "clinical_narrative"
                    )
                    n = 5 if is_small else 30
                    for _ in range(n):
                        rows.append(
                            {
                                "hadm_id": len(rows),
                                "bhc_category": fmt,
                                "los_quartile": q,
                                "note_category_availability": availability,
                            }
                        )
        df = pd.DataFrame(rows)
        with caplog.at_level(logging.WARNING):
            result = sample_stratified(df, grouping_columns=_AXES, samples_per_group=25, seed=42)
        small = result[
            (result["bhc_category"] == "narrative")
            & (result["los_quartile"] == "Q1")
            & (result["note_category_availability"] == "clinical_narrative")
        ]
        assert len(small) == 5
        assert "narrative" in caplog.text
        assert "Q1" in caplog.text

    def test_dict_target_per_cell(self) -> None:
        """A dict per-cell target sets each cell's draw size independently.

        The asymmetric cohort allocation needs different N per stratum, so
        sample_stratified accepts a dict keyed by the grouping-column value
        in addition to the uniform int.
        """
        rows = []
        for label, n in [("A", 10), ("B", 10), ("C", 10)]:
            for _ in range(n):
                rows.append({"hadm_id": len(rows), "stratum": label})
        df = pd.DataFrame(rows)
        targets = {"A": 2, "B": 3, "C": 4}
        result = sample_stratified(
            df, grouping_columns=["stratum"], samples_per_group=targets, seed=42
        )
        counts = result["stratum"].value_counts().to_dict()
        assert counts == targets

    def test_dict_missing_key_raises(self) -> None:
        """A grouping value not in the dict raises KeyError.

        Silent fallback to a default would hide allocation drift. The
        sampler refuses to draw from a stratum the caller did not name.
        """
        rows = [{"hadm_id": i, "stratum": label} for label in ("A", "B") for i in range(5)]
        df = pd.DataFrame(rows)
        targets = {"A": 2}  # B is missing on purpose
        with pytest.raises(KeyError, match="B"):
            sample_stratified(df, grouping_columns=["stratum"], samples_per_group=targets, seed=42)

    def test_dict_underflow_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """The take-all-and-warn branch fires on the dict path too.

        4K is structurally underflowing in the cohort pool, so the dict
        target can exceed available rows. The dict path must inherit the
        same fallback behavior as the int path. The warning prints the
        bare stratum label.
        """
        rows = [{"hadm_id": i, "stratum": "small"} for i in range(3)] + [
            {"hadm_id": i + 10, "stratum": "big"} for i in range(20)
        ]
        df = pd.DataFrame(rows)
        targets = {"small": 25, "big": 5}
        with caplog.at_level(logging.WARNING):
            result = sample_stratified(
                df, grouping_columns=["stratum"], samples_per_group=targets, seed=42
            )
        counts = result["stratum"].value_counts().to_dict()
        assert counts == {"small": 3, "big": 5}
        assert "Cell small" in caplog.text
        assert "('small',)" not in caplog.text


def _make_admissions_df(hadm_ids: list[int]) -> pd.DataFrame:
    """Build a DataFrame with all required columns for write_cohort_file.

    Placeholder values for every column write_cohort_file reads.
    """
    n = len(hadm_ids)
    return pd.DataFrame(
        {
            "hadm_id": hadm_ids,
            "subject_id": [10000 + h for h in hadm_ids],
            "bhc_category": ["narrative"] * n,
            "los_quartile": ["Q1"] * n,
            "los_days": [4] * n,
            "note_category_availability": ["clinical_narrative"] * n,
            "bhc_numbering_convention": [None] * n,
            "bhc_modifiers": [[] for _ in range(n)],
            "note_count": [5] * n,
            "note_categories": [{"Nursing": 5} for _ in range(n)],
            "lab_count": [20] * n,
            "lab_abnormal_count": [4] * n,
            "prescription_count": [4] * n,
            "bhc_word_count": [100] * n,
            "bhc_token_counts": [per_model(130) for _ in range(n)],
            "charting_days": [3] * n,
            "record_token_counts": [_rtc(10000) for _ in range(n)],
            "max_total_tokens": [10_000] * n,
            "narrative_record_token_counts": [per_model(7_000) for _ in range(n)],
            "token_stratum": ["16K"] * n,
            "hospital_expire_flag": [0] * n,
            "temporal_density": [_temporal_density() for _ in range(n)],
            "timestamp_completeness": [_timestamp_completeness() for _ in range(n)],
            "source_diversity": [_source_diversity() for _ in range(n)],
            "pre_admission_notes": [_pre_admission_notes() for _ in range(n)],
            "gender": ["M"] * n,
            "age": [60] * n,
            "service": ["MED"] * n,
            "admittime": ["2150-01-01 00:00:00"] * n,
            "dischtime": ["2150-01-05 00:00:00"] * n,
        }
    )


def _token_strata_meta() -> dict:
    """Self-describing metadata block for the seven strata.

    Tests that exercise write_cohort_file pass this through verbatim.
    """
    return {
        "labels": ["4K", "8K", "16K", "32K", "64K", "96K", "112K"],
        "edges": [4_000, 8_000, 16_000, 32_000, 64_000, 96_000, 112_000],
        "binning_rule": "max across Llama 3.2 3B, Qwen3.5-4B, and Gemma 4 E4B",
    }


def _per_stratum_counts(target: int, actuals: dict[str, int]) -> dict[str, dict[str, int]]:
    """Format the per-stratum target/actual counts the way the writer expects."""
    return {label: {"target": target, "actual": actuals.get(label, 0)} for label in actuals}


def _cohort_writer_kwargs(path: Path, **overrides) -> dict:
    """Default kwargs for write_cohort_file.

    Tests pass overrides to vary specific fields. The cohort write
    interface stratifies on token_stratum. BHC format, LOS quartile, and
    note-category availability are covariates.
    """
    base = {
        "path": path,
        "seed": 42,
        "stratification_axes": ["token_stratum"],
        "los_quartile_edges": [5.0, 8.0, 14.0],
        "token_strata": _token_strata_meta(),
        "per_stratum_counts": {},
        "cohort_size": 15306,
        "bhc_jsonl_path": "/fake/bhc.jsonl",
        "bhc_jsonl_sha256": "abc123",
        "bhc_extraction_commit": ("a1b2c3d", "msg"),
        "sampling_script_commit": ("d4e5f6a", "msg"),
        "tokenizer": {
            "models": {k: {"hf_repo": f"fake/{k}", "vocab_size": 1000} for k in MODEL_KEYS},
            "transformers_version": "5.6.2",
            "add_special_tokens": False,
        },
        "pool_file": {"path": "/fake/filtered_pool.json", "sha256": "deadbeef"},
    }
    base.update(overrides)
    return base


class TestWriteCohortFile:
    """Tests for write_cohort_file."""

    def test_cohort_file_structure(self, tmp_path: Path) -> None:
        """Cohort file has metadata and admissions keys with the expected fields."""
        df = _make_admissions_df([100, 200])
        path = tmp_path / "cohort.json"
        write_cohort_file(df, **_cohort_writer_kwargs(path))
        cohort = json.loads(path.read_text())
        assert "metadata" in cohort
        assert "admissions" in cohort
        assert len(cohort["admissions"]) == 2
        meta = cohort["metadata"]
        assert meta["seed"] == 42
        assert meta["sample_size"] == 2
        assert meta["cohort_size"] == 15306
        assert meta["los_quartile_edges"] == [5.0, 8.0, 14.0]
        assert meta["stratification_axes"] == ["token_stratum"]
        assert meta["data_sources"]["mimic"] == "MIMIC-III v1.4"
        assert meta["data_sources"]["bhc_jsonl"]["sha256"] == "abc123"
        assert meta["data_sources"]["bhc_extraction_repo"]["commit"] == "a1b2c3d"

    def test_admissions_sorted_by_hadm_id(self, tmp_path: Path) -> None:
        """Admissions list is sorted by hadm_id."""
        df = _make_admissions_df([300, 100, 200])
        path = tmp_path / "cohort.json"
        write_cohort_file(df, **_cohort_writer_kwargs(path))
        cohort = json.loads(path.read_text())
        ids = [a["hadm_id"] for a in cohort["admissions"]]
        assert ids == sorted(ids)

    def test_record_counts_in_admissions(self, tmp_path: Path) -> None:
        """Cohort file includes per-admission record counts and BHC word count."""
        df = _make_admissions_df([100])
        df.at[0, "note_count"] = 12
        df.at[0, "note_categories"] = {"Nursing": 8, "Radiology": 3, "ECG": 1}
        df.at[0, "lab_count"] = 45
        df.at[0, "prescription_count"] = 8
        df.at[0, "bhc_word_count"] = 120
        df.at[0, "charting_days"] = 3
        path = tmp_path / "cohort.json"
        write_cohort_file(df, **_cohort_writer_kwargs(path))
        cohort = json.loads(path.read_text())
        adm = cohort["admissions"][0]
        assert adm["note_count"] == 12
        assert adm["note_categories"] == {"Nursing": 8, "Radiology": 3, "ECG": 1}
        assert adm["note_count"] == sum(adm["note_categories"].values())
        assert adm["lab_count"] == 45
        assert adm["prescription_count"] == 8
        assert adm["bhc_word_count"] == 120
        assert adm["charting_days"] == 3

    def test_admission_includes_token_stratum(self, tmp_path: Path) -> None:
        """Each admission has the token-stratum label assigned at sample time."""
        df = _make_admissions_df([100, 200])
        df.at[0, "token_stratum"] = "16K"
        df.at[1, "token_stratum"] = "32K"
        path = tmp_path / "cohort.json"
        write_cohort_file(df, **_cohort_writer_kwargs(path))
        cohort = json.loads(path.read_text())
        labels = [a["token_stratum"] for a in cohort["admissions"]]
        assert labels == ["16K", "32K"]

    def test_admission_includes_max_total_tokens(self, tmp_path: Path) -> None:
        """max_total_tokens is propagated from the pool to the cohort record."""
        df = _make_admissions_df([100])
        df.at[0, "max_total_tokens"] = 18_500
        path = tmp_path / "cohort.json"
        write_cohort_file(df, **_cohort_writer_kwargs(path))
        cohort = json.loads(path.read_text())
        assert cohort["admissions"][0]["max_total_tokens"] == 18_500

    def test_admission_includes_hospital_expire_flag(self, tmp_path: Path) -> None:
        """Mortality covariate passes through to the cohort record."""
        df = _make_admissions_df([100, 200])
        df.at[0, "hospital_expire_flag"] = 0
        df.at[1, "hospital_expire_flag"] = 1
        path = tmp_path / "cohort.json"
        write_cohort_file(df, **_cohort_writer_kwargs(path))
        cohort = json.loads(path.read_text())
        flags = [a["hospital_expire_flag"] for a in cohort["admissions"]]
        assert flags == [0, 1]

    def test_metadata_includes_token_strata(self, tmp_path: Path) -> None:
        """The seven-stratum definition appears in the cohort metadata."""
        df = _make_admissions_df([100])
        path = tmp_path / "cohort.json"
        write_cohort_file(df, **_cohort_writer_kwargs(path))
        cohort = json.loads(path.read_text())
        token_strata = cohort["metadata"]["token_strata"]
        assert token_strata["labels"] == ["4K", "8K", "16K", "32K", "64K", "96K", "112K"]
        assert token_strata["edges"] == [4_000, 8_000, 16_000, 32_000, 64_000, 96_000, 112_000]
        assert "max" in token_strata["binning_rule"]

    def test_metadata_includes_per_stratum_counts(self, tmp_path: Path) -> None:
        """Per-stratum target and actual counts appear in the cohort metadata.

        Underflow on the long-tail strata is expected, so the metadata
        records both the target and the achieved count, so the shortfall is
        readable from the file.
        """
        df = _make_admissions_df([100])
        actuals = {
            "4K": 100,
            "8K": 100,
            "16K": 100,
            "32K": 100,
            "64K": 100,
            "96K": 80,
            "112K": 60,
        }
        counts = _per_stratum_counts(target=100, actuals=actuals)
        path = tmp_path / "cohort.json"
        write_cohort_file(df, **_cohort_writer_kwargs(path, per_stratum_counts=counts))
        cohort = json.loads(path.read_text())
        meta = cohort["metadata"]["per_stratum_counts"]
        assert meta["96K"] == {"target": 100, "actual": 80}
        assert meta["112K"] == {"target": 100, "actual": 60}

    def test_metadata_includes_tokenizer_block(self, tmp_path: Path) -> None:
        """Cohort metadata forwards the pool's tokenizer block.

        Without it, the cohort file does not say which tokenizer set produced
        ``record_token_counts``, ``narrative_record_token_counts``,
        ``max_total_tokens``, or ``token_stratum``.
        """
        df = _make_admissions_df([100])
        path = tmp_path / "cohort.json"
        write_cohort_file(df, **_cohort_writer_kwargs(path))
        cohort = json.loads(path.read_text())
        tokenizer = cohort["metadata"]["tokenizer"]
        assert tokenizer["transformers_version"] == "5.6.2"
        assert set(tokenizer["models"].keys()) == set(MODEL_KEYS)

    def test_metadata_includes_pool_file_reference(self, tmp_path: Path) -> None:
        """Cohort metadata records the pool file path and sha256.

        The pool file has the filters and note_redundancy_config, and the
        cohort points at it.
        """
        df = _make_admissions_df([100])
        path = tmp_path / "cohort.json"
        write_cohort_file(df, **_cohort_writer_kwargs(path))
        cohort = json.loads(path.read_text())
        pool_ref = cohort["metadata"]["data_sources"]["pool_file"]
        assert pool_ref == {"path": "/fake/filtered_pool.json", "sha256": "deadbeef"}


def _make_pool_df(hadm_ids: list[int]) -> pd.DataFrame:
    """Build a DataFrame matching what load_filtered_pool produces.

    Every column write_filtered_pool_file expects, with per-admission values
    so the writer exercises each type conversion.
    """
    n = len(hadm_ids)
    return pd.DataFrame(
        {
            "hadm_id": hadm_ids,
            "subject_id": [1000 + h for h in hadm_ids],
            "bhc_char_count": [2500] * n,
            "bhc_word_count": [400] * n,
            "bhc_category": ["narrative"] * n,
            "bhc_numbering_convention": [None] * n,
            "bhc_modifiers": [[] for _ in range(n)],
            "los_days": [5] * n,
            "note_count": [7] * n,
            "charting_days": [4] * n,
            "lab_count": [25] * n,
            "lab_abnormal_count": [5] * n,
            "prescription_count": [3] * n,
            "admittime": ["2150-01-01 08:00:00"] * n,
            "dischtime": ["2150-01-06 14:00:00"] * n,
            "hospital_expire_flag": [0] * n,
            "gender": ["M"] * n,
            "age": [60] * n,
            "mesh_age_group": ["Middle Aged"] * n,
            "service": ["MED"] * n,
            "bhc_token_counts": [per_model(520 * (i + 1)) for i in range(n)],
            "record_token_counts": [_rtc(12_000 * (i + 1)) for i in range(n)],
            "max_total_tokens": [12_000 * (i + 1) for i in range(n)],
            "narrative_record_token_counts": [per_model(8_000 * (i + 1)) for i in range(n)],
            "temporal_density": [_temporal_density() for _ in range(n)],
            "timestamp_completeness": [_timestamp_completeness() for _ in range(n)],
            "source_diversity": [_source_diversity() for _ in range(n)],
            "pre_admission_notes": [_pre_admission_notes() for _ in range(n)],
            "note_categories": [{"Nursing": 4, "Physician ": 3} for _ in range(n)],
        }
    )


class TestWriteFilteredPoolFile:
    """Tests for write_filtered_pool_file."""

    def _common_kwargs(self, path: Path) -> dict:
        return {
            "path": path,
            "bhc_jsonl_path": "/fake/bhc.jsonl",
            "bhc_jsonl_sha256": "abc123",
            "bhc_extraction_commit": ("a1b2c3d", "extraction commit"),
            "build_script_commit": ("d4e5f6a", "build commit"),
            "pool_definition": {
                "metavision_only": True,
                "filters": {
                    "los_days_min": 3,
                    "charting_days_min": 3,
                    "note_count_min": 2,
                    "lab_count_min": 10,
                    "bhc_char_count_max": 15000,
                },
            },
            "tokenizer": {
                "models": {k: {"hf_repo": f"fake/{k}", "vocab_size": 1000} for k in MODEL_KEYS},
                "transformers_version": "5.6.2",
                "add_special_tokens": False,
                "records_included": {"notes": "all except discharge summaries"},
            },
            "pool_filter_stats": {
                "before_token_budget": 5,
                "dropped_above_token_budget": 2,
            },
        }

    def test_file_structure(self, tmp_path: Path) -> None:
        """Pool file has metadata and admissions keys and expected provenance."""
        path = tmp_path / "filtered_pool.json"
        df = _make_pool_df([100, 200, 300])
        write_filtered_pool_file(df, **self._common_kwargs(path))
        pool = json.loads(path.read_text())
        assert "admissions" in pool
        meta = pool["metadata"]
        assert meta["pool_size"] == 3
        assert meta["script_commit"]["hash"] == "d4e5f6a"
        assert meta["data_sources"]["bhc_extraction_repo"]["commit"] == "a1b2c3d"
        assert meta["pool_definition"]["filters"]["los_days_min"] == 3
        assert meta["tokenizer"]["add_special_tokens"] is False
        assert meta["pool_filter_stats"] == {
            "before_token_budget": 5,
            "dropped_above_token_budget": 2,
        }

    def test_admissions_sorted_by_hadm_id(self, tmp_path: Path) -> None:
        """Admissions list is sorted by hadm_id for stable diffs."""
        path = tmp_path / "filtered_pool.json"
        df = _make_pool_df([300, 100, 200])
        write_filtered_pool_file(df, **self._common_kwargs(path))
        pool = json.loads(path.read_text())
        hadm_ids = [a["hadm_id"] for a in pool["admissions"]]
        assert hadm_ids == [100, 200, 300]

    def test_admission_record_shape(self, tmp_path: Path) -> None:
        """Each admission record has the expected fields and types."""
        path = tmp_path / "filtered_pool.json"
        df = _make_pool_df([100])
        write_filtered_pool_file(df, **self._common_kwargs(path))
        pool = json.loads(path.read_text())
        admission = pool["admissions"][0]
        assert admission["hadm_id"] == 100
        assert admission["note_categories"] == {"Nursing": 4, "Physician ": 3}
        assert admission["bhc_numbering_convention"] is None
        assert admission["record_token_counts"] == _rtc(12_000)
        assert admission["bhc_token_counts"] == per_model(520)
        assert admission["lab_abnormal_count"] == 5
        # No stratification fields leak in.
        for stratum_field in ("los_quartile", "note_category_availability"):
            assert stratum_field not in admission

    def test_admission_record_includes_max_total_tokens(self, tmp_path: Path) -> None:
        """max_total_tokens is computed at pool-build time so the
        binning rule (max across the three pivot tokenizers) is visible
        in the manifest, so nothing has to recompute it.
        """
        path = tmp_path / "filtered_pool.json"
        df = _make_pool_df([100, 200])
        write_filtered_pool_file(df, **self._common_kwargs(path))
        pool = json.loads(path.read_text())
        assert pool["admissions"][0]["max_total_tokens"] == 12_000
        assert pool["admissions"][1]["max_total_tokens"] == 24_000
