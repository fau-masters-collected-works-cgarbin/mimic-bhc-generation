"""Tests for the typed cohort reader."""

import json
from pathlib import Path

import duckdb
import pytest

from mimic_data_prep import mimic
from mimic_data_prep.cohort import (
    Admission,
    Cohort,
    _pull_record_text_parts,
    _zero_record_counts,
    compute_mesh_age_group,
    compute_token_counts,
    load_filtered_pool_file,
    max_total_tokens,
    total_tokens,
)
from mimic_data_prep.models import MODEL_KEYS

from .conftest import per_model, pool_admission
from .test_loader import _write_loader_fixtures


class TestZeroRecordCounts:
    def test_every_model_starts_at_zero_for_each_record_type(self):
        zeros = _zero_record_counts()
        assert set(zeros) == set(MODEL_KEYS)
        for per_type in zeros.values():
            assert per_type == {
                "notes": 0,
                "labs_all": 0,
                "labs_abnormal": 0,
                "prescriptions": 0,
                "demographics": 0,
            }

    def test_each_model_gets_an_independent_dict(self):
        # A later pass writes counts into record_counts[hid][model][record_type], so each
        # model needs its own dict. A shared inner dict would leak one model's
        # write into the others.
        zeros = _zero_record_counts()
        first, second = MODEL_KEYS[0], MODEL_KEYS[1]
        zeros[first]["notes"] = 5
        assert zeros[second]["notes"] == 0


def _cohort_dict(admissions: list[dict]) -> dict:
    return {
        "metadata": {"seed": 42, "sample_size": len(admissions)},
        "admissions": admissions,
    }


def _write(path: Path, cohort: dict) -> None:
    path.write_text(json.dumps(cohort) + "\n")


class TestAdmissionFromDict:
    def test_constructs_from_full_dict(self, admission_dict) -> None:
        adm = Admission.from_dict(admission_dict(hadm_id=123))
        assert adm.hadm_id == 123
        assert adm.bhc_category == "narrative"
        # record_token_counts is a per-model dict of per-component counts.
        assert set(adm.record_token_counts.keys()) == set(MODEL_KEYS)
        for rtc in adm.record_token_counts.values():
            assert rtc["notes"] == 12000
            assert rtc["labs_abnormal"] == 400
        for rtc in adm.record_token_counts.values():
            assert total_tokens(rtc) == 12000 + 2000 + 500 + 500
        # bhc_token_counts is a per-model dict.
        assert set(adm.bhc_token_counts.keys()) == set(MODEL_KEYS)


class TestTotalTokens:
    """Tests for the total_tokens helper."""

    def test_excludes_labs_abnormal(self) -> None:
        """labs_abnormal is a subset of labs_all and must not contribute
        to the total."""
        rtc = {
            "notes": 1000,
            "labs_all": 200,
            "labs_abnormal": 9999,  # Large value to make double-counting obvious.
            "prescriptions": 80,
            "demographics": 20,
        }
        assert total_tokens(rtc) == 1000 + 200 + 80 + 20


class TestMaxTotalTokens:
    """Tests for max_total_tokens."""

    def test_picks_largest_model_total(self) -> None:
        per_model_counts = {
            "llama_3_2_3b": {
                "notes": 1000,
                "labs_all": 100,
                "labs_abnormal": 0,
                "prescriptions": 50,
                "demographics": 10,
            },
            "qwen_3_5_4b": {
                "notes": 1200,
                "labs_all": 110,
                "labs_abnormal": 0,
                "prescriptions": 55,
                "demographics": 12,
            },
            "gemma_4_e4b": {
                "notes": 1100,
                "labs_all": 105,
                "labs_abnormal": 0,
                "prescriptions": 52,
                "demographics": 11,
            },
        }
        # Qwen is largest: 1200 + 110 + 55 + 12 = 1377.
        assert max_total_tokens(per_model_counts) == 1377


class TestCohortFromDict:
    def test_constructs_from_dict(self, admission_dict) -> None:
        data = _cohort_dict([admission_dict(hadm_id=1), admission_dict(hadm_id=2)])
        cohort = Cohort.from_dict(data)
        assert isinstance(cohort.admissions[0], Admission)
        assert cohort.admissions[0].hadm_id == 1


class TestComputeMeshAgeGroup:
    """compute_mesh_age_group maps integer ages to MeSH life-stage labels.

    Boundaries are tested at both the lower edge (first age in the bucket)
    and the upper edge (last age in the bucket) for every category. MeSH
    "Newborn" (≤1 month) is not in the table because the sampler stores
    age as integer years, so age=0 falls into "Infant" (the more inclusive
    sub-year bucket). Age 90 is the MIMIC-III HIPAA-shifted cap for
    patients >89, which falls into "Aged, 80 and over".
    """

    @pytest.mark.parametrize(
        ("age", "expected"),
        [
            (0, "Infant"),
            (1, "Infant"),
            (2, "Preschool"),
            (5, "Preschool"),
            (6, "Child"),
            (12, "Child"),
            (13, "Adolescent"),
            (18, "Adolescent"),
            # MeSH overlap (Young Adult 19-24, Adult 19-44) resolves to Young Adult.
            (19, "Young Adult"),
            (24, "Young Adult"),
            # Past Young Adult, the bucket is Adult.
            (25, "Adult"),
            (44, "Adult"),
            (45, "Middle Aged"),
            (64, "Middle Aged"),
            (65, "Aged"),
            (79, "Aged"),
            (80, "Aged, 80 and over"),
            (90, "Aged, 80 and over"),
        ],
    )
    def test_age_maps_to_mesh_group(self, age: int, expected: str) -> None:
        assert compute_mesh_age_group(age) == expected


class TestLoadFilteredPoolFile:
    """Tests for load_filtered_pool_file."""

    def test_roundtrip_with_writer(self, tmp_path: Path) -> None:
        """Write with write_filtered_pool_file, read with load_filtered_pool_file."""
        import pandas as pd

        from mimic_data_prep.sampling import write_filtered_pool_file

        path = tmp_path / "filtered_pool.json"
        # Distinct 'notes' values let the assertion below tell which row is which.
        df_in = pd.DataFrame(
            [
                pool_admission(
                    hadm_id=200,
                    record_token_counts=per_model(
                        {
                            "notes": 15_000,
                            "labs_all": 0,
                            "labs_abnormal": 0,
                            "prescriptions": 0,
                            "demographics": 0,
                        }
                    ),
                ),
                pool_admission(
                    hadm_id=100,
                    record_token_counts=per_model(
                        {
                            "notes": 8_000,
                            "labs_all": 0,
                            "labs_abnormal": 0,
                            "prescriptions": 0,
                            "demographics": 0,
                        }
                    ),
                ),
            ]
        )
        write_filtered_pool_file(
            df_in,
            path=path,
            bhc_jsonl_path="/fake/bhc.jsonl",
            bhc_jsonl_sha256="abc123",
            bhc_extraction_commit=("a1b2c3d", "extraction commit"),
            build_script_commit=("d4e5f6a", "build commit"),
            pool_definition={"metavision_only": True, "filters": {}},
            tokenizer={
                "models": {k: {"hf_repo": f"fake/{k}", "vocab_size": 1000} for k in MODEL_KEYS},
                "transformers_version": "5.6.2",
                "add_special_tokens": False,
                "records_included": {"notes": "all"},
            },
            pool_filter_stats={"before_token_budget": 2, "dropped_above_token_budget": 0},
        )
        df_out, metadata = load_filtered_pool_file(path)
        # Writer sorts by hadm_id, so the reader sees them in sorted order.
        assert df_out["hadm_id"].tolist() == [100, 200]
        # record_token_counts round-trips as a per-model dict of per-component dicts.
        assert df_out["record_token_counts"].iloc[0]["llama_3_2_3b"]["notes"] == 8_000
        assert df_out["record_token_counts"].iloc[1]["llama_3_2_3b"]["notes"] == 15_000
        assert metadata["tokenizer"]["add_special_tokens"] is False
        assert metadata["data_sources"]["bhc_jsonl"]["sha256"] == "abc123"


class _WordTokenizer:
    """Fake tokenizer: one token per whitespace-delimited word, so a token count
    equals a word count. Enough to exercise compute_token_counts without loading
    the real Hugging Face tokenizers."""

    def encode(self, text: str, **_: object) -> list[str]:
        return text.split()


class TestComputeTokenCounts:
    """BHC text and the per-admission record-type counts."""

    @pytest.fixture
    def _synthetic_mimic(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(mimic, "MIMIC_DIR", tmp_path)
        _write_loader_fixtures(tmp_path, tmp_path / "bhc.jsonl")

    @pytest.mark.usefixtures("_synthetic_mimic")
    def test_counts_every_type_and_bhc_per_model(self) -> None:
        conn = duckdb.connect()
        tokenizers = {key: _WordTokenizer() for key in MODEL_KEYS}
        # bhc_texts is supplied by the caller, so the BHC counts are exactly the
        # word counts passed in, per model.
        bhc_texts = {1001: "one two three", 1002: "alpha"}

        record_counts, bhc_counts = compute_token_counts(conn, [1001, 1002], bhc_texts, tokenizers)

        assert bhc_counts == {
            1001: dict.fromkeys(MODEL_KEYS, 3),
            1002: dict.fromkeys(MODEL_KEYS, 1),
        }

        record_types = {"notes", "labs_all", "labs_abnormal", "prescriptions", "demographics"}
        for hadm_id in (1001, 1002):
            for model in MODEL_KEYS:
                assert set(record_counts[hadm_id][model]) == record_types

        for model in MODEL_KEYS:
            # 1001 has notes, both lab subsets, and a prescription.
            counts_1001 = record_counts[1001][model]
            assert counts_1001["notes"] > 0
            assert counts_1001["labs_all"] >= counts_1001["labs_abnormal"] > 0
            assert counts_1001["prescriptions"] > 0
            # 1002 has a note but no labs and no prescriptions, so those are zero.
            counts_1002 = record_counts[1002][model]
            assert counts_1002["notes"] > 0
            assert counts_1002["labs_all"] == 0
            assert counts_1002["labs_abnormal"] == 0
            assert counts_1002["prescriptions"] == 0

    @pytest.mark.usefixtures("_synthetic_mimic")
    def test_requested_admission_with_no_records_is_all_zeros(self) -> None:
        conn = duckdb.connect()
        tokenizers = {key: _WordTokenizer() for key in MODEL_KEYS}
        # 4001 is absent from the fixtures, so it has no records and an empty BHC.
        record_counts, bhc_counts = compute_token_counts(conn, [4001], {4001: ""}, tokenizers)

        all_zero_counts = dict.fromkeys(
            ["notes", "labs_all", "labs_abnormal", "prescriptions", "demographics"], 0
        )
        for model in MODEL_KEYS:
            assert record_counts[4001][model] == all_zero_counts
            assert bhc_counts[4001][model] == 0


class TestPullRecordTextPartsOrder:
    """The per-admission text that feeds tokenization concatenates records in a
    deterministic order.

    Admission 1004 in the shared fixture has records that tie on their sort
    timestamp, written to the source files in reverse ROW_ID order. Without a
    deterministic ORDER BY the string_agg concatenates in scan order, which on
    the real multi-million-row tables is not reproducible.
    """

    @pytest.fixture
    def _synthetic_mimic(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(mimic, "MIMIC_DIR", tmp_path)
        _write_loader_fixtures(tmp_path, tmp_path / "bhc.jsonl")

    @pytest.mark.usefixtures("_synthetic_mimic")
    def test_notes_concatenate_in_row_id_order(self) -> None:
        conn = duckdb.connect()
        text = _pull_record_text_parts(conn, [1004])["notes"][1004]
        # ROW_ID 10 is "first by ROW_ID", ROW_ID 11 is "second by ROW_ID".
        assert text.index("first by ROW_ID") < text.index("second by ROW_ID")

    @pytest.mark.usefixtures("_synthetic_mimic")
    def test_labs_all_concatenate_in_row_id_order(self) -> None:
        conn = duckdb.connect()
        text = _pull_record_text_parts(conn, [1004])["labs_all"][1004]
        # ROW_ID 10 is Creatinine and ROW_ID 11 is Hemoglobin. Both share a CHARTTIME.
        assert text.index("Creatinine") < text.index("Hemoglobin")

    @pytest.mark.usefixtures("_synthetic_mimic")
    def test_labs_abnormal_concatenate_in_row_id_order(self) -> None:
        conn = duckdb.connect()
        text = _pull_record_text_parts(conn, [1004])["labs_abnormal"][1004]
        # Both 1004 labs are abnormal and share a CHARTTIME, so ROW_ID breaks the tie.
        assert text.index("Creatinine") < text.index("Hemoglobin")

    @pytest.mark.usefixtures("_synthetic_mimic")
    def test_prescriptions_concatenate_in_row_id_order(self) -> None:
        conn = duckdb.connect()
        text = _pull_record_text_parts(conn, [1004])["prescriptions"][1004]
        # ROW_ID 10 is Warfarin and ROW_ID 11 is Aspirin. Both share a STARTDATE.
        assert text.index("Warfarin") < text.index("Aspirin")
