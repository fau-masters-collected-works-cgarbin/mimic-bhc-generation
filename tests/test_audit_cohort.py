"""Tests for scripts/pipeline/audit_cohort.py."""

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "pipeline"
sys.path.insert(0, str(SCRIPTS_DIR))

import audit_cohort  # noqa: E402

from .conftest import per_model, pool_admission, pool_dict  # noqa: E402


def _adm(
    hadm_id: int,
    *,
    token_stratum: str,
    bhc_category: str = "narrative",
    note_category_availability: str = "clinical_narrative",
    los_days: int = 5,
    hospital_expire_flag: int = 0,
    service: str = "MED",
    narrative_tokens: int | dict[str, int] = 5_000,
    note_count: int = 10,
) -> dict:
    """Minimal cohort admission record covering the fields the audit reads.

    ``narrative_tokens`` may be one count spread across every model, or an
    explicit per-model dict when a test needs the models to differ.
    """
    narrative_counts = (
        narrative_tokens if isinstance(narrative_tokens, dict) else per_model(narrative_tokens)
    )
    return {
        "hadm_id": hadm_id,
        "subject_id": 10_000 + hadm_id,
        "token_stratum": token_stratum,
        "bhc_category": bhc_category,
        "note_category_availability": note_category_availability,
        "los_days": los_days,
        "hospital_expire_flag": hospital_expire_flag,
        "service": service,
        "note_count": note_count,
        "narrative_record_token_counts": narrative_counts,
    }


def _cohort_doc(admissions: list[dict]) -> dict:
    return {
        "metadata": {
            "seed": 42,
            "sample_size": len(admissions),
            "stratification_axes": ["token_stratum"],
            "token_strata": {
                "labels": ["4K", "8K", "16K", "32K", "64K", "96K", "112K"],
                "edges": [4_000, 8_000, 16_000, 32_000, 64_000, 96_000, 112_000],
                "binning_rule": "max across pivot tokenizers",
            },
            "per_stratum_counts": {
                label: {"target": 100, "actual": 100}
                for label in ["4K", "8K", "16K", "32K", "64K", "96K", "112K"]
            },
        },
        "admissions": admissions,
    }


def _balanced_cohort_admissions() -> list[dict]:
    """Two strata, each balanced across BHC format and note availability.

    Per stratum: 10 admissions split 50/50 on bhc_category and 50/50 on
    note_category_availability.
    """
    rows: list[dict] = []
    for stratum in ("4K", "8K"):
        for i in range(10):
            rows.append(
                _adm(
                    hadm_id=len(rows) + 1,
                    token_stratum=stratum,
                    bhc_category="narrative" if i < 5 else "domain_organized",
                    note_category_availability=(
                        "clinical_narrative" if i % 2 == 0 else "imaging_only"
                    ),
                    los_days=5 + i,
                    hospital_expire_flag=1 if i == 0 else 0,
                )
            )
    return rows


_STRATUM_MAX_TOKENS = {
    "4K": 4_000,
    "8K": 8_000,
    "16K": 16_000,
    "32K": 32_000,
    "64K": 64_000,
    "96K": 96_000,
    "112K": 112_000,
}


def _write_pool(tmp_path: Path, admissions: list[dict] | None = None) -> Path:
    """Write a filtered-pool JSON to tmp_path. Uses one minimal admission when not provided."""
    path = tmp_path / "filtered_pool.json"
    adms = admissions or [pool_admission(1, max_total_tokens=5_000, note_categories={"Nursing": 1})]
    path.write_text(json.dumps(pool_dict(adms)) + "\n")
    return path


def _write_cohort(tmp_path: Path, admissions: list[dict]) -> Path:
    """Write a cohort JSON and a sibling pool covering every cohort stratum.

    Production cohorts are sampled from the pool, so the pool here has
    one admission per cohort stratum.
    """
    strata_used = sorted({adm["token_stratum"] for adm in admissions})
    pool_rows = [
        pool_admission(
            hadm_id=i,
            max_total_tokens=_STRATUM_MAX_TOKENS[stratum],
            note_categories={"Nursing": 1},
        )
        for i, stratum in enumerate(strata_used, start=1)
    ]
    _write_pool(tmp_path, pool_rows)
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(_cohort_doc(admissions)) + "\n")
    return path


class TestSummarizeStratum:
    """summarize_stratum produces per-covariate distributions for one stratum."""

    def test_format_mix_counts(self) -> None:
        admissions = [
            _adm(1, token_stratum="4K", bhc_category="narrative"),
            _adm(2, token_stratum="4K", bhc_category="narrative"),
            _adm(3, token_stratum="4K", bhc_category="domain_organized"),
        ]
        summary = audit_cohort.summarize_stratum("4K", admissions)
        assert summary["bhc_format_mix"]["narrative"] == 2
        assert summary["bhc_format_mix"]["domain_organized"] == 1

    def test_los_median_and_iqr(self) -> None:
        admissions = [
            _adm(1, token_stratum="4K", los_days=3),
            _adm(2, token_stratum="4K", los_days=5),
            _adm(3, token_stratum="4K", los_days=7),
            _adm(4, token_stratum="4K", los_days=11),
        ]
        summary = audit_cohort.summarize_stratum("4K", admissions)
        assert summary["los_days"]["median"] == 6.0
        assert summary["los_days"]["q1"] == 4.5
        assert summary["los_days"]["q3"] == 8.0

    def test_mortality_rate(self) -> None:
        admissions = [
            _adm(1, token_stratum="4K", hospital_expire_flag=0),
            _adm(2, token_stratum="4K", hospital_expire_flag=0),
            _adm(3, token_stratum="4K", hospital_expire_flag=1),
            _adm(4, token_stratum="4K", hospital_expire_flag=1),
        ]
        summary = audit_cohort.summarize_stratum("4K", admissions)
        assert summary["mortality"]["deaths"] == 2
        assert summary["mortality"]["total"] == 4
        assert summary["mortality"]["rate"] == 0.5

    def test_narrative_token_total_uses_max_across_models(self) -> None:
        m1, m2, m3 = per_model(0)
        admissions = [
            _adm(1, token_stratum="4K", narrative_tokens={m1: 2_000, m2: 100, m3: 100}),
            _adm(2, token_stratum="4K", narrative_tokens={m1: 100, m2: 4_000, m3: 100}),
            _adm(3, token_stratum="4K", narrative_tokens={m1: 100, m2: 100, m3: 10_000}),
        ]
        summary = audit_cohort.summarize_stratum("4K", admissions)
        assert summary["narrative_token_total"]["median"] == 4_000


class TestPoolComposition:
    """Pool admissions are bucketed and classified the same way as cohort admissions."""

    def _write_cohort_with_pool(
        self, tmp_path: Path, pool_admissions: list[dict], cohort_admissions: list[dict]
    ) -> tuple[Path, Path]:
        """Write pool and cohort without the auto-pool overwrite in _write_cohort."""
        pool_path = _write_pool(tmp_path, pool_admissions)
        cohort_path = tmp_path / "cohort.json"
        cohort_path.write_text(json.dumps(_cohort_doc(cohort_admissions)) + "\n")
        return cohort_path, pool_path

    def test_bucketing_edge_values_land_in_lower_stratum(self, tmp_path: Path) -> None:
        """4,000 tokens -> 4K, 4,001 -> 8K. Mirrors pd.cut(right=True).

        The pool bucketing must match the cohort sampler exactly so
        pool vs. cohort comparisons distinguish structural pool skew from
        sampling artifacts. A one-token drift on the edges would scramble
        the comparison.
        """
        cohort_path, pool_path = self._write_cohort_with_pool(
            tmp_path,
            pool_admissions=[
                pool_admission(1, max_total_tokens=4_000, note_categories={"Nursing": 1}),
                pool_admission(2, max_total_tokens=4_001, note_categories={"Nursing": 1}),
                pool_admission(3, max_total_tokens=8_000, note_categories={"Nursing": 1}),
                pool_admission(4, max_total_tokens=8_001, note_categories={"Nursing": 1}),
            ],
            cohort_admissions=[_adm(1, token_stratum="4K")],
        )
        result = audit_cohort.audit_cohort(cohort_path, pool_path=pool_path)
        pool_st = result["pool_per_stratum"]
        assert pool_st["4K"]["n"] == 1
        assert pool_st["8K"]["n"] == 2
        assert pool_st["16K"]["n"] == 1

    def test_nursing_or_physician_is_clinical_narrative(self, tmp_path: Path) -> None:
        cohort_path, pool_path = self._write_cohort_with_pool(
            tmp_path,
            pool_admissions=[
                pool_admission(1, max_total_tokens=5_000, note_categories={"Nursing": 4}),
                pool_admission(
                    2, max_total_tokens=5_000, note_categories={"Physician ": 1, "Radiology": 2}
                ),
                pool_admission(3, max_total_tokens=5_000, note_categories={"Radiology": 3}),
            ],
            cohort_admissions=[_adm(1, token_stratum="8K")],
        )
        result = audit_cohort.audit_cohort(cohort_path, pool_path=pool_path)
        mix = result["pool_per_stratum"]["8K"]["note_category_availability_mix"]
        assert mix.get("clinical_narrative") == 2
        assert mix.get("imaging_only") == 1


class TestStatusReport:
    """The status block reports counts and a mechanical pass/resample verdict.

    Verdict logic: any sampling-kind violation triggers RESAMPLE,
    everything else (clean, structural-only, undersized) is PASS.
    """

    def test_no_violations_no_undersized_reports_pass(self, tmp_path: Path) -> None:
        path = _write_cohort(tmp_path, _balanced_cohort_admissions())
        result = audit_cohort.audit_cohort(path)
        status = result["status"]
        assert status["violation_count"] == 0
        assert status["strata_with_violations"] == []
        assert status["undersized_strata"] == []
        assert status["violations"] == []
        assert status["verdict"] == "PASS"
        assert status["summary"].startswith("Verdict: PASS.")
        assert "All covariates >= 30% floor" in status["summary"]
        assert "all strata at target n" in status["summary"]

    def test_single_stratum_violation_counted(self, tmp_path: Path) -> None:
        """One stratum dips below 30% on note_category_availability.

        First stratum has 9/10 clinical_narrative (90%) and 1/10
        imaging_only (10%). The 10% breaches the 30% floor. BHC format
        is balanced 50/50 in every stratum so the only violation falls
        on note_category_availability and is isolated to one stratum.
        The summary line names the affected cell.
        """
        rows: list[dict] = []
        for i in range(10):
            rows.append(
                _adm(
                    hadm_id=len(rows) + 1,
                    token_stratum="4K",
                    bhc_category="narrative" if i < 5 else "domain_organized",
                    note_category_availability=("clinical_narrative" if i < 9 else "imaging_only"),
                )
            )
        # Second stratum is balanced on every covariate.
        for i in range(10):
            rows.append(
                _adm(
                    hadm_id=len(rows) + 1,
                    token_stratum="8K",
                    bhc_category="narrative" if i < 5 else "domain_organized",
                    note_category_availability=("clinical_narrative" if i < 5 else "imaging_only"),
                )
            )
        path = _write_cohort(tmp_path, rows)
        result = audit_cohort.audit_cohort(path)
        status = result["status"]
        assert status["violation_count"] == 1
        assert status["strata_with_violations"] == ["4K"]
        assert status["undersized_strata"] == []
        assert any(v["stratum"] == "4K" for v in status["violations"])
        assert "4K(imaging_only)" in status["summary"]

    def test_multi_stratum_violations_counted(self, tmp_path: Path) -> None:
        """Two strata both below 30% on imaging_only.

        The script lists both affected cells in the summary line.
        """
        rows: list[dict] = []
        for stratum in ("4K", "8K"):
            for i in range(10):
                rows.append(
                    _adm(
                        hadm_id=len(rows) + 1,
                        token_stratum=stratum,
                        bhc_category="narrative" if i < 5 else "domain_organized",
                        note_category_availability=(
                            "clinical_narrative" if i < 9 else "imaging_only"
                        ),
                    )
                )
        path = _write_cohort(tmp_path, rows)
        result = audit_cohort.audit_cohort(path)
        status = result["status"]
        assert status["violation_count"] == 2
        assert status["strata_with_violations"] == ["4K", "8K"]
        assert "4K(imaging_only)" in status["summary"]
        assert "8K(imaging_only)" in status["summary"]

    def test_undersized_strata_counted_from_metadata(self, tmp_path: Path) -> None:
        """A stratum with actual < target is reported as undersized.

        The audit reads per_stratum_counts from the cohort manifest.
        """
        admissions = _balanced_cohort_admissions()
        doc = _cohort_doc(admissions)
        # Force 4K under target so the audit picks it up as undersized.
        doc["metadata"]["per_stratum_counts"]["4K"] = {"target": 100, "actual": 10}
        _write_pool(tmp_path)
        path = tmp_path / "cohort.json"
        path.write_text(json.dumps(doc) + "\n")
        result = audit_cohort.audit_cohort(path)
        status = result["status"]
        assert status["undersized_strata"] == ["4K"]
        assert "Undersized" in status["summary"]
        assert "4K(10/100)" in status["summary"]


class TestStructuralVsSampling:
    """Floor violations are labeled by whether the pool itself clears the floor.

    Mirrors the audit procedure's classification:
    structural violations cannot be fixed by resampling (the pool is also below
    the floor), sampling violations might be (the pool clears it, the draw
    missed).
    """

    def _write_pool_and_cohort(
        self, tmp_path: Path, pool_admissions: list[dict], cohort_admissions: list[dict]
    ) -> Path:
        pool_path = tmp_path / "filtered_pool.json"
        pool_path.write_text(json.dumps(pool_dict(pool_admissions)) + "\n")
        cohort_path = tmp_path / "cohort.json"
        cohort_path.write_text(json.dumps(_cohort_doc(cohort_admissions)) + "\n")
        return cohort_path

    def _pool_4k_clinical_narrative(self) -> list[dict]:
        """Five 4K pool admissions, all clinical_narrative + narrative BHC."""
        return [
            pool_admission(
                hadm_id=i,
                max_total_tokens=4_000,
                note_categories={"Nursing": 1},
                bhc_category="narrative",
            )
            for i in range(1, 6)
        ]

    def _pool_4k_balanced(self) -> list[dict]:
        """Ten 4K pool admissions split 50/50 on note availability and BHC format."""
        narrative = [
            pool_admission(
                hadm_id=i,
                max_total_tokens=4_000,
                note_categories={"Nursing": 1},
                bhc_category="narrative",
            )
            for i in range(1, 6)
        ]
        domain = [
            pool_admission(
                hadm_id=i,
                max_total_tokens=4_000,
                note_categories={"Radiology": 1},
                bhc_category="domain_organized",
            )
            for i in range(6, 11)
        ]
        return [*narrative, *domain]

    def _mixed_kind_setup(self, tmp_path: Path) -> Path:
        """Cohort with one structural (4K) and one sampling (8K) violation.

        4K pool is all clinical_narrative + narrative BHC, so cohort 4K
        breaches the floor on imaging_only and domain_organized as
        structural. 8K pool is balanced 50/50. Cohort 8K is balanced on
        BHC but dips on imaging_only, which counts as a sampling violation.
        """
        pool_8k_narrative = [
            pool_admission(
                hadm_id=i,
                max_total_tokens=8_000,
                note_categories={"Nursing": 1},
                bhc_category="narrative",
            )
            for i in range(11, 16)
        ]
        pool_8k_domain = [
            pool_admission(
                hadm_id=i,
                max_total_tokens=8_000,
                note_categories={"Radiology": 1},
                bhc_category="domain_organized",
            )
            for i in range(16, 21)
        ]
        pool_admissions = [
            *self._pool_4k_clinical_narrative(),
            *pool_8k_narrative,
            *pool_8k_domain,
        ]
        cohort_4k = [
            _adm(
                hadm_id=i,
                token_stratum="4K",
                bhc_category="narrative",
                note_category_availability="clinical_narrative",
            )
            for i in range(1, 6)
        ]
        cohort_8k = [
            _adm(
                hadm_id=i,
                token_stratum="8K",
                bhc_category="narrative" if i <= 10 else "domain_organized",
                note_category_availability=("clinical_narrative" if i <= 14 else "imaging_only"),
            )
            for i in range(6, 16)
        ]
        return self._write_pool_and_cohort(tmp_path, pool_admissions, [*cohort_4k, *cohort_8k])

    def test_pool_below_floor_marks_violation_structural(self, tmp_path: Path) -> None:
        cohort_admissions = [
            _adm(
                hadm_id=i,
                token_stratum="4K",
                bhc_category="narrative",
                note_category_availability="clinical_narrative",
            )
            for i in range(1, 6)
        ]
        cohort_path = self._write_pool_and_cohort(
            tmp_path, self._pool_4k_clinical_narrative(), cohort_admissions
        )
        result = audit_cohort.audit_cohort(cohort_path)
        flagged = {
            (v["covariate"], v["level"]): v
            for v in result["status"]["violations"]
            if v["stratum"] == "4K"
        }
        assert ("note_category_availability", "imaging_only") in flagged
        assert ("bhc_category", "domain_organized") in flagged
        for v in flagged.values():
            assert v["kind"] == "structural"
            assert v["pool_share"] == 0.0

    def test_pool_above_floor_marks_violation_sampling(self, tmp_path: Path) -> None:
        cohort_admissions = [
            _adm(
                hadm_id=i,
                token_stratum="4K",
                bhc_category="narrative" if i <= 5 else "domain_organized",
                note_category_availability=("clinical_narrative" if i <= 9 else "imaging_only"),
            )
            for i in range(1, 11)
        ]
        cohort_path = self._write_pool_and_cohort(
            tmp_path, self._pool_4k_balanced(), cohort_admissions
        )
        result = audit_cohort.audit_cohort(cohort_path)
        violations = [
            v
            for v in result["status"]["violations"]
            if v["stratum"] == "4K" and v["level"] == "imaging_only"
        ]
        assert len(violations) == 1
        assert violations[0]["kind"] == "sampling"
        assert violations[0]["pool_share"] == 0.5

    def test_pool_within_margin_of_floor_marks_violation_structural(self, tmp_path: Path) -> None:
        """A pool barely above the floor cannot reliably clear it by resampling.

        With pool_share just above 30%, a different seed has near-even
        odds of clearing the floor, so the cell behaves structurally.
        The script flags this as structural to keep the verdict from
        recommending a resample that will not robustly help.
        """
        # Pool 30 admissions, 10 imaging_only -> pool_share 0.333. Above the
        # 30% floor but inside the 5pp margin, so the audit flags this as structural.
        narrative = [
            pool_admission(
                hadm_id=i,
                max_total_tokens=4_000,
                note_categories={"Nursing": 1},
                bhc_category="narrative",
            )
            for i in range(1, 21)
        ]
        imaging = [
            pool_admission(
                hadm_id=i,
                max_total_tokens=4_000,
                note_categories={"Radiology": 1},
                bhc_category="narrative",
            )
            for i in range(21, 31)
        ]
        cohort_admissions = [
            _adm(
                hadm_id=i,
                token_stratum="4K",
                bhc_category="narrative",
                note_category_availability="clinical_narrative",
            )
            for i in range(1, 11)
        ]
        cohort_path = self._write_pool_and_cohort(
            tmp_path, [*narrative, *imaging], cohort_admissions
        )
        result = audit_cohort.audit_cohort(cohort_path)
        violations = [
            v
            for v in result["status"]["violations"]
            if v["stratum"] == "4K" and v["level"] == "imaging_only"
        ]
        assert len(violations) == 1
        assert violations[0]["kind"] == "structural"
        assert 0.30 < violations[0]["pool_share"] < 0.30 + audit_cohort.RESAMPLE_MARGIN

    def test_status_breaks_down_count_by_kind(self, tmp_path: Path) -> None:
        cohort_path = self._mixed_kind_setup(tmp_path)
        status = audit_cohort.audit_cohort(cohort_path)["status"]
        assert status["structural_count"] >= 1
        assert status["sampling_count"] >= 1
        assert status["structural_count"] + status["sampling_count"] == status["violation_count"]
        assert "Structural" in status["summary"]
        assert "Sampling" in status["summary"]

    def test_any_sampling_violation_yields_resample_verdict(self, tmp_path: Path) -> None:
        """A single sampling-kind violation flips the verdict to RESAMPLE
        even when structural breaches dominate the count."""
        cohort_path = self._mixed_kind_setup(tmp_path)
        status = audit_cohort.audit_cohort(cohort_path)["status"]
        assert status["sampling_count"] >= 1
        assert status["verdict"] == "RESAMPLE"
        assert status["summary"].startswith("Verdict: RESAMPLE.")

    def test_structural_only_yields_pass_verdict(self, tmp_path: Path) -> None:
        """Structural breaches alone keep the verdict at PASS because
        resampling cannot fix a pool-level shortfall."""
        cohort_admissions = [
            _adm(
                hadm_id=i,
                token_stratum="4K",
                bhc_category="narrative",
                note_category_availability="clinical_narrative",
            )
            for i in range(1, 6)
        ]
        cohort_path = self._write_pool_and_cohort(
            tmp_path, self._pool_4k_clinical_narrative(), cohort_admissions
        )
        status = audit_cohort.audit_cohort(cohort_path)["status"]
        assert status["sampling_count"] == 0
        assert status["structural_count"] >= 1
        assert status["verdict"] == "PASS"
        assert status["summary"].startswith("Verdict: PASS.")
        assert "Structural" in status["summary"]
