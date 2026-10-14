"""Tests for ``attach_cohort``."""

import json
from pathlib import Path

import pandas as pd

from bhc_experiment.results import ResultRecord, ScoredResult
from bhc_experiment.results_reader import attach_cohort, build_per_admission_df
from mimic_data_prep.cohort import load_cohort

from .conftest import make_meta


def _make_scored(hadm_id: int, score: float = 0.7) -> ScoredResult:
    """Build a minimal ScoredResult with one metric and a successful llm_result."""
    record_dict = {
        "hadm_id": hadm_id,
        "template_name": "unformatted",
        "prompt": {"system": "sys", "user": "usr"},
        "llm_result": {
            "text": "Generated text.",
            "input_tokens": 1000,
            "output_tokens": 200,
            "cached_tokens": None,
            "model": "gemma-4-e4b",
            "latency_ms": 1500.0,
        },
        "error": None,
        "timestamp": "2026-04-01T00:00:00Z",
    }
    return ScoredResult(
        generation=ResultRecord.from_dict(record_dict),
        scores={"sapbert/f1": score},
    )


def _write_cohort(path: Path, admissions: list[dict]) -> None:
    cohort = {
        "metadata": {"seed": 42, "sample_size": len(admissions)},
        "admissions": admissions,
    }
    path.write_text(json.dumps(cohort) + "\n")


class TestAttachCohort:
    """attach_cohort merges per-admission results with cohort metadata."""

    def test_happy_path_extends_results_with_cohort_columns(self, tmp_path, admission_dict):
        runs = [(make_meta(), [_make_scored(100, 0.7), _make_scored(200, 0.8)])]
        results_df = build_per_admission_df(runs)

        cohort_path = tmp_path / "cohort.json"
        _write_cohort(
            cohort_path,
            [
                admission_dict(hadm_id=100, bhc_category="narrative", los_days=4),
                admission_dict(hadm_id=200, bhc_category="domain_organized", los_days=12),
            ],
        )
        cohort = load_cohort(cohort_path)

        merged = attach_cohort(results_df, cohort)

        assert len(merged) == 2
        assert set(merged["sapbert/f1"]) == {0.7, 0.8}
        assert "bhc_category" in merged.columns
        assert "los_quartile" in merged.columns
        row_100 = merged[merged["hadm_id"] == 100].iloc[0]
        assert row_100["bhc_category"] == "narrative"
        assert row_100["los_days"] == 4
        row_200 = merged[merged["hadm_id"] == 200].iloc[0]
        assert row_200["bhc_category"] == "domain_organized"

    def test_inner_join_drops_result_without_cohort_match(self, tmp_path, admission_dict):
        runs = [(make_meta(), [_make_scored(100), _make_scored(999)])]
        results_df = build_per_admission_df(runs)

        cohort_path = tmp_path / "cohort.json"
        _write_cohort(cohort_path, [admission_dict(hadm_id=100)])
        cohort = load_cohort(cohort_path)

        merged = attach_cohort(results_df, cohort)

        assert len(merged) == 1
        assert merged["hadm_id"].iloc[0] == 100

    def test_inner_join_drops_cohort_admission_without_results(self, tmp_path, admission_dict):
        runs = [(make_meta(), [_make_scored(100)])]
        results_df = build_per_admission_df(runs)

        cohort_path = tmp_path / "cohort.json"
        _write_cohort(
            cohort_path,
            [admission_dict(hadm_id=100), admission_dict(hadm_id=999)],
        )
        cohort = load_cohort(cohort_path)

        merged = attach_cohort(results_df, cohort)

        assert len(merged) == 1
        assert merged["hadm_id"].iloc[0] == 100

    def test_left_join_keeps_result_without_cohort_match(self, tmp_path, admission_dict):
        runs = [(make_meta(), [_make_scored(100), _make_scored(999)])]
        results_df = build_per_admission_df(runs)

        cohort_path = tmp_path / "cohort.json"
        _write_cohort(cohort_path, [admission_dict(hadm_id=100, bhc_category="narrative")])
        cohort = load_cohort(cohort_path)

        merged = attach_cohort(results_df, cohort, how="left")

        assert len(merged) == 2
        unmatched = merged[merged["hadm_id"] == 999].iloc[0]
        assert pd.isna(unmatched["bhc_category"])

    def test_multiple_runs_each_get_cohort_attributes(self, tmp_path, admission_dict):
        """Same admission across two runs produces two rows, both with cohort attrs."""
        runs = [
            (make_meta(prompt="unformatted"), [_make_scored(100, 0.7)]),
            (make_meta(prompt="brief"), [_make_scored(100, 0.8)]),
        ]
        results_df = build_per_admission_df(runs)

        cohort_path = tmp_path / "cohort.json"
        _write_cohort(cohort_path, [admission_dict(hadm_id=100, bhc_category="narrative")])
        cohort = load_cohort(cohort_path)

        merged = attach_cohort(results_df, cohort)

        assert len(merged) == 2
        assert (merged["bhc_category"] == "narrative").all()
        assert set(merged["run"]) == {
            "all-records/unformatted-prompt",
            "all-records/brief-prompt",
        }
