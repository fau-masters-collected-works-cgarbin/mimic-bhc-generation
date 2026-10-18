"""Tests for bhc_experiment.results_reader."""

import json
import warnings
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from bhc_experiment import results_reader
from bhc_experiment.config import CliParameters
from bhc_experiment.prepared_store import write_prepared
from bhc_experiment.results import (
    ResultRecord,
    ScoredResult,
)
from bhc_experiment.results_reader import (
    build_per_admission_df,
    build_summary_df,
    discover_model_dirs,
    discover_runs,
    extract_run_label,
    get_result_record,
    load_eval_scores,
    load_generations,
    load_reference_bhc,
    load_result_records,
    load_run,
    load_run_df,
    merge_run_pair,
    run_cli_parameters,
    run_prepared_store,
    worst_by_composite_z,
)

from .conftest import make_bundle, make_meta, write_model_dir

RUN_DIR = Path("all-records") / "format-matched-prompt"


def _make_result(hadm_id: int, *, error: str | None = None, truncated: bool = False) -> dict:
    """Build a generation result record (matches run_results.jsonl format, no scores)."""
    llm_result = (
        None
        if error is not None
        else {
            "text": "Generated text.",
            "input_tokens": 1000,
            "output_tokens": 200,
            "cached_tokens": None,
            "model": "gemma-4-e4b",
            "latency_ms": 1500.0,
            "truncated": truncated,
        }
    )
    return {
        "hadm_id": hadm_id,
        "template_name": "unformatted",
        "prompt": {"system": "sys", "user": "usr"},
        "llm_result": llm_result,
        "error": error,
        "timestamp": "2026-04-01T00:00:00Z",
    }


_DEFAULT_SCORES = {"rouge_1": 0.4, "rouge_2": 0.1, "rouge_l": 0.3}


def _scored(rec: dict, scores: dict[str, float] | None = None) -> ScoredResult:
    """Wrap a generation record dict as a ScoredResult with optional scores."""
    return ScoredResult(generation=ResultRecord.from_dict(rec), scores=scores or {})


def _eval(hadm_id: int, scores: dict[str, float]) -> dict:
    """An eval results row."""
    return {"hadm_id": hadm_id, "scores": scores, "coverage": None}


def _meta_for(cli_parameters: CliParameters) -> dict:
    return {"cli_parameters": asdict(cli_parameters), "status": "completed"}


class TestDiscoverModelDirs:
    def test_finds_model_dirs(self, tmp_path):
        meta = make_meta()
        write_model_dir(tmp_path / RUN_DIR, "gemma_4_e4b", meta)
        write_model_dir(tmp_path / RUN_DIR, "qwen_3_5_4b", meta)
        (tmp_path / "not_a_model").mkdir()

        assert discover_model_dirs(tmp_path) == [
            tmp_path / RUN_DIR / "qwen_3_5_4b",
            tmp_path / RUN_DIR / "gemma_4_e4b",
        ]

    def test_prefix_filter(self, tmp_path):
        write_model_dir(tmp_path / RUN_DIR, "gemma_4_e4b", make_meta())
        write_model_dir(tmp_path / "smoke" / "format-matched-prompt", "qwen_3_5_4b", make_meta())

        model_dirs = discover_model_dirs(tmp_path, prefix="gemma")
        assert len(model_dirs) == 1
        assert model_dirs[0].name == "gemma_4_e4b"


class TestLoadResultRecords:
    def test_duplicate_hadm_ids_dedup_keep_last(self, tmp_path):
        # run_results.jsonl is append-only. A resumed run or a manual re-run
        # can produce multiple lines for the same admission. The loader must
        # return one record per hadm_id, keeping the last occurrence.
        first = _make_result(100)
        first["llm_result"]["text"] = "first generation"
        second = _make_result(100)
        second["llm_result"]["text"] = "second generation"
        third = _make_result(200)
        model_dir = write_model_dir(tmp_path, "gemma_4_e4b", make_meta(), [first, second, third])

        loaded = load_result_records(model_dir)
        assert [r.hadm_id for r in loaded] == [100, 200]
        # The kept record for 100 is the second (last) occurrence.
        loaded_by_hadm = {r.hadm_id: r for r in loaded}
        assert loaded_by_hadm[100].llm_result.text == "second generation"


class TestLoadGenerations:
    def test_keeps_truncated_and_surfaces_flag(self, tmp_path):
        # Truncated generations are kept, flagged True. Records with no llm_result are skipped.
        records = [
            _make_result(100),
            _make_result(200, truncated=True),
            _make_result(300, error="fail"),
        ]
        model_dir = write_model_dir(tmp_path, "gemma_4_e4b", make_meta(), records)

        gens = load_generations(model_dir)

        assert set(gens) == {100, 200}
        assert gens[100] == (False, "Generated text.")
        assert gens[200] == (True, "Generated text.")


class TestGetResultRecord:
    def test_finds_record(self, tmp_path):
        model_dir = write_model_dir(
            tmp_path, "gemma_4_e4b", make_meta(), [_make_result(100), _make_result(200)]
        )

        rec = get_result_record(model_dir, 200)
        assert rec is not None
        assert rec.hadm_id == 200


class TestRunCliParameters:
    @pytest.mark.parametrize(
        "cli_parameters",
        [
            CliParameters(model="gemma_4_e4b", prompt="format-matched", oracle_n=2),
            CliParameters(model="gemma_4_e4b", prompt="detailed", deduplication="none"),
            CliParameters(model="qwen_3_5_4b", prompt="unformatted", smoke=True),
        ],
    )
    def test_reads_back_the_cli_parameters_a_run_was_generated_with(self, tmp_path, cli_parameters):
        write_model_dir(tmp_path / RUN_DIR, "gemma_4_e4b", _meta_for(cli_parameters))
        assert run_cli_parameters(tmp_path / RUN_DIR) == cli_parameters


class TestExtractRunLabel:
    def test_extract(self):
        assert extract_run_label(make_meta("qwen_3_5_4b")) == (
            "all-records/format-matched-prompt",
            "qwen_3_5_4b",
        )


class TestBuildPerAdmissionDfColumns:
    def test_success_records(self):
        records = [
            _scored(_make_result(100), _DEFAULT_SCORES),
            _scored(_make_result(200), _DEFAULT_SCORES),
        ]
        df = build_per_admission_df([(make_meta(), records)])

        assert len(df) == 2
        assert list(df["hadm_id"]) == [100, 200]
        assert "rouge_1" in df.columns
        assert "input_tokens" in df.columns
        assert df["error"].isna().all()

    def test_failed_record_has_null_fields(self):
        df = build_per_admission_df([(make_meta(), [_scored(_make_result(100, error="timeout"))])])
        assert df.iloc[0]["error"] == "timeout"
        assert pd.isna(df.iloc[0].get("input_tokens"))
        assert pd.isna(df.iloc[0].get("cached_tokens"))

    def test_mixed_success_and_failure(self):
        """Failed records get NaN score columns when mixed with successes."""
        records = [
            _scored(_make_result(100), _DEFAULT_SCORES),
            _scored(_make_result(200, error="timeout")),
        ]
        df = build_per_admission_df([(make_meta(), records)])

        assert len(df) == 2
        assert "rouge_1" in df.columns
        assert df.iloc[0]["rouge_1"] == 0.4
        assert pd.isna(df.iloc[1]["rouge_1"])

    def test_score_columns_from_data_only(self):
        """Score columns come only from the data, with no hardcoded metric names.

        Keys pass through verbatim, including slash-style names.
        """
        scored = _scored(_make_result(100), {"bertscore_f1": 0.9, "sapbert/f1": 0.85})
        df = build_per_admission_df([(make_meta(), [scored])])

        assert "bertscore_f1" in df.columns
        assert "sapbert/f1" in df.columns
        assert df.iloc[0]["sapbert/f1"] == pytest.approx(0.85)
        assert "rouge_1" not in df.columns

    def test_cached_tokens_extracted(self):
        rec = _make_result(100)
        rec["llm_result"]["cached_tokens"] = 400
        df = build_per_admission_df([(make_meta(), [_scored(rec)])])

        assert df.iloc[0]["cached_tokens"] == 400

    def test_truncated_flagged_and_kept(self):
        """Truncated rows are kept with the flag True. Error and success
        records read False."""
        records = [
            _scored(_make_result(100), _DEFAULT_SCORES),
            _scored(_make_result(200, truncated=True), _DEFAULT_SCORES),
            _scored(_make_result(300, error="fail")),
        ]
        df = build_per_admission_df([(make_meta(), records)])

        assert len(df) == 3
        assert "truncated" in df.columns
        by_id = df.set_index("hadm_id")["truncated"]
        assert not by_id[100]
        assert by_id[200]
        assert not by_id[300]


class TestBuildSummaryDf:
    def test_single_run(self):
        meta = make_meta("llama_3_2_3b")
        records = [_scored(_make_result(100)), _scored(_make_result(200))]

        df = build_summary_df([(meta, records)])
        assert len(df) == 1
        assert df.iloc[0]["run"] == "all-records/format-matched-prompt"
        assert df.iloc[0]["model"] == "llama_3_2_3b"
        assert df.iloc[0]["n_admissions"] == 2
        assert df.iloc[0]["n_errors"] == 0

    def test_errors_excluded_from_scores(self):
        meta = make_meta()
        records = [_scored(_make_result(100)), _scored(_make_result(200, error="fail"))]

        df = build_summary_df([(meta, records)])
        assert df.iloc[0]["n_admissions"] == 1
        assert df.iloc[0]["n_errors"] == 1

    def test_truncated_counted_separately_and_excluded_from_completed_columns(self):
        """Truncated records get their own count and do not pollute the token
        means and totals with their runaway output length."""
        meta = make_meta()
        trunc = _make_result(300, truncated=True)
        trunc["llm_result"]["output_tokens"] = 4096  # the runaway cap value
        records = [
            _scored(_make_result(100), _DEFAULT_SCORES),
            _scored(_make_result(200), _DEFAULT_SCORES),
            _scored(trunc),
        ]

        df = build_summary_df([(meta, records)])
        assert df.iloc[0]["n_truncated"] == 1
        assert df.iloc[0]["n_errors"] == 0
        assert df.iloc[0]["n_admissions"] == 2
        # Both successes are output_tokens=200. The 4096 runaway is excluded.
        assert df.iloc[0]["output_tokens_mean"] == 200
        assert df.iloc[0]["output_tokens_completed_total"] == 400

    def test_multiple_runs(self):
        runs = [
            (make_meta("llama_3_2_3b"), [_scored(_make_result(100))]),
            (make_meta("qwen_3_5_4b"), [_scored(_make_result(200))]),
        ]
        df = build_summary_df(runs)
        assert len(df) == 2


class TestBuildPerAdmissionDf:
    def test_basic(self):
        meta = make_meta("llama_3_2_3b")
        records = [_scored(_make_result(100)), _scored(_make_result(200))]

        df = build_per_admission_df([(meta, records)])
        assert len(df) == 2
        assert list(df["hadm_id"]) == [100, 200]
        assert df.iloc[0]["run"] == "all-records/format-matched-prompt"
        assert df.iloc[0]["model"] == "llama_3_2_3b"

    def test_error_record_included(self):
        meta = make_meta()
        records = [_scored(_make_result(100, error="fail"))]

        df = build_per_admission_df([(meta, records)])
        assert len(df) == 1
        assert df.iloc[0]["error"] == "fail"

    def test_truncated_record_flagged(self):
        meta = make_meta()
        records = [
            _scored(_make_result(100)),
            _scored(_make_result(200, truncated=True)),
        ]

        df = build_per_admission_df([(meta, records)])
        assert "truncated" in df.columns
        by_id = df.set_index("hadm_id")["truncated"]
        assert not by_id[100]
        assert by_id[200]


class TestLoadReferenceBhc:
    def test_loads_matching_records(self, tmp_path):
        bhc_path = tmp_path / "bhc.jsonl"
        lines = [
            json.dumps({"hadm_id": 100, "bhc_text": "Course for 100", "bhc_char_count": 14}),
            json.dumps({"hadm_id": 200, "bhc_text": "Course for 200", "bhc_char_count": 14}),
            json.dumps({"hadm_id": 300, "bhc_text": "Course for 300", "bhc_char_count": 14}),
        ]
        bhc_path.write_text("\n".join(lines) + "\n")

        result = load_reference_bhc(bhc_path, hadm_ids={100, 300})
        assert len(result) == 2
        assert result[100] == "Course for 100"
        assert result[300] == "Course for 300"
        assert 200 not in result


class TestLoadEvalScores:
    def test_loads_scores(self, tmp_path):
        evals = [
            _eval(100, {"rouge_1": 0.4, "rouge_l": 0.3}),
            _eval(200, {"rouge_1": 0.5, "rouge_l": 0.4}),
        ]
        model_dir = write_model_dir(tmp_path, "gemma_4_e4b", make_meta(), evals=evals)

        result = load_eval_scores(model_dir)
        assert result[100] == {"rouge_1": 0.4, "rouge_l": 0.3}
        assert result[200] == {"rouge_1": 0.5, "rouge_l": 0.4}


class TestLoadResultRecordsWithEvalScores:
    def test_merges_eval_scores_into_records(self, tmp_path):
        records = [_make_result(100), _make_result(200)]
        evals = [_eval(100, {"rouge_1": 0.4}), _eval(200, {"rouge_1": 0.5})]
        model_dir = write_model_dir(tmp_path, "gemma_4_e4b", make_meta(), records, evals)

        loaded = load_result_records(model_dir)
        assert loaded[0].scores == {"rouge_1": 0.4}
        assert loaded[1].scores == {"rouge_1": 0.5}


class TestLoadRunDf:
    def test_reads_the_cohort_file_from_the_cohorts_dir(self, tmp_path, monkeypatch):
        # Run records store the cohort's bare file name. The cohort lives in
        # the data repo's cohorts directory.
        seen: list[Path] = []
        monkeypatch.setattr(results_reader, "COHORTS_DIR", tmp_path / "cohorts")
        monkeypatch.setattr(
            results_reader, "load_cohort", lambda path: seen.append(path) or object()
        )
        monkeypatch.setattr(results_reader, "attach_cohort", lambda df, _cohort: df)
        meta = make_meta()
        write_model_dir(tmp_path / RUN_DIR, "gemma_4_e4b", meta, [_make_result(100)])

        load_run_df(tmp_path / RUN_DIR)

        assert seen == [tmp_path / "cohorts" / "experiment_cohort.json"]


class TestDiscoverRuns:
    """A run is the parent of each model directory."""

    def test_returns_each_run_once(self, tmp_path):
        meta = make_meta()
        unformatted_run = tmp_path / "all-records" / "unformatted-prompt"
        write_model_dir(tmp_path / RUN_DIR, "gemma_4_e4b", meta)
        write_model_dir(tmp_path / RUN_DIR, "qwen_3_5_4b", meta)
        write_model_dir(unformatted_run, "gemma_4_e4b", meta)

        assert discover_runs(tmp_path) == [tmp_path / RUN_DIR, unformatted_run]


class TestLoadRun:
    def test_loads_all_models(self, tmp_path):
        meta = make_meta()
        for model_key in ("gemma_4_e4b", "llama_3_2_3b", "qwen_3_5_4b"):
            write_model_dir(
                tmp_path / RUN_DIR, model_key, meta, [_make_result(100), _make_result(200)]
            )

        loaded = load_run(tmp_path / RUN_DIR)
        assert len(loaded) == 3
        for meta_loaded, scored in loaded:
            assert meta_loaded["cli_parameters"]["prompt"] == "format-matched"
            assert len(scored) == 2

    def test_skips_models_without_run_record(self, tmp_path):
        meta = make_meta()
        write_model_dir(tmp_path / RUN_DIR, "gemma_4_e4b", meta)
        # A sibling directory with no run record.
        (tmp_path / RUN_DIR / "not_a_model").mkdir()

        loaded = load_run(tmp_path / RUN_DIR)
        assert len(loaded) == 1

    def test_other_runs_not_included(self, tmp_path):
        meta = make_meta()
        write_model_dir(tmp_path / RUN_DIR, "gemma_4_e4b", meta)
        write_model_dir(tmp_path / "all-records" / "unformatted-prompt", "qwen_3_5_4b", meta)

        loaded = load_run(tmp_path / RUN_DIR)
        assert len(loaded) == 1


class TestGetResultRecordWithEvalScores:
    def test_merges_eval_scores(self, tmp_path):
        model_dir = write_model_dir(
            tmp_path,
            "gemma_4_e4b",
            make_meta(),
            [_make_result(100)],
            [_eval(100, {"rouge_1": 0.4})],
        )

        rec = get_result_record(model_dir, 100)
        assert rec is not None
        assert rec.scores == {"rouge_1": 0.4}


def _make_result_with_timings(hadm_id: int, timings: dict | None) -> dict:
    """A success record whose llm_result has a raw timings object."""
    rec = _make_result(hadm_id)
    rec["llm_result"]["timings"] = timings
    return rec


_SAMPLE_TIMINGS = {
    "prompt_ms": 200.0,
    "predicted_ms": 40.0,
    "prompt_per_second": 100.0,
    "predicted_per_second": 50.0,
}


class TestTimingBreakdownInReader:
    """The reader DataFrames have the prefill and decode fields."""

    def test_per_admission_has_breakdown_columns(self) -> None:
        scored = _scored(_make_result_with_timings(100, _SAMPLE_TIMINGS))
        df = build_per_admission_df([(make_meta(), [scored])])
        assert df.iloc[0]["prefill_ms"] == 200.0
        assert df.iloc[0]["decode_ms"] == 40.0
        assert df.iloc[0]["prefill_tok_s"] == 100.0
        assert df.iloc[0]["decode_tok_s"] == 50.0

    def test_per_admission_row_without_timings_is_none(self) -> None:
        """A record with no timings yields None breakdown fields."""
        scored = _scored(_make_result(100))
        df = build_per_admission_df([(make_meta(), [scored])])
        assert df.iloc[0]["prefill_ms"] is None

    def test_build_per_admission_df_has_breakdown(self) -> None:
        scored = _scored(_make_result_with_timings(100, _SAMPLE_TIMINGS))
        df = build_per_admission_df([(make_meta(), [scored])])
        assert df.iloc[0]["prefill_ms"] == 200.0
        assert df.iloc[0]["decode_tok_s"] == 50.0

    def test_summary_has_breakdown_means(self) -> None:
        a = _scored(_make_result_with_timings(100, _SAMPLE_TIMINGS))
        b = _scored(
            _make_result_with_timings(
                200,
                {
                    "prompt_ms": 100.0,
                    "predicted_ms": 20.0,
                    "prompt_per_second": 200.0,
                    "predicted_per_second": 100.0,
                },
            )
        )
        df = build_summary_df([(make_meta(), [a, b])])
        assert df.iloc[0]["prefill_ms_mean"] == pytest.approx(150.0)
        assert df.iloc[0]["decode_ms_mean"] == pytest.approx(30.0)
        assert df.iloc[0]["prefill_tok_s_mean"] == pytest.approx(150.0)
        assert df.iloc[0]["decode_tok_s_mean"] == pytest.approx(75.0)

    def test_summary_means_none_for_a_run_without_timings(self) -> None:
        """A run with no timings reports None means."""
        scored = _scored(_make_result(100))
        df = build_summary_df([(make_meta(), [scored])])
        assert df.iloc[0]["prefill_ms_mean"] is None


class TestMergeRunPair:
    def test_shared_metric_columns_get_base_and_treat_suffixes(self) -> None:
        base = pd.DataFrame({"hadm_id": [1, 2], "model": ["m", "m"], "recall": [0.5, 0.6]})
        treat = pd.DataFrame({"hadm_id": [1, 2], "model": ["m", "m"], "recall": [0.7, 0.8]})
        merged = merge_run_pair(base, treat)
        assert list(merged["recall__base"]) == [0.5, 0.6]
        assert list(merged["recall__treat"]) == [0.7, 0.8]


def _metric_df(rows: list[tuple[int, float, float, float]]) -> pd.DataFrame:
    """Build a per-admission DataFrame with the three PRIMARY_METRICS columns.

    Each row is (hadm_id, rougeL/f1, sapbert/f1, scispacy-clinical/f1). A
    hadm_id may repeat to represent multiple models for one admission.
    """
    records = []
    for h, r, s, c in rows:
        records.append({"hadm_id": h, "rougeL/f1": r, "sapbert/f1": s, "scispacy-clinical/f1": c})
    return pd.DataFrame(records)


class TestWorstByCompositeZ:
    """Rank admissions worst-first by the max-z composite across primary metrics."""

    def test_ranks_worst_first_with_driver_and_exact_value(self):
        df = _metric_df([(100, 0.2, 0.3, 0.25), (200, 0.8, 0.7, 0.6), (300, 0.5, 0.5, 0.5)])
        picks = worst_by_composite_z(df, 3)
        assert [hadm_id for hadm_id, _z, _driver in picks] == [100, 300, 200]
        hadm_id, composite_z, driver = picks[0]
        assert hadm_id == 100
        assert composite_z == pytest.approx(1.3587324409735149)
        assert driver == "scispacy-clinical/f1"

    def test_respects_n(self):
        df = _metric_df([(100, 0.2, 0.3, 0.25), (200, 0.8, 0.7, 0.6), (300, 0.5, 0.5, 0.5)])
        picks = worst_by_composite_z(df, 1)
        assert [hadm_id for hadm_id, _z, _driver in picks] == [100]

    def test_averages_models_per_admission(self):
        # hadm 100 has two models averaging to 0.2/0.3/0.25, matching the
        # single-model admission that the exact test ranks worst.
        df = _metric_df(
            [
                (100, 0.1, 0.2, 0.2),
                (100, 0.3, 0.4, 0.3),
                (200, 0.8, 0.7, 0.6),
                (300, 0.5, 0.5, 0.5),
            ]
        )
        picks = worst_by_composite_z(df, 3)
        assert [hadm_id for hadm_id, _z, _driver in picks] == [100, 300, 200]
        assert picks[0][1] == pytest.approx(1.3587324409735149)

    def test_ignores_flat_metric(self):
        # sapbert is identical across admissions: zero spread, undefined z,
        # so it is dropped and never becomes a driver.
        df = _metric_df([(100, 0.1, 0.5, 0.1), (200, 0.9, 0.5, 0.9)])
        picks = worst_by_composite_z(df, 2)
        assert [hadm_id for hadm_id, _z, _driver in picks] == [100, 200]
        assert all(driver != "sapbert/f1" for _h, _z, driver in picks)

    def test_excludes_admission_with_all_nan_metrics(self):
        # A fully-errored admission is NaN on every metric. It must be dropped
        # from the worklist without reaching idxmax on an all-NaN row, which
        # pandas deprecates and will raise on in a future version.
        nan = float("nan")
        df = _metric_df([(100, 0.1, 0.1, 0.1), (200, 0.9, 0.9, 0.9), (300, nan, nan, nan)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            picks = worst_by_composite_z(df, 5)
        assert [hadm_id for hadm_id, _z, _driver in picks] == [100, 200]


class TestRunPreparedStore:
    def test_reads_the_prepared_records_of_the_runs_deduplication(self, tmp_path, monkeypatch):
        prepared_dir = tmp_path / "prepared"
        cohort = "experiment_cohort.json"
        write_prepared(
            prepared_dir,
            cohort,
            "none",
            [make_bundle(hadm_id=1)],
            {},
            git_commit=None,
            git_dirty=False,
        )
        write_prepared(
            prepared_dir,
            cohort,
            "sentence",
            [make_bundle(hadm_id=2)],
            {},
            git_commit=None,
            git_dirty=False,
        )
        monkeypatch.setattr(results_reader, "PREPARED_DIR", prepared_dir)
        cli_parameters = CliParameters(
            model="gemma_4_e4b", prompt="format-matched", deduplication="none"
        )
        write_model_dir(tmp_path / RUN_DIR, "gemma_4_e4b", _meta_for(cli_parameters))

        store = run_prepared_store(tmp_path / RUN_DIR)

        assert [b.admission.HADM_ID for b in store.bundles()] == [1]
