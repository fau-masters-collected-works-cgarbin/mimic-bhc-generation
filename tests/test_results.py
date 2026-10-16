"""Tests for bhc_experiment.results: experiment result persistence."""

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from bhc_experiment.config import DRY_SAMPLER, CliParameters, inference_for
from bhc_experiment.llm import LlmResult
from bhc_experiment.results import (
    RUN_RECORD_FILENAME,
    RUN_RESULTS_FILENAME,
    EvalRecord,
    ResultRecord,
    append_result,
    create_model_dir_and_record,
    finalize_run,
    load_completed_ids,
    mark_interrupted,
    model_dir_path,
)


class TestEvalRecord:
    def test_round_trips_scores_and_coverage(self):
        data = {
            "hadm_id": 100,
            "scores": {"scispacy-clinical/recall": 0.15},
            "coverage": {"ceiling": 0.5, "n_ref": 4, "n_overlap": 2},
        }
        record = EvalRecord.from_dict(data)
        assert record == EvalRecord(
            hadm_id=100,
            scores={"scispacy-clinical/recall": 0.15},
            coverage={"ceiling": 0.5, "n_ref": 4, "n_overlap": 2},
        )


def _make_record(hadm_id: int = 100, *, error: str | None = None) -> ResultRecord:
    """Build a ResultRecord with minimal required fields.

    Shared across test classes to avoid duplicating the same factory.
    """
    llm_result = (
        None
        if error is not None
        else LlmResult(
            text=f"Generated for {hadm_id}.",
            input_tokens=1000 + hadm_id,
            output_tokens=200,
            cached_tokens=None,
            model="google/gemma-4-e4b",
            latency_ms=2000.0,
        )
    )
    return ResultRecord(
        hadm_id=hadm_id,
        template_name="unformatted",
        prompt={"system": "sys", "user": f"Patient {hadm_id} data."},
        llm_result=llm_result,
        error=error,
        timestamp="2026-03-30T14:31:12Z",
    )


class TestModelDirPath:
    """results / <selection> / <prompt>-prompt / <model>"""

    def test_layout(self, tmp_path):
        cli_parameters = CliParameters(model="gemma_4_e4b", prompt="format-matched")
        path = model_dir_path(tmp_path, cli_parameters)
        assert path == tmp_path / "all-records" / "format-matched-prompt" / "gemma_4_e4b"


def _create_run(tmp_path: Path, *, total_admissions: int = 5, **overrides) -> Path:
    """Create a run directory and record under tmp_path with test defaults."""
    cli_parameters = CliParameters(model="gemma_4_e4b", prompt="format-matched")
    model_dir = model_dir_path(tmp_path, cli_parameters)
    fields = {
        "cli_parameters": cli_parameters,
        "total_admissions": total_admissions,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "prepared_commit": "b" * 40,
        "inputs": {"cohort_file": "test_cohort.json"},
    }
    fields.update(overrides)
    create_model_dir_and_record(model_dir, **fields)
    return model_dir


class TestCreateModelDir:
    def test_run_record_logs_the_models_own_sampler(self, tmp_path):
        """A Llama run that recorded the shared settings would hide that it ran with DRY."""
        cli_parameters = CliParameters(model="llama_3_2_3b", prompt="format-matched")
        model_dir = _create_run(tmp_path, cli_parameters=cli_parameters)
        run_record = yaml.safe_load((model_dir / RUN_RECORD_FILENAME).read_text())
        assert run_record["inference"]["dry_sampler"] == DRY_SAMPLER

    def test_creates_directory_and_run_record(self, tmp_path):
        model_dir = _create_run(tmp_path, total_admissions=200)

        run_record = yaml.safe_load((model_dir / RUN_RECORD_FILENAME).read_text())
        assert run_record["git_commit"] == "a" * 40
        assert run_record["git_dirty"] is False
        assert run_record["prepared_commit"] == "b" * 40
        # The model is a CLI parameter, so it appears only in that section.
        assert "model" not in run_record
        cli_parameters = CliParameters(model="gemma_4_e4b", prompt="format-matched")
        assert CliParameters(**run_record["cli_parameters"]) == cli_parameters
        assert run_record["inference"] == asdict(inference_for("gemma_4_e4b"))
        assert run_record["total_admissions"] == 200
        assert run_record["status"] == "running"
        assert run_record["completed_at"] is None
        assert run_record["completed_count"] == 0
        assert run_record["failed_count"] == 0
        assert "started_at" in run_record


class TestLoadCompletedIds:
    """load_completed_ids reads JSONL and returns hadm_id set."""

    def test_returns_hadm_ids(self, tmp_path):
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        append_result(model_dir, _make_record(hadm_id=100))
        append_result(model_dir, _make_record(hadm_id=200))
        append_result(model_dir, _make_record(hadm_id=300))

        ids = load_completed_ids(model_dir)
        assert ids == {100, 200, 300}

    def test_includes_failed_records(self, tmp_path):
        """A record with an error still counts as completed."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        append_result(model_dir, _make_record(hadm_id=100))
        append_result(model_dir, _make_record(hadm_id=200, error="timeout"))

        ids = load_completed_ids(model_dir)
        assert ids == {100, 200}


class TestFinalizeRun:
    """finalize_run sets status=completed and computes counts from JSONL."""

    def test_sets_completed_status(self, tmp_path):
        model_dir = _create_run(tmp_path, total_admissions=2)
        append_result(model_dir, _make_record(hadm_id=1))
        append_result(model_dir, _make_record(hadm_id=2))

        finalize_run(model_dir)

        run_record = yaml.safe_load((model_dir / RUN_RECORD_FILENAME).read_text())
        assert run_record["status"] == "completed"
        assert run_record["completed_at"] is not None
        assert run_record["completed_count"] == 2
        assert run_record["failed_count"] == 0

    def test_preserves_existing_run_record_fields(self, tmp_path):
        model_dir = _create_run(tmp_path, total_admissions=1, prepared_commit="c" * 40)
        append_result(model_dir, _make_record(hadm_id=1))
        finalize_run(model_dir)

        run_record = yaml.safe_load((model_dir / RUN_RECORD_FILENAME).read_text())
        assert run_record["prepared_commit"] == "c" * 40


class TestMarkInterrupted:
    """mark_interrupted sets status=interrupted and computes counts."""

    def test_sets_interrupted_status(self, tmp_path):
        model_dir = _create_run(tmp_path, total_admissions=10)
        append_result(model_dir, _make_record(hadm_id=1))

        mark_interrupted(model_dir)

        run_record = yaml.safe_load((model_dir / RUN_RECORD_FILENAME).read_text())
        assert run_record["status"] == "interrupted"
        assert run_record["completed_at"] is not None
        assert run_record["completed_count"] == 1
        assert run_record["failed_count"] == 0

    def test_no_results_yet(self, tmp_path):
        model_dir = _create_run(tmp_path, total_admissions=10)
        mark_interrupted(model_dir)

        run_record = yaml.safe_load((model_dir / RUN_RECORD_FILENAME).read_text())
        assert run_record["status"] == "interrupted"
        assert run_record["completed_count"] == 0
        assert run_record["failed_count"] == 0


class TestRunLifecycle:
    """End-to-end test: create -> append -> finalize -> reload."""

    def test_full_lifecycle(self, tmp_path):
        model_dir = _create_run(tmp_path, total_admissions=3)

        # Append results (2 success, 1 failure)
        append_result(model_dir, _make_record(hadm_id=100))
        append_result(model_dir, _make_record(hadm_id=200, error="timeout"))
        append_result(model_dir, _make_record(hadm_id=300))

        ids = load_completed_ids(model_dir)
        assert ids == {100, 200, 300}

        finalize_run(model_dir)

        # Verify run record
        run_record = yaml.safe_load((model_dir / RUN_RECORD_FILENAME).read_text())
        assert run_record["status"] == "completed"
        assert run_record["completed_count"] == 2
        assert run_record["failed_count"] == 1
        assert run_record["total_admissions"] == 3

        # Verify JSONL content
        lines = (model_dir / RUN_RESULTS_FILENAME).read_text().strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["hadm_id"] == 100
        assert first["llm_result"]["input_tokens"] == 1100
        failed = json.loads(lines[1])
        assert failed["error"] == "timeout"
        assert failed["llm_result"] is None

    def test_resume_after_interruption(self, tmp_path):
        model_dir = _create_run(tmp_path)
        append_result(model_dir, _make_record(hadm_id=100))
        append_result(model_dir, _make_record(hadm_id=200))
        mark_interrupted(model_dir)

        # Simulate resume: check what's done
        ids = load_completed_ids(model_dir)
        assert ids == {100, 200}

        run_record = yaml.safe_load((model_dir / RUN_RECORD_FILENAME).read_text())
        assert run_record["status"] == "interrupted"
        assert run_record["completed_count"] == 2
