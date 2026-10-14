"""Tests for scripts/pipeline/run_experiment.py helper functions."""

# ruff: noqa: SLF001 (testing private helper functions)

import pytest
import run_experiment
import yaml

from bhc_experiment.llm import LlmResult, ReasoningNotDisabledError
from bhc_experiment.prompt_builder import PromptResult
from bhc_experiment.results import RUN_RECORD_FILENAME, RUN_RESULTS_FILENAME, append_result

from .conftest import write_model_dir


class TestBuildResultRecord:
    """_build_result_record constructs a valid ResultRecord."""

    def test_success_record(self):
        """Successful pipeline outputs produce a complete record."""
        prompt_result = PromptResult(system="sys", user="usr", template_name="unformatted")
        llm_result = LlmResult(
            text="generated",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=None,
            model="test-model",
            latency_ms=500.0,
        )
        record = run_experiment._build_result_record(
            hadm_id=12345,
            prompt_result=prompt_result,
            llm_result=llm_result,
        )

        assert record.hadm_id == 12345
        assert record.template_name == "unformatted"
        assert record.prompt == {"system": "sys", "user": "usr"}
        assert record.llm_result.input_tokens == 100
        assert record.error is None

    def test_error_record(self):
        """Error record has None for llm_result."""
        prompt_result = PromptResult(system="", user="", template_name="unknown")

        record = run_experiment._build_result_record(
            hadm_id=12345,
            prompt_result=prompt_result,
            llm_result=None,
            error="ValueError: something broke",
        )

        assert record.hadm_id == 12345
        assert record.llm_result is None
        assert record.error == "ValueError: something broke"


class TestExistingRunAction:
    """existing_run_action: start, resume, or skip a run directory, given HEAD."""

    HEAD = "a" * 40

    def _record(
        self, model_dir, *, status: str, commit: str, dirty: bool = False, done: int = 0
    ) -> None:
        record = {"status": status, "git_commit": commit, "git_dirty": dirty, "total_admissions": 2}
        write_model_dir(model_dir.parent, model_dir.name, record)
        prompt = PromptResult(system="", user="", template_name="unformatted")
        for hadm_id in range(100, 100 + done):
            record = run_experiment._build_result_record(hadm_id, prompt, None, "failed")
            append_result(model_dir, record)

    def test_no_record_starts(self, tmp_path):
        assert run_experiment.existing_run_action(tmp_path / "gemma_4_e4b", self.HEAD) == "start"

    def test_completed_run_is_skipped(self, tmp_path):
        model_dir = tmp_path / "gemma_4_e4b"
        self._record(model_dir, status="completed", commit=self.HEAD)
        assert run_experiment.existing_run_action(model_dir, self.HEAD) == "skip"

    def test_partial_run_at_the_same_commit_resumes(self, tmp_path):
        model_dir = tmp_path / "gemma_4_e4b"
        self._record(model_dir, status="interrupted", commit=self.HEAD)
        assert run_experiment.existing_run_action(model_dir, self.HEAD) == "resume"

    def test_partial_run_at_another_commit_refuses(self, tmp_path):
        # Resuming would mix generations from two versions of the code.
        model_dir = tmp_path / "gemma_4_e4b"
        self._record(model_dir, status="running", commit="b" * 40)
        with pytest.raises(SystemExit, match="Delete"):
            run_experiment.existing_run_action(model_dir, self.HEAD)

    def test_partial_run_from_a_dirty_tree_refuses(self, tmp_path):
        # Its commit does not pin the code it ran, so a resume could mix versions.
        model_dir = tmp_path / "gemma_4_e4b"
        self._record(model_dir, status="interrupted", commit=self.HEAD, dirty=True)
        with pytest.raises(SystemExit, match="uncommitted"):
            run_experiment.existing_run_action(model_dir, self.HEAD)

    def test_partial_run_with_every_admission_recorded_refuses(self, tmp_path):
        # Every admission is recorded, and failed ones count as done, so a resume
        # has nothing left to try and the run would stay interrupted forever.
        model_dir = tmp_path / "gemma_4_e4b"
        self._record(model_dir, status="interrupted", commit=self.HEAD, done=2)
        with pytest.raises(SystemExit, match="Delete"):
            run_experiment.existing_run_action(model_dir, self.HEAD)


class TestDecideRunOutcome:
    """_decide_run_outcome chooses between interrupted and completed."""

    def test_all_admissions_in_run_failed_interrupts(self):
        outcome = run_experiment._decide_run_outcome(
            remaining_count=10, new_failures=10, completed_count=10, failed_count=10
        )
        assert outcome == "interrupted"

    def test_partial_failure_completes(self):
        outcome = run_experiment._decide_run_outcome(
            remaining_count=10, new_failures=3, completed_count=10, failed_count=3
        )
        assert outcome == "completed"

    def test_all_succeeded_completes(self):
        outcome = run_experiment._decide_run_outcome(
            remaining_count=10, new_failures=0, completed_count=10, failed_count=0
        )
        assert outcome == "completed"

    def test_resume_no_remaining_all_prior_errored_stays_interrupted(self):
        """Re-running a fully-errored run should not silently flip to completed.

        Otherwise ``finalize_run`` marks the run completed even though no
        admission ever succeeded.
        """
        outcome = run_experiment._decide_run_outcome(
            remaining_count=0, new_failures=0, completed_count=10, failed_count=10
        )
        assert outcome == "interrupted"

    def test_resume_no_remaining_mixed_completes(self):
        """Resume with prior partial failures + successes finalizes normally."""
        outcome = run_experiment._decide_run_outcome(
            remaining_count=0, new_failures=0, completed_count=10, failed_count=3
        )
        assert outcome == "completed"

    def test_resume_no_remaining_all_clean_completes(self):
        outcome = run_experiment._decide_run_outcome(
            remaining_count=0, new_failures=0, completed_count=10, failed_count=0
        )
        assert outcome == "completed"

    def test_empty_cohort_completes(self):
        """Empty hadm_ids should not be flagged as systemic failure."""
        outcome = run_experiment._decide_run_outcome(
            remaining_count=0, new_failures=0, completed_count=0, failed_count=0
        )
        assert outcome == "completed"


class TestProcessAdmissions:
    def test_reasoning_stops_the_run_without_recording_the_admission(self, tmp_path, monkeypatch):
        # A server that reasons would do it on every admission, so the run stops
        # and a resume retries the admission once the server is fixed.
        record = {"status": "running", "total_admissions": 1}
        model_dir = write_model_dir(tmp_path, "gemma_4_e4b", record)

        def reasoning_call(*_args):
            msg = "reasoned"
            raise ReasoningNotDisabledError(msg)

        monkeypatch.setattr(run_experiment, "create_llm_client", lambda: None)
        monkeypatch.setattr(run_experiment, "build_prompt", lambda _bundle, _variant: None)
        monkeypatch.setattr(run_experiment, "call_llm", reasoning_call)

        with pytest.raises(ReasoningNotDisabledError):
            run_experiment.process_admissions(
                [100],
                set(),
                variant="unformatted",
                model_alias="gemma-4-e4b",
                model_dir=model_dir,
                load_bundle=lambda _hadm_id: None,
            )

        assert yaml.safe_load((model_dir / RUN_RECORD_FILENAME).read_text())["status"] == (
            "interrupted"
        )
        assert not (model_dir / RUN_RESULTS_FILENAME).exists()
