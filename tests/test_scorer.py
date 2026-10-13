"""Tests for bhc_experiment.scorer module."""

import json

import yaml

from bhc_experiment.results import (
    EVAL_RECORD_FILENAME,
    EVAL_RESULTS_FILENAME,
)
from bhc_experiment.scorer import load_generated_texts, score_model_dir

from .conftest import write_model_dir

RUN_RECORD = {"status": "completed"}


def _make_gen_record(hadm_id, *, error=None, truncated=False):
    llm_result = (
        None
        if error is not None
        else {
            "text": f"Generated BHC for {hadm_id}.",
            "input_tokens": 1000,
            "output_tokens": 200,
            "cached_tokens": None,
            "model": "test-model",
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


class TestLoadGeneratedTexts:
    def test_loads_successful_records(self, tmp_path):
        model_dir = write_model_dir(
            tmp_path, "gemma_4_e4b", RUN_RECORD, [_make_gen_record(100), _make_gen_record(200)]
        )

        texts, errors = load_generated_texts(model_dir)
        assert texts == {
            100: "Generated BHC for 100.",
            200: "Generated BHC for 200.",
        }
        assert errors == {}

    def test_skips_error_records(self, tmp_path):
        records = [
            _make_gen_record(100),
            _make_gen_record(200, error="LLM timeout"),
            _make_gen_record(300),
        ]
        model_dir = write_model_dir(tmp_path, "gemma_4_e4b", RUN_RECORD, records)

        texts, errors = load_generated_texts(model_dir)
        assert len(texts) == 2
        assert 200 not in texts
        assert errors == {200: "LLM timeout"}

    def test_skips_truncated_records(self, tmp_path):
        records = [
            _make_gen_record(100),
            _make_gen_record(200, truncated=True),
            _make_gen_record(300),
        ]
        model_dir = write_model_dir(tmp_path, "gemma_4_e4b", RUN_RECORD, records)

        texts, errors = load_generated_texts(model_dir)
        assert set(texts) == {100, 300}
        assert 200 not in texts
        assert "truncat" in errors[200].lower()


class TestScoreRunDir:
    def _mock_scorers(self):
        def fn(cands, _refs):
            return [{"test_metric": 0.5}] * len(cands)

        return {"test_metric": fn}

    def _provenance(self):
        return {
            "coverage": {},
            "git_commit": "c" * 40,
            "git_dirty": False,
        }

    def _ground_truth(self, *hadm_ids):
        return {hid: f"Reference BHC for {hid}." for hid in hadm_ids}

    def _read_eval_lines(self, model_dir):
        eval_path = model_dir / EVAL_RESULTS_FILENAME
        if not eval_path.exists():
            return []
        with eval_path.open() as f:
            return [json.loads(line) for line in f]

    def test_scores_all_admissions(self, tmp_path):
        model_dir = write_model_dir(
            tmp_path, "gemma_4_e4b", RUN_RECORD, [_make_gen_record(100), _make_gen_record(200)]
        )

        score_model_dir(
            model_dir,
            self._mock_scorers(),
            self._ground_truth(100, 200),
            **self._provenance(),
        )

        lines = self._read_eval_lines(model_dir)
        assert len(lines) == 2
        hadm_ids = {line["hadm_id"] for line in lines}
        assert hadm_ids == {100, 200}
        for line in lines:
            assert line["scores"]["test_metric"] == 0.5

    def test_skips_error_records(self, tmp_path):
        records = [
            _make_gen_record(100),
            _make_gen_record(200, error="fail"),
            _make_gen_record(300),
        ]
        model_dir = write_model_dir(tmp_path, "gemma_4_e4b", RUN_RECORD, records)

        score_model_dir(
            model_dir,
            self._mock_scorers(),
            self._ground_truth(100, 200, 300),
            **self._provenance(),
        )

        lines = self._read_eval_lines(model_dir)
        assert len(lines) == 2
        hadm_ids = {line["hadm_id"] for line in lines}
        assert hadm_ids == {100, 300}

    def test_attaches_coverage_when_provided(self, tmp_path):
        model_dir = write_model_dir(
            tmp_path, "gemma_4_e4b", RUN_RECORD, [_make_gen_record(100), _make_gen_record(200)]
        )

        score_model_dir(
            model_dir,
            self._mock_scorers(),
            self._ground_truth(100, 200),
            **{
                **self._provenance(),
                "coverage": {100: {"ceiling": 0.4, "n_ref": 10, "n_overlap": 4}},
            },
        )

        lines = {line["hadm_id"]: line for line in self._read_eval_lines(model_dir)}
        assert lines[100]["coverage"] == {"ceiling": 0.4, "n_ref": 10, "n_overlap": 4}
        # 200 has no coverage entry, so it is written as None.
        assert lines[200]["coverage"] is None

    def test_writes_the_eval_record_once_per_pass(self, tmp_path):
        model_dir = write_model_dir(
            tmp_path,
            "gemma_4_e4b",
            RUN_RECORD,
            [_make_gen_record(100), _make_gen_record(200, error="fail")],
        )

        score_model_dir(
            model_dir, self._mock_scorers(), self._ground_truth(100, 200), **self._provenance()
        )

        eval_record = yaml.safe_load((model_dir / EVAL_RECORD_FILENAME).read_text())
        assert eval_record["git_commit"] == "c" * 40
        assert eval_record["n_scored"] == 1
        assert eval_record["n_excluded"] == 1
        assert "scored_at" in eval_record
        (line,) = self._read_eval_lines(model_dir)
        assert set(line) == {"hadm_id", "scores", "coverage"}

    def test_rescoring_replaces_the_whole_file(self, tmp_path):
        model_dir = write_model_dir(tmp_path, "gemma_4_e4b", RUN_RECORD, [_make_gen_record(100)])
        stale = {"hadm_id": 999, "scores": {"old/f1": 0.1}, "coverage": None}
        (model_dir / EVAL_RESULTS_FILENAME).write_text(json.dumps(stale) + "\n")

        score_model_dir(
            model_dir, self._mock_scorers(), self._ground_truth(100), **self._provenance()
        )

        assert [line["hadm_id"] for line in self._read_eval_lines(model_dir)] == [100]
