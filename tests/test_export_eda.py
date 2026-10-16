"""Tests for scripts/pipeline/export_eda.py."""

import argparse
import json
from pathlib import Path

from bhc_experiment import results_reader
from bhc_experiment.config import CliParameters

from .conftest import make_meta, write_model_dir


def _generation(hadm_id: int, model_key: str) -> dict:
    return {
        "hadm_id": hadm_id,
        "template_name": "x",
        "prompt": {"system": "s", "user": "u"},
        "llm_result": {
            "text": "out",
            "input_tokens": 500,
            "output_tokens": 100,
            "cached_tokens": None,
            "model": model_key,
            "latency_ms": 1500.0,
        },
        "error": None,
        "timestamp": "2026-04-01T00:00:00Z",
    }


class TestExportSummary:
    """End-to-end: disk batch -> markdown file."""

    def _setup(self, tmp_path: Path, admission_dict, monkeypatch) -> None:
        """Build one run with two models and two admissions under a synthetic
        results root, at ``tmp_path/results/all-records/format-matched-prompt``.
        """
        results_root = tmp_path / "results"
        monkeypatch.setattr(results_reader, "COHORTS_DIR", tmp_path)
        cohort_path = tmp_path / CliParameters(prompt="format-matched").cohort_file
        cohort_path.write_text(
            json.dumps(
                {
                    "metadata": {},
                    "admissions": [
                        admission_dict(hadm_id=100, bhc_category="narrative"),
                        admission_dict(hadm_id=200, bhc_category="domain_organized"),
                    ],
                }
            )
        )
        run_dir = results_root / "all-records" / "format-matched-prompt"
        scores = {100: {"sapbert/f1": 0.8}, 200: {"sapbert/f1": 0.7}}
        evals = [{"hadm_id": h, "scores": s, "coverage": None} for h, s in scores.items()]
        for model_key in ("gemma_4_e4b", "llama_3_2_3b"):
            results = [_generation(hadm_id, model_key) for hadm_id in scores]
            write_model_dir(run_dir, model_key, make_meta(model_key), results, evals)

    def test_writes_markdown_with_series_rows(self, tmp_path, admission_dict, monkeypatch):
        import export_eda

        self._setup(tmp_path, admission_dict, monkeypatch)

        out_path = tmp_path / "out.md"
        args = argparse.Namespace(
            run_dir=tmp_path / "results" / "all-records" / "format-matched-prompt",
            stratify_by=None,
            output=str(out_path),
        )
        export_eda.export_summary(args)

        content = out_path.read_text()
        assert "# Summary: all-records/format-matched-prompt" in content
        assert "- Series: 2" in content
        assert "- Admissions in the run: 2" in content
        assert "| series_label | n | sapbert/f1 |" in content
        assert "gemma_4_e4b" in content
        assert "llama_3_2_3b" in content
