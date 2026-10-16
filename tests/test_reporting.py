"""Tests for bhc_experiment.reporting."""

from dataclasses import asdict

import pandas as pd
import pytest

from bhc_experiment.config import CliParameters
from bhc_experiment.reporting import require_full_input, require_oracle_selection, stratum_summary

from .conftest import write_model_dir


class TestRequireFullInput:
    """Scripts that measure the ceiling over every record refuse an oracle run."""

    def _run(self, tmp_path, cli_parameters: CliParameters):
        run_dir = tmp_path / "format-matched-prompt"
        write_model_dir(run_dir, "gemma_4_e4b", {"cli_parameters": asdict(cli_parameters)})
        return run_dir

    def test_oracle_run_raises(self, tmp_path):
        run_dir = self._run(
            tmp_path, CliParameters(model="gemma_4_e4b", prompt="format-matched", oracle_n=0)
        )
        with pytest.raises(ValueError, match="is an oracle run"):
            require_full_input(run_dir)

    def test_all_records_run_passes(self, tmp_path):
        run_dir = self._run(tmp_path, CliParameters(model="gemma_4_e4b", prompt="format-matched"))
        require_full_input(run_dir)


class TestRequireOracleSelection:
    def _run(self, tmp_path, cli_parameters: CliParameters):
        run_dir = tmp_path / "format-matched-prompt"
        write_model_dir(run_dir, "gemma_4_e4b", {"cli_parameters": asdict(cli_parameters)})
        return run_dir

    def test_all_records_run_raises(self, tmp_path):
        run_dir = self._run(tmp_path, CliParameters(model="gemma_4_e4b", prompt="format-matched"))
        with pytest.raises(ValueError, match="is not an oracle run"):
            require_oracle_selection(run_dir)

    def test_oracle_run_passes(self, tmp_path):
        run_dir = self._run(
            tmp_path, CliParameters(model="gemma_4_e4b", prompt="format-matched", oracle_n=0)
        )
        require_oracle_selection(run_dir)


class TestStratumSummary:
    def _sample(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": ["m1", "m1", "m1", "m2", "m2"],
                "token_stratum": ["4K", "4K", "8K", "4K", "4K"],
                "scispacy-clinical/f1": [0.4, 0.6, 0.9, 0.2, None],
            }
        )

    def test_one_row_per_model_stratum(self):
        out = stratum_summary(self._sample(), "scispacy-clinical/f1", seed=1)
        keys = set(zip(out["model"], out["token_stratum"], strict=True))
        assert keys == {("m1", "4K"), ("m1", "8K"), ("m2", "4K")}

    def test_mean_matches_manual(self):
        out = stratum_summary(self._sample(), "scispacy-clinical/f1", seed=1)
        indexed = out.set_index(["model", "token_stratum"])
        assert indexed.loc[("m1", "4K"), "mean"] == pytest.approx(0.5)

    def test_n_excludes_null_metric_rows(self):
        out = stratum_summary(self._sample(), "scispacy-clinical/f1", seed=1)
        # m2/4K has one scored and one null row, so n is 1.
        indexed = out.set_index(["model", "token_stratum"])
        assert indexed.loc[("m2", "4K"), "n"] == 1
