"""Tests for bhc_experiment.config: run parameters, in code."""

import pytest

from bhc_experiment.config import SMOKE_COHORT_FILE, CliParameters


class TestCliParameters:
    def test_smoke_runs_on_the_smoke_cohort(self) -> None:
        assert CliParameters(prompt="unformatted", smoke=True).cohort_file == SMOKE_COHORT_FILE

    @pytest.mark.parametrize(
        ("cli_parameters", "name"),
        [
            (CliParameters(prompt="brief"), "all-records"),
            (CliParameters(prompt="brief", deduplication="none"), "all-records-no-dedup"),
            (CliParameters(prompt="brief", oracle_n=0), "oracle-n-0"),
            (CliParameters(prompt="brief", oracle_n=2), "oracle-n-2"),
            (
                CliParameters(prompt="brief", oracle_n=2, deduplication="none"),
                "oracle-n-2-no-dedup",
            ),
            (CliParameters(prompt="unformatted", smoke=True), "smoke"),
        ],
    )
    def test_selection_name(self, cli_parameters: CliParameters, name: str) -> None:
        assert cli_parameters.selection_name == name

    def test_oracle_selection_on_the_smoke_cohort_raises(self) -> None:
        # Its directory would be named smoke while the model saw a reduced input.
        with pytest.raises(ValueError, match="smoke"):
            CliParameters(prompt="brief", oracle_n=2, smoke=True)

    def test_run_name_is_the_selection_and_prompt(self) -> None:
        cli_parameters = CliParameters(prompt="thorough", oracle_n=2)
        assert cli_parameters.run_name == "oracle-n-2/thorough-prompt"
