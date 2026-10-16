"""Tests for bhc_experiment.config: run parameters, in code."""

import pytest

from bhc_experiment.config import (
    DRY_SAMPLER,
    INFERENCE,
    PROMPT_TEST_COHORT_FILE,
    SMOKE_COHORT_FILE,
    CliParameters,
    inference_for,
)


class TestCliParameters:
    def test_smoke_runs_on_the_smoke_cohort(self) -> None:
        assert CliParameters(prompt="unformatted", smoke=True).cohort_file == SMOKE_COHORT_FILE

    def test_prompt_test_runs_on_the_prompt_test_cohort(self) -> None:
        assert (
            CliParameters(prompt="format-matched", prompt_test=True).cohort_file
            == PROMPT_TEST_COHORT_FILE
        )

    @pytest.mark.parametrize(
        ("cli_parameters", "name"),
        [
            (CliParameters(prompt="format-matched"), "all-records"),
            (CliParameters(prompt="format-matched", deduplication="none"), "all-records-no-dedup"),
            (CliParameters(prompt="format-matched", oracle_n=0), "oracle-n-0"),
            (CliParameters(prompt="format-matched", oracle_n=2), "oracle-n-2"),
            (
                CliParameters(prompt="format-matched", oracle_n=2, deduplication="none"),
                "oracle-n-2-no-dedup",
            ),
            (CliParameters(prompt="unformatted", smoke=True), "smoke"),
            (CliParameters(prompt="format-matched", prompt_test=True), "prompt-test"),
            (
                CliParameters(prompt="format-matched", prompt_test=True, deduplication="none"),
                "prompt-test-no-dedup",
            ),
        ],
    )
    def test_selection_name(self, cli_parameters: CliParameters, name: str) -> None:
        assert cli_parameters.selection_name == name

    @pytest.mark.parametrize(
        "selections",
        [
            {"oracle_n": 2, "smoke": True},
            {"oracle_n": 2, "prompt_test": True},
            {"smoke": True, "prompt_test": True},
        ],
    )
    def test_more_than_one_selection_raises(self, selections: dict) -> None:
        # The directory names one selection, so a second would be mislabeled.
        with pytest.raises(ValueError, match="at most one"):
            CliParameters(prompt="format-matched", **selections)

    def test_run_name_is_the_selection_and_prompt(self) -> None:
        cli_parameters = CliParameters(prompt="enumerated", oracle_n=2)
        assert cli_parameters.run_name == "oracle-n-2/enumerated-prompt"


class TestInferenceFor:
    """A model given the wrong sampler produces normal-looking output, so pin which gets DRY."""

    def test_llama_gets_the_dry_sampler(self) -> None:
        inference = inference_for("llama_3_2_3b")
        assert inference.dry_sampler == DRY_SAMPLER
        assert inference.frequency_penalty == INFERENCE.frequency_penalty

    @pytest.mark.parametrize("model", ["gemma_4_e4b", "qwen_3_5_4b"])
    def test_other_models_run_without_dry(self, model: str) -> None:
        assert inference_for(model) == INFERENCE
