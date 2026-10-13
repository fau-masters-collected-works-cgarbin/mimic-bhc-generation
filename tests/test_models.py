"""Tests for the model registry (mimic_data_prep.models)."""

import subprocess
import sys

import pytest

from mimic_data_prep.models import model_by_alias, model_by_key


class TestLookup:
    def test_by_key(self) -> None:
        assert model_by_key("gemma_4_e4b").alias == "gemma-4-e4b"

    def test_unknown_key_lists_the_valid_ones(self) -> None:
        with pytest.raises(ValueError, match="gemma_4_e4b"):
            model_by_key("gemma4_e4b")

    def test_by_alias(self) -> None:
        assert model_by_alias("qwen3.5-4b").key == "qwen_3_5_4b"

    def test_unknown_alias_raises(self) -> None:
        with pytest.raises(ValueError, match=r"qwen3\.5-4b"):
            model_by_alias("qwen")


class TestShellQuery:
    def test_prints_one_field(self) -> None:
        # The server scripts read the alias and weights file this way.
        result = subprocess.run(
            [sys.executable, "-m", "mimic_data_prep.models", "gemma_4_e4b", "gguf_file"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "google_gemma-4-E4B-it-Q4_K_M.gguf"

    def test_booleans_print_as_shell_words(self) -> None:
        # serve_local_llm.sh compares flash_attn against the string "true".
        result = subprocess.run(
            [sys.executable, "-m", "mimic_data_prep.models", "gemma_4_e4b", "flash_attn"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "false"

    def test_unknown_key_exits_nonzero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mimic_data_prep.models", "nope", "alias"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
