"""Tests for mimic_data_prep.tokenizers.

Real tokenizer downloads need network access and HF_TOKEN for the gated
Llama repo. These tests mock AutoTokenizer.from_pretrained so they run
offline and stay fast.
"""

from unittest.mock import MagicMock, patch

from mimic_data_prep import tokenizers as tk
from mimic_data_prep.models import MODEL_KEYS


class _FakeTokenizer:
    """Returns a token count equal to the word count of the input.

    Deterministic and lets fixtures pick any number by choosing the word
    count of the test string. ``__len__`` mirrors HF tokenizers, which
    return the full vocab size (base plus added tokens) for ``len(tok)``.
    """

    def __init__(self, vocab_size: int = 100) -> None:
        self._vocab_size = vocab_size

    def __len__(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG002, FBT001, FBT002
        return [0] * len(text.split())


def test_load_all_tokenizers_returns_one_per_model() -> None:
    with patch.object(tk, "AutoTokenizer") as mock_auto:
        mock_auto.from_pretrained.side_effect = lambda _repo: _FakeTokenizer()
        toks = tk.load_all_tokenizers()
    assert set(toks.keys()) == set(MODEL_KEYS)
    assert mock_auto.from_pretrained.call_count == 3


def test_count_tokens_excludes_special_tokens() -> None:
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3, 4]
    n = tk.count_tokens(tok, "hello world")
    assert n == 4
    tok.encode.assert_called_once_with("hello world", add_special_tokens=False)


def test_count_per_model_returns_dict_keyed_by_model() -> None:
    toks = {key: _FakeTokenizer() for key in MODEL_KEYS}
    counts = tk.count_per_model(toks, "one two three four five")
    assert counts == {"llama_3_2_3b": 5, "qwen_3_5_4b": 5, "gemma_4_e4b": 5}


def test_tokenizer_metadata_records_repo_and_vocab_per_model() -> None:
    toks = {
        "llama_3_2_3b": _FakeTokenizer(vocab_size=128_256),
        "qwen_3_5_4b": _FakeTokenizer(vocab_size=248_044),
        "gemma_4_e4b": _FakeTokenizer(vocab_size=262_144),
    }
    meta = tk.tokenizer_metadata(toks)
    assert meta["add_special_tokens"] is False
    assert meta["models"]["llama_3_2_3b"]["hf_repo"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert meta["models"]["llama_3_2_3b"]["vocab_size"] == 128_256
    assert meta["models"]["qwen_3_5_4b"]["hf_repo"] == "Qwen/Qwen3.5-4B"
    assert meta["models"]["gemma_4_e4b"]["hf_repo"] == "google/gemma-4-E4B-it"
    assert "transformers_version" in meta
