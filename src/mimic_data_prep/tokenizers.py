"""Hugging Face tokenizers for the target models.

The pool stores per-model counts so the cost-efficiency analysis does not
depend on a single proxy tokenizer.

Tokenizer downloads need the setup in docs/setup.md.
"""

import logging
from typing import TYPE_CHECKING

import transformers
from transformers import AutoTokenizer

from mimic_data_prep.models import MODELS, model_by_key

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_all_tokenizers() -> dict[str, "PreTrainedTokenizerBase"]:
    """Load a tokenizer per model, keyed by model_key."""
    out: dict[str, PreTrainedTokenizerBase] = {}
    for model in MODELS:
        logger.info("Loading tokenizer %s from %s", model.key, model.hf_repo)
        tok = AutoTokenizer.from_pretrained(model.hf_repo)
        # These tokenizers only count tokens, never feed a model, so the context
        # cap doesn't apply. Lifting model_max_length silences the "sequence
        # length > model_max_length" warning for admissions whose text exceeds
        # 128K or 262K tokens.
        tok.model_max_length = int(1e30)
        out[model.key] = tok
    return out


def count_tokens(tokenizer: "PreTrainedTokenizerBase", text: str) -> int:
    """Token count for one tokenizer. Excludes special tokens, so the count is
    the text alone."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_per_model(tokenizers: dict[str, "PreTrainedTokenizerBase"], text: str) -> dict[str, int]:
    """Token counts for all loaded tokenizers on one text."""
    return {key: count_tokens(tok, text) for key, tok in tokenizers.items()}


def tokenizer_metadata(tokenizers: dict[str, "PreTrainedTokenizerBase"]) -> dict:
    """Self-describing metadata block for the pool file."""
    return {
        "models": {
            key: {
                "hf_repo": model_by_key(key).hf_repo,
                "vocab_size": len(tok),
            }
            for key, tok in tokenizers.items()
        },
        "transformers_version": transformers.__version__,
        "add_special_tokens": False,
    }
