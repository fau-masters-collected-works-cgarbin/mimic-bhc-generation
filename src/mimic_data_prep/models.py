"""The three local models, defined once.

Every place that names a model uses ``key``: model directories, results tables,
the cohort's per-model token counts, and the tokenizers. ``alias`` is the name
llama-server serves the model under, used only to talk to the server.

The server scripts read fields through ``python -m mimic_data_prep.models <key> <field>``.
"""

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    key: str
    alias: str
    display_name: str
    hf_repo: str  # tokenizer source
    gguf_file: str  # weights under local-llm/models/
    flash_attn: bool  # llama-server --flash-attn default for this model
    dry_sampler: bool  # generate with config.DRY_SAMPLER


MODELS: tuple[Model, ...] = (
    Model(
        key="llama_3_2_3b",
        alias="llama-3.2-3b",
        display_name="Llama 3.2 3B",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        gguf_file="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        flash_attn=True,
        # The only model whose repetition loops reach the output cap often enough
        # to matter. See "DRY sampler" in docs/specs/prompt-builder-design.md.
        dry_sampler=True,
    ),
    Model(
        key="qwen_3_5_4b",
        alias="qwen3.5-4b",
        display_name="Qwen3.5-4B",
        hf_repo="Qwen/Qwen3.5-4B",
        gguf_file="Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        flash_attn=True,
        dry_sampler=False,
    ),
    Model(
        key="gemma_4_e4b",
        alias="gemma-4-e4b",
        display_name="Gemma 4 E4B",
        hf_repo="google/gemma-4-E4B-it",
        gguf_file="google_gemma-4-E4B-it-Q4_K_M.gguf",
        # Off on the Apple GPU: faster prefill, the cohort's dominant cost, wins
        # total wall time at 16K+ even though on is faster for Gemma decode. See
        # "Local llama.cpp inference setup.md" in the writing vault.
        flash_attn=False,
        dry_sampler=False,
    ),
)
MODEL_KEYS: tuple[str, ...] = tuple(m.key for m in MODELS)


def model_by_key(key: str) -> Model:
    """The registered model with this key. Raises ValueError naming the valid keys."""
    for model in MODELS:
        if model.key == key:
            return model
    msg = f"Unknown model {key!r}. Expected one of {list(MODEL_KEYS)}."
    raise ValueError(msg)


def model_by_alias(alias: str) -> Model:
    """The registered model llama-server serves under this alias."""
    for model in MODELS:
        if model.alias == alias:
            return model
    msg = f"Unknown model alias {alias!r}. Expected one of {[m.alias for m in MODELS]}."
    raise ValueError(msg)


def main(argv: list[str]) -> int:
    if len(argv) != 2:  # noqa: PLR2004
        sys.stderr.write("usage: python -m mimic_data_prep.models <key> <field>\n")
        return 2
    key, field = argv
    try:
        value = getattr(model_by_key(key), field)
    except (ValueError, AttributeError) as e:
        sys.stderr.write(f"{e}\n")
        return 1
    # Shell-friendly booleans for flash_attn.
    sys.stdout.write(f"{str(value).lower() if isinstance(value, bool) else value}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
