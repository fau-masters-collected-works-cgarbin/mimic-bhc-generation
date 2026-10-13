"""Verify the server turns reasoning off.

Reasoning is turned off through chat_template_kwargs (enable_thinking), which
llama-server honors only when started with --jinja. A server started without it
reasons anyway. ``call_llm`` raises when the model returns reasoning text, and
this probe also catches reasoning that inflates the output token count without
coming back as reasoning text.

Run it after (re)starting serve_local_llm.sh.

Usage:
    uv run python scripts/setup/reasoning_control_smoke.py [model key, default gemma_4_e4b]
"""

import argparse
import sys

from bhc_experiment.config import INFERENCE
from bhc_experiment.llm import (
    LOCAL_BASE_URL,
    ReasoningNotDisabledError,
    call_llm,
    create_llm_client,
)
from bhc_experiment.prompt_builder import PromptResult
from mimic_data_prep.models import MODEL_KEYS, model_by_key

DEFAULT_MODEL = "gemma_4_e4b"

# A substantive clinical prompt that reliably elicits a chain-of-thought when
# reasoning is on.
_PROMPT = PromptResult(
    system="You are a clinical documentation specialist. Write the Brief Hospital Course.",
    user=(
        "65F admitted with community-acquired pneumonia, treated with IV ceftriaxone "
        "and azithromycin, course complicated by mild AKI that resolved with fluids, "
        "improved, discharged home on day 4. Write the brief hospital course."
    ),
    template_name="smoke",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, choices=MODEL_KEYS)
    alias = model_by_key(parser.parse_args().model).alias
    print(f"Probing {alias} at {LOCAL_BASE_URL}")

    try:
        result = call_llm(_PROMPT, alias, INFERENCE, create_llm_client())
    except ReasoningNotDisabledError as e:
        print(f"FAIL: reasoning not disabled.\n  {e}")
        return 1
    print(f"  output_tokens={result.output_tokens} text_chars={len(result.text)}")

    # Catches silently-dropped reasoning: tokens counted but never returned in
    # reasoning_content, because Gemma's channel tokens can be mishandled. A
    # clean answer runs roughly len(text) / 4 tokens, so several times that is the
    # signal.
    est_text_tokens = max(1, len(result.text) // 4)
    if result.output_tokens > 3 * est_text_tokens:
        print(
            f"FAIL: {result.output_tokens} output tokens for ~{est_text_tokens} tokens "
            "of text with no surfaced reasoning. The model is likely still reasoning "
            "but not reporting it. Check --jinja."
        )
        return 1

    print("PASS: reasoning is disabled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
