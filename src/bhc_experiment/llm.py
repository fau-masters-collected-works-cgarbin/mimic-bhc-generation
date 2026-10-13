"""LLM calling layer for the experiment pipeline.

Thin wrapper around the OpenAI SDK, talking to a local llama-server through its
OpenAI-compatible API.
"""

import logging
import time
from dataclasses import dataclass

import httpx
import openai
from openai.types.chat import ChatCompletion

from bhc_experiment.prompt_builder import PromptResult

log = logging.getLogger(__name__)

_RETRY_SLEEP_SECONDS = 5
_HTTP_OK = 200

# On-device timeout. The OpenAI SDK default is 600s, which is too tight for
# long admissions on a local llama-server: prompt processing plus decode at
# Metal speeds can exceed 10 minutes once the prompt is large enough, and
# the SDK then retries the same hung request twice before raising the
# error. max_retries is set to 0 because retrying a hung local server just
# multiplies the wait.
_ON_DEVICE_TIMEOUT_SECONDS = 1800.0
_ON_DEVICE_MAX_RETRIES = 0

# The OpenAI SDK requires a non-empty API key. The local server ignores it.
_ON_DEVICE_API_KEY = "on-device"

# Must match serve_local_llm.sh's --host and --port.
LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"

# Reasoning is turned off through enable_thinking in the Jinja chat template, which
# takes effect only when the server runs with --jinja. A template without a
# reasoning mode never reads it.
_REASONING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class InferenceConfig:
    """Inference parameters for LLM calls.

    frequency_penalty breaks degenerate repetition loops while staying
    deterministic at temperature 0.
    """

    temperature: float
    max_completion_tokens: int
    frequency_penalty: float


@dataclass(frozen=True)
class LlmResult:
    """Result from a single LLM call.

    Reasoning is always turned off. ``reasoning_content`` is the reasoning text
    the server returned, kept so the results show it was empty.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int | None
    model: str
    latency_ms: float
    # Raw llama-server ``timings`` object, stored verbatim.
    timings: dict | None = None
    reasoning_content: str | None = None
    # True when generation stopped because it hit max_completion_tokens
    # (finish_reason "length"), so the text is kept and incomplete.
    truncated: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "LlmResult":
        """Construct an LlmResult from a JSON-deserialized dict."""
        return cls(**data)


def timing_breakdown(timings: dict | None) -> dict[str, float | None]:
    """Map llama-server's raw ``timings`` object to neutral timing fields."""
    t = timings or {}
    return {
        "prefill_ms": t.get("prompt_ms"),
        "decode_ms": t.get("predicted_ms"),
        "prefill_tok_s": t.get("prompt_per_second"),
        "decode_tok_s": t.get("predicted_per_second"),
    }


class ReasoningNotDisabledError(RuntimeError):
    """The model returned reasoning text even though reasoning was turned off."""


def _verify_reasoning_disabled(result: LlmResult) -> None:
    """Raise when the model returned any reasoning text.

    llama-server never reports a reasoning token count, so the reasoning text it
    returns is the only evidence of whether the model reasoned.
    """
    if result.reasoning_content:
        msg = (
            f"Reasoning is not disabled: the model returned "
            f"{len(result.reasoning_content)} characters of reasoning text. Verify the "
            "server runs with --jinja and that enable_thinking=false reaches the chat "
            "template."
        )
        raise ReasoningNotDisabledError(msg)


def _parse_response(response: ChatCompletion, latency_ms: float) -> LlmResult:
    """Extract an LlmResult from an OpenAI SDK ChatCompletion."""
    if response.usage is None:
        msg = "API response missing usage data"
        raise ValueError(msg)
    usage = response.usage
    cached_tokens = None
    if usage.prompt_tokens_details is not None:
        cached_tokens = usage.prompt_tokens_details.cached_tokens

    message = response.choices[0].message
    return LlmResult(
        text=message.content or "",
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cached_tokens=cached_tokens,
        model=response.model,
        latency_ms=latency_ms,
        timings=response.model_extra.get("timings"),
        # llama-server returns reasoning in this extra message field, which the
        # SDK keeps.
        reasoning_content=getattr(message, "reasoning_content", None),
        truncated=response.choices[0].finish_reason == "length",
    )


def create_llm_client() -> openai.OpenAI:
    """Create an OpenAI SDK client for the local server.

    Call once at startup and reuse across all LLM calls in a run
    to benefit from the SDK's connection pooling.
    """
    return openai.OpenAI(
        base_url=LOCAL_BASE_URL,
        api_key=_ON_DEVICE_API_KEY,
        timeout=_ON_DEVICE_TIMEOUT_SECONDS,
        max_retries=_ON_DEVICE_MAX_RETRIES,
    )


def probe_on_device_server(base_url: str, *, timeout_s: float = 5.0) -> dict:
    """Probe an on-device OpenAI-compatible server for runtime metadata.

    ``model_alias`` is the id from ``/v1/models``. ``model_path``,
    ``n_ctx``, and ``n_gpu_layers`` come from llama-server's ``/props``
    endpoint when available.

    Raises ``RuntimeError`` if the server is not reachable.
    """
    info: dict = {
        "base_url": base_url,
        "model_alias": None,
        "model_path": None,
        "n_ctx": None,
        "n_gpu_layers": None,
    }
    models_url = base_url.rstrip("/") + "/models"
    try:
        models_resp = httpx.get(models_url, timeout=timeout_s)
        models_resp.raise_for_status()
        models_data = models_resp.json()
    except (httpx.HTTPError, ValueError) as e:
        msg = f"On-device server not reachable at {base_url}: {e}"
        raise RuntimeError(msg) from e
    data = models_data.get("data") or []
    if data:
        info["model_alias"] = data[0].get("id")

    # /props is llama-server-specific. If absent, those fields stay None.
    root = base_url.rstrip("/").removesuffix("/v1")
    props_url = root + "/props"
    try:
        props_resp = httpx.get(props_url, timeout=timeout_s)
        if props_resp.status_code == _HTTP_OK:
            props = props_resp.json()
            settings = props.get("default_generation_settings") or {}
            info["n_ctx"] = settings.get("n_ctx") or props.get("n_ctx")
            info["n_gpu_layers"] = props.get("n_gpu_layers")
            info["model_path"] = props.get("model_path") or props.get("model")
    except (httpx.HTTPError, ValueError):
        pass
    return info


def call_llm(
    prompt: PromptResult,
    model_alias: str,
    inference: InferenceConfig,
    llm_client: openai.OpenAI,
) -> LlmResult:
    """Call an OpenAI-compatible API and return the result.

    On any SDK error, retries once before propagating the exception.

    A truncated response is not an error and is not retried, since with
    greedy decoding the retry would reproduce the same truncated output.
    """

    def _attempt() -> LlmResult:
        start = time.perf_counter()
        response = llm_client.chat.completions.create(
            model=model_alias,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            temperature=inference.temperature,
            max_completion_tokens=inference.max_completion_tokens,
            frequency_penalty=inference.frequency_penalty,
            extra_body=_REASONING_OFF,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return _parse_response(response, latency_ms=elapsed_ms)

    try:
        result = _attempt()
    except Exception as e:  # noqa: BLE001
        log.warning("LLM call failed: %s. Retrying once in %ds.", e, _RETRY_SLEEP_SECONDS)
        time.sleep(_RETRY_SLEEP_SECONDS)
        result = _attempt()
    _verify_reasoning_disabled(result)
    return result
