"""Tests for the LLM interface (bhc_experiment.llm)."""

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import PromptTokensDetails

from bhc_experiment.llm import (
    InferenceConfig,
    LlmResult,
    ReasoningNotDisabledError,
    _parse_response,
    call_llm,
    timing_breakdown,
)


def _make_completion(
    content: str = "Generated BHC text.",
    prompt_tokens: int = 500,
    completion_tokens: int = 200,
    cached_tokens: int | None = None,
    model: str = "gemma-4-e4b",
    timings: dict | None = None,
    reasoning_content: str | None = None,
) -> ChatCompletion:
    """Build a ChatCompletion fixture with the given parameters.

    When ``timings`` is provided it is attached as an extra top-level
    field, mirroring how llama-server returns it. The OpenAI SDK keeps
    extra fields because its response models use ``extra='allow'``.
    ``reasoning_content`` is the chain-of-thought llama-server returns in
    an extra message field when a model thinks. It is kept for the same reason.
    """
    prompt_details = None
    if cached_tokens is not None:
        prompt_details = PromptTokensDetails(cached_tokens=cached_tokens)
    extra = {"timings": timings} if timings is not None else {}
    message_kwargs: dict = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message_kwargs["reasoning_content"] = reasoning_content
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(**message_kwargs),
            )
        ],
        created=1700000000,
        model=model,
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=prompt_details,
        ),
        **extra,
    )


def _capturing_client(captured: dict, response: ChatCompletion | None = None):
    """A fake OpenAI client that records the create() kwargs into ``captured``.

    Returns ``response`` (or a default completion) so a test can both assert on
    what call_llm forwarded and drive the parse path.
    """

    def fake_create(**kwargs: object) -> ChatCompletion:
        captured.update(kwargs)
        return response if response is not None else _make_completion()

    class FakeCompletions:
        create = staticmethod(fake_create)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    return FakeClient()


class TestParseResponse:
    """Tests for response-to-LlmResult parsing."""

    def test_normal_response(self) -> None:
        """All token fields populated, including cached tokens."""
        response = _make_completion(cached_tokens=100)
        result = _parse_response(response, latency_ms=1234.5)
        assert result == LlmResult(
            text="Generated BHC text.",
            input_tokens=500,
            output_tokens=200,
            cached_tokens=100,
            model="gemma-4-e4b",
            latency_ms=1234.5,
        )

    def test_cached_tokens_extracted(self) -> None:
        response = _make_completion(cached_tokens=300)
        result = _parse_response(response, latency_ms=100.0)
        assert result.cached_tokens == 300

    @pytest.mark.parametrize("prompt_details", [None, PromptTokensDetails()])
    def test_cached_tokens_none_when_unset(self, prompt_details) -> None:
        """cached_tokens is None whether prompt_tokens_details is absent or present-but-empty."""
        response = _make_completion(cached_tokens=None)
        response.usage.prompt_tokens_details = prompt_details
        result = _parse_response(response, latency_ms=100.0)
        assert result.cached_tokens is None

    def test_zero_completion_tokens(self) -> None:
        response = _make_completion(content="", completion_tokens=0)
        result = _parse_response(response, latency_ms=50.0)
        assert result.output_tokens == 0
        assert result.text == ""

    def test_truncated_response_preserves_text_and_flags(self) -> None:
        """A length-finish response is preserved with truncated=True.

        A partial output is kept with truncated=True so it can be inspected
        and excluded from scoring.
        """
        response = _make_completion(content="partial hospital course")
        response.choices[0].finish_reason = "length"
        result = _parse_response(response, latency_ms=100.0)
        assert result.truncated is True
        assert result.text == "partial hospital course"
        assert result.output_tokens == 200

    def test_captures_timings_when_present(self) -> None:
        """llama-server's timings object is captured verbatim."""
        timings = {
            "prompt_ms": 231.541,
            "predicted_ms": 25.927,
            "prompt_per_second": 99.33,
            "predicted_per_second": 77.14,
            "prompt_n": 23,
            "predicted_n": 2,
        }
        response = _make_completion(timings=timings)
        result = _parse_response(response, latency_ms=100.0)
        assert result.timings == timings


class TestCallLlm:
    """Tests for call_llm parameter forwarding."""

    def test_forwards_parameters_correctly(self) -> None:
        from bhc_experiment.prompt_builder import PromptResult

        captured: dict = {}
        prompt = PromptResult(
            system="You are a clinical assistant.",
            user="Summarize this patient.",
            template_name="unformatted",
        )
        inference = InferenceConfig(
            temperature=0.0, max_completion_tokens=2048, frequency_penalty=0.5
        )

        result = call_llm(prompt, "gemma-4-e4b", inference, _capturing_client(captured))

        assert captured == {
            "model": "gemma-4-e4b",
            "messages": [
                {"role": "system", "content": "You are a clinical assistant."},
                {"role": "user", "content": "Summarize this patient."},
            ],
            "temperature": 0.0,
            "max_completion_tokens": 2048,
            "frequency_penalty": 0.5,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        assert result.text == "Generated BHC text."


class TestReasoningControl:
    """Reasoning is turned off for every model and verified after the call.

    enable_thinking in chat_template_kwargs is honored only when the server runs
    with --jinja. A model that reasons anyway is a hard error, so a run that is
    not what it claims fails fast.
    """

    def _prompt(self):
        from bhc_experiment.prompt_builder import PromptResult

        return PromptResult(system="sys", user="usr", template_name="unformatted")

    def _inference(self) -> InferenceConfig:
        return InferenceConfig(temperature=0.0, max_completion_tokens=2048, frequency_penalty=0.5)

    @pytest.mark.parametrize("reasoning_content", ["I should think...", " "])
    def test_raises_when_the_reasoning_text_is_not_empty(self, reasoning_content) -> None:
        resp = _make_completion(reasoning_content=reasoning_content)
        client = _capturing_client({}, response=resp)
        with pytest.raises(ReasoningNotDisabledError):
            call_llm(self._prompt(), "test-model", self._inference(), client)

    def test_no_raise_when_reasoning_is_off(self) -> None:
        resp = _make_completion(content="clean answer", reasoning_content="")
        client = _capturing_client({}, response=resp)
        result = call_llm(self._prompt(), "test-model", self._inference(), client)
        assert result.text == "clean answer"


class TestTimingBreakdown:
    """timing_breakdown maps llama-server's raw timings to neutral fields."""

    def test_full_object(self) -> None:
        timings = {
            "prompt_ms": 231.541,
            "predicted_ms": 25.927,
            "prompt_per_second": 99.33,
            "predicted_per_second": 77.14,
            "prompt_n": 23,
            "predicted_n": 2,
        }
        assert timing_breakdown(timings) == {
            "prefill_ms": 231.541,
            "decode_ms": 25.927,
            "prefill_tok_s": 99.33,
            "decode_tok_s": 77.14,
        }
