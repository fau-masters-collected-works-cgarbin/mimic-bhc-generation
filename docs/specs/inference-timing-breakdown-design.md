# Inference Timing Breakdown Design

## Questions this answers

- How long did prefill (prompt processing) take versus decode (token generation)?
- What were the prefill and decode throughputs (tokens/second) for each admission?

`latency_ms` alone is an end-to-end wall clock measured around the SDK call. It conflates prompt length with generation length and includes HTTP and queue overhead, so it cannot separate prefill from decode. llama-server reports the split in its `timings` object.

## Decision

Capture llama-server's `timings` object verbatim, store it on disk, and derive the breakdown in the analysis layer. Storing the raw object lets the derivation change later without re-running anything.

**The summary throughput aggregate is a plain mean of per-admission rates.** Long admissions therefore weigh the same as short ones. A table that compares throughput across admission sizes should use total tokens over total time, from `prompt_n` and `predicted_n` in the stored object.

## References

- llama-server response format, `timings` object: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- OpenAI Python SDK response models use Pydantic `extra='allow'`, preserving non-standard fields (verified against openai 2.41.0).
- `results-storage-design.md`, the JSONL record that stores the `timings` object.
