# Analysis reporting scripts

Each script reads one experiment run folder, computes one aspect of it, and writes its report and figures back into that folder.

A run folder is a directory like `.../results/all-records/format-matched-prompt`. It has one subdirectory per model, each with that model's run record and per-admission results.

## How to run

The run folder is the only argument:

```bash
uv run python scripts/analysis/quality.py <run-folder>
uv run python scripts/analysis/completion.py <run-folder>
uv run python scripts/analysis/cost.py <run-folder>
uv run python scripts/analysis/length.py <run-folder>
uv run python scripts/analysis/output_checks.py <run-folder>
uv run python scripts/analysis/sapbert_window.py <run-folder>
```

| Script | Aspect | Report | Figures |
| --- | --- | --- | --- |
| `quality.py` | Quality vs admission size | `quality.md` | `quality_vs_size.png` |
| `completion.py` | Completion and non-termination | `completion.md` | `completion_rate.png`, `output_tokens.png` |
| `cost.py` | Cost and latency | `cost.md` | `latency_vs_size.png`, `cost_quality_frontier.png` |
| `length.py` | Output length against reference length | `length.md` | none |
| `output_checks.py` | Title echoes, missing-information lines and header use, the prompt test gate counts | `output_checks.md` | none |
| `sapbert_window.py` | Share of references and outputs longer than SapBERT's 510-wordpiece window | `sapbert_window.md` | none |

Reports are stored in the run folder, figures in `<run-folder>/assets/`, referenced from the report by relative path. The scripts are independent. Run them in any order. A re-run rewrites only that script's own outputs.

## What the reports assume

The primary quality metric is the scispaCy UMLS-concept F1, the MEDCON-style metric.

Quality and latency count completed generations only. A truncated generation has no meaningful score. Its time is inflated by hitting the output cap. Completion counts every generation, since truncation is what it measures.

## Conventions for new scripts

The shared logic (the report writer, per-stratum aggregation with bootstrap confidence intervals, the table formatter, and the figure saver) lives in `src/bhc_experiment/reporting.py`. The scripts stay thin so that logic is unit tested on its own.

- **Always show n.** Strata are uneven (4K is the smallest), so a mean is never reported without the count behind it.
- **Fixed order.** Strata render 4K through 112K and models in a fixed order, never DataFrame order.
