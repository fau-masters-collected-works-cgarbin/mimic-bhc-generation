"""Cost and latency report.

Questions answered:

- What does a Brief Hospital Course cost in wall-clock time at each admission
  size, and how is that time split between prefill and decode?
- What is the cost-quality tradeoff: how much latency buys how much quality?

Cost here is time and tokens. Latency uses completed generations only, since a
truncated run hits the cap and inflates its time.

Usage:

    uv run python scripts/analysis/cost.py <run-folder>
"""

import argparse
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless backend: save figures without a display

import matplotlib.pyplot as plt
import pandas as pd

from bhc_experiment import reporting
from bhc_experiment.metrics import PRIMARY_METRIC

logger = logging.getLogger(__name__)

REPORT_NAME = "cost"
LATENCY_FIGURE = "latency_vs_size"
FRONTIER_FIGURE = "cost_quality_frontier"

MS_PER_SECOND = 1000.0


def _cost_summary(completed: pd.DataFrame) -> pd.DataFrame:
    """Summarize latency, tokens, and quality for each model and stratum."""
    grouped = completed.groupby(["model", "token_stratum"], dropna=False)
    # Median for skewed timing and token counts, robust to the occasional slow
    # generation. Mean for quality.
    summary = grouped.agg(
        latency_ms=("latency_ms", "median"),
        prefill_ms=("prefill_ms", "median"),
        decode_ms=("decode_ms", "median"),
        input_tokens=("input_tokens", "median"),
        output_tokens=("output_tokens", "median"),
        quality=(PRIMARY_METRIC, "mean"),
        n=("latency_ms", "size"),
    ).reset_index()
    summary["latency_s"] = summary["latency_ms"] / MS_PER_SECOND
    summary["prefill_s"] = summary["prefill_ms"] / MS_PER_SECOND
    summary["decode_s"] = summary["decode_ms"] / MS_PER_SECOND
    summary = summary[
        [
            "model",
            "token_stratum",
            "latency_s",
            "prefill_s",
            "decode_s",
            "input_tokens",
            "output_tokens",
            "quality",
            "n",
        ]
    ]
    # Strata labels like "500-1000" sort lexically, so order on the numeric
    # sort key instead.
    stratum_rank = summary["token_stratum"].map(reporting.stratum_sort_key)
    ordered = summary.assign(stratum_rank=stratum_rank).sort_values(["model", "stratum_rank"])
    return ordered.drop(columns="stratum_rank").reset_index(drop=True)


def _cost_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Long table with one row per model and stratum."""
    table = summary[
        [
            "model",
            "token_stratum",
            "latency_s",
            "prefill_s",
            "decode_s",
            "input_tokens",
            "output_tokens",
            "n",
        ]
    ].copy()
    table["latency_s"] = table["latency_s"].map(lambda v: f"{v:.1f}")
    table["prefill_s"] = table["prefill_s"].map(lambda v: f"{v:.1f}")
    table["decode_s"] = table["decode_s"].map(lambda v: f"{v:.1f}")
    table["input_tokens"] = table["input_tokens"].map(lambda v: f"{v:.0f}")
    table["output_tokens"] = table["output_tokens"].map(lambda v: f"{v:.0f}")
    return table.rename(
        columns={
            "model": "Model",
            "token_stratum": "Admission size",
            "latency_s": "Median latency (s)",
            "prefill_s": "Prefill (s)",
            "decode_s": "Decode (s)",
            "input_tokens": "Input tokens",
            "output_tokens": "Output tokens",
        }
    )


def _latency_figure(completed: pd.DataFrame, run_dir: Path) -> str:
    """Per-admission latency against input size, one series per model.

    Plotted against input tokens so the prefill-dominated growth in time is
    visible directly.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in sorted(completed["model"].unique()):
        sub = completed[completed["model"] == model]
        ax.scatter(sub["input_tokens"], sub["latency_ms"] / 1000.0, s=16, alpha=0.5, label=model)
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency vs admission size")
    ax.legend(title="model")
    return reporting.save_figure(fig, run_dir, LATENCY_FIGURE)


def _frontier_figure(summary: pd.DataFrame, run_dir: Path) -> str:
    """Cost-quality frontier: median latency against mean concept F1 per cell."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in sorted(summary["model"].unique()):
        model_rows = summary[summary["model"] == model]
        ax.plot(
            model_rows["latency_s"], model_rows["quality"], marker="o", linestyle="-", label=model
        )
        for _, row in model_rows.iterrows():
            ax.annotate(
                row["token_stratum"],
                (row["latency_s"], row["quality"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlabel("Median latency (s)")
    ax.set_ylabel("Mean scispaCy concept F1")
    ax.set_title("Cost-quality frontier")
    ax.legend(title="model")
    return reporting.save_figure(fig, run_dir, FRONTIER_FIGURE)


def build_cost_report(df: pd.DataFrame, run_dir: Path) -> str:
    """Build the markdown for the cost report and save its figures."""
    completed = df[~df["truncated"]]
    summary = _cost_summary(completed)
    table = reporting.markdown_table(_cost_table(summary))
    latency_path = _latency_figure(completed, run_dir)
    frontier_path = _frontier_figure(summary, run_dir)
    return (
        "# Cost and latency\n\n"
        f"_Run: `{run_dir.name}`_\n\n"
        "Cost is wall-clock time and tokens. Latency is the median over completed "
        "generations, split into prefill and decode time so the prefill-dominated cost at long "
        "context is visible. The frontier shows how much latency buys how much "
        "quality.\n\n"
        f"{table}\n\n"
        f"![Latency vs admission size]({latency_path})\n\n"
        f"![Cost-quality frontier]({frontier_path})\n"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="path to the experiment run folder")
    args = parser.parse_args()

    df = reporting.resolve_run_df(args.run_dir)
    body = build_cost_report(df, args.run_dir)
    report_path = reporting.write_report(args.run_dir, REPORT_NAME, body)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
