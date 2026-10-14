"""Completion and non-termination report.

Questions answered:

- How often does each model finish a Brief Hospital Course instead of running
  into the output cap, and how does that change with admission size?
- Where does repetition collapse still bite after the frequency-penalty fix?

This report uses every generation, since a truncated run is exactly what we
are counting.

Usage:

    uv run python scripts/analysis/completion.py <run-folder>
"""

import argparse
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless backend: save figures without a display

import matplotlib.pyplot as plt
import pandas as pd

from bhc_experiment import reporting

logger = logging.getLogger(__name__)

REPORT_NAME = "completion"
RATE_FIGURE = "completion_rate"
TOKENS_FIGURE = "output_tokens"

# Fans each model's points sideways so they do not overlap at a shared x position.
JITTER_STEP = 0.12


def _completion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count completed and truncated generations for each model and stratum."""
    grouped = df.groupby(["model", "token_stratum"], dropna=False)
    summary = grouped.agg(
        n=("truncated", "size"),
        truncated=("truncated", "sum"),
        output_median=("output_tokens", "median"),
        output_max=("output_tokens", "max"),
    ).reset_index()
    summary["truncated"] = summary["truncated"].astype(int)
    truncation_fraction = summary["truncated"] / summary["n"]
    summary["completion_rate"] = 1.0 - truncation_fraction
    summary["_rank"] = summary["token_stratum"].map(reporting.stratum_sort_key)
    return summary.sort_values(["model", "_rank"]).drop(columns="_rank").reset_index(drop=True)


def _completion_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Long table with one row per model and stratum, rate shown as a percent."""
    table = summary.copy()
    table["Completion rate"] = (100.0 * table["completion_rate"]).map(lambda v: f"{v:.1f}%")
    # Token counts read as whole numbers. An even-n median can be a half value.
    table["output_median"] = table["output_median"].map(lambda v: f"{v:g}")
    table["output_max"] = table["output_max"].map(lambda v: f"{v:.0f}")
    table = table[
        [
            "model",
            "token_stratum",
            "Completion rate",
            "truncated",
            "n",
            "output_median",
            "output_max",
        ]
    ]
    return table.rename(
        columns={
            "model": "Model",
            "token_stratum": "Admission size",
            "truncated": "Truncated",
            "n": "n",
            "output_median": "Output tokens (median)",
            "output_max": "Output tokens (max)",
        }
    )


def _rate_figure(summary: pd.DataFrame, run_dir: Path) -> str:
    """Completion rate against admission size, one line per model."""
    strata = reporting.sort_strata(summary["token_stratum"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in sorted(summary["model"].unique()):
        model_rows = summary[summary["model"] == model].set_index("token_stratum")
        rate_by_stratum = model_rows.reindex(strata)
        rate_along_axis = 100.0 * rate_by_stratum["completion_rate"]
        ax.plot(range(len(strata)), rate_along_axis, marker="o", label=model)
    ax.set_xticks(list(range(len(strata))))
    ax.set_xticklabels(strata)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Admission size (token stratum)")
    ax.set_ylabel("Completion rate (%)")
    ax.set_title("Completion rate vs admission size")
    ax.legend(title="model")
    return reporting.save_figure(fig, run_dir, RATE_FIGURE)


def _tokens_figure(df: pd.DataFrame, run_dir: Path) -> str:
    """Output-token counts per stratum, with the output cap drawn in.

    The cap is inferred from the truncated rows, since a truncated generation
    terminates exactly at it.
    """
    strata = reporting.sort_strata(df["token_stratum"].unique())
    positions = {s: i for i, s in enumerate(strata)}
    models = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for offset, model in enumerate(models):
        sub = df[df["model"] == model]
        jitter = (offset - (len(models) - 1) / 2) * JITTER_STEP
        ax.scatter(
            [positions[s] + jitter for s in sub["token_stratum"]],
            sub["output_tokens"],
            s=14,
            alpha=0.4,
            label=model,
        )
    truncated = df[df["truncated"]]
    if not truncated.empty:
        cap = int(truncated["output_tokens"].max())
        ax.axhline(cap, linestyle="--", color="grey", linewidth=1)
        ax.text(0, cap, f" output cap ({cap})", va="bottom", ha="left", color="grey", fontsize=8)
    ax.set_xticks(list(range(len(strata))))
    ax.set_xticklabels(strata)
    ax.set_xlabel("Admission size (token stratum)")
    ax.set_ylabel("Output tokens")
    ax.set_title("Output-token distribution vs cap")
    ax.legend(title="model")
    return reporting.save_figure(fig, run_dir, TOKENS_FIGURE)


def build_completion_report(df: pd.DataFrame, run_dir: Path) -> str:
    """Build the markdown for the completion report and save its figures."""
    summary = _completion_summary(df)
    table = reporting.markdown_table(_completion_table(summary))
    rate_path = _rate_figure(summary, run_dir)
    tokens_path = _tokens_figure(df, run_dir)
    truncated_total = int(df["truncated"].sum())
    return (
        "# Completion and non-termination\n\n"
        f"_Run: `{run_dir.name}`_\n\n"
        "Completion rate is the share of generations that terminated normally "
        "rather than hitting the output cap. A capped generation is a "
        "repetition collapse and is excluded from quality scoring. This "
        f"report counts every generation. {truncated_total} of {len(df)} were "
        "truncated.\n\n"
        f"{table}\n\n"
        f"![Completion rate vs admission size]({rate_path})\n\n"
        f"![Output-token distribution vs cap]({tokens_path})\n"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="path to the experiment run folder")
    args = parser.parse_args()

    df = reporting.resolve_run_df(args.run_dir)
    body = build_completion_report(df, args.run_dir)
    report_path = reporting.write_report(args.run_dir, REPORT_NAME, body)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
