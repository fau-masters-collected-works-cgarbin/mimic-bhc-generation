"""Summarize experiment results across the models of one run.

Why: the results viewer shows one model at a time, but comparing the models of
a run needs them side by side in a single table.

Questions answered:
- How do the run's models compare on each metric?
- What are each model's token counts and latency?

Usage:
    RESULTS=../phd-dissertation-experiment-results/results
    # Mean scores per model for the brief prompt over all records
    uv run python scripts/pipeline/summarize_runs.py $RESULTS/all-records/brief-prompt
    # Per-admission breakdown instead of per-model means
    uv run python scripts/pipeline/summarize_runs.py $RESULTS/all-records/brief-prompt \
        --per-admission
    # Output as markdown table (default is ascii)
    uv run python scripts/pipeline/summarize_runs.py $RESULTS/all-records/brief-prompt \
        --format markdown
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from bhc_experiment.results_reader import (
    build_per_admission_df,
    build_summary_df,
    load_result_records,
    load_run_record,
    model_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def print_table(df: pd.DataFrame, *, fmt: str = "ascii", transpose: bool = False) -> None:
    """Print a DataFrame as an ascii or markdown table."""
    df = df.copy()  # avoid mutating the caller's DataFrame
    float_cols = df.select_dtypes(include="float").columns
    for col in float_cols:
        # Token-count columns print as whole numbers. A missing count prints blank.
        if "token" in col:
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{round(x):d}")
        else:
            df[col] = df[col].map(lambda x: f"{x:.3f}")

    if transpose and not df.empty:
        # Model per column, metric per row
        df = df.set_index("model").T
        df.index.name = None
        df.columns.name = None
        df = df.reset_index().rename(columns={"index": ""})

    if fmt == "markdown":
        cols = [str(c) for c in df.columns]
        header = "| " + " | ".join(cols) + " |"
        separator = "| " + " | ".join("---" for _ in cols) + " |"
        print(header)
        print(separator)
        for row in df.itertuples(index=False):
            print("| " + " | ".join(str(v) for v in row) + " |")
    else:
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.add_argument("run_dir", type=Path, help="The run: results/<selection>/<prompt>-prompt")
    parser.add_argument(
        "--per-admission", action="store_true", help="Show per-admission breakdown instead of means"
    )
    parser.add_argument(
        "--format",
        choices=["ascii", "markdown"],
        default="ascii",
        help="Output format (default: ascii)",
    )
    args = parser.parse_args()

    dirs = model_dirs(args.run_dir) if args.run_dir.is_dir() else []
    if not dirs:
        logger.error("No models found in %s", args.run_dir)
        raise SystemExit(1)
    logger.info("Found %d models in %s", len(dirs), args.run_dir)

    models = [(load_run_record(d), load_result_records(d)) for d in dirs]

    df = build_per_admission_df(models) if args.per_admission else build_summary_df(models)

    # Transpose the summary (few models, many metrics) so models are columns. The
    # per-admission view already has many rows, so a transpose would not help it.
    print_table(df, fmt=args.format, transpose=not args.per_admission)


if __name__ == "__main__":
    main()
