"""Export EDA tables as markdown for dissertation write-up.

Usage:

    uv run python scripts/pipeline/export_eda.py summary <run directory> \\
        [--stratify-by <cohort_column>] --output <path>
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from bhc_experiment.analysis import metric_columns, series_metric_means_with_n
from bhc_experiment.results_reader import load_run_df

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_FLOAT_DECIMALS = 3


def format_means_markdown(means: pd.DataFrame, *, title: str | None = None) -> str:
    """Render a series-indexed means DataFrame as a markdown table.

    Floats are rounded to three decimals. When ``title`` is given, an H2
    heading is prepended.
    """
    body = means.round(_FLOAT_DECIMALS).reset_index()
    cols = list(body.columns)
    lines: list[str] = []
    if title:
        lines.append(f"## {title}")
        lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in body.iterrows():
        lines.append("| " + " | ".join(_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _cell(value: object) -> str:
    """Render a cell value, handling NaN and trimming float precision."""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{_FLOAT_DECIMALS}f}"
    return str(value)


def export_summary(args: argparse.Namespace) -> None:
    run_name = f"{args.run_dir.parent.name}/{args.run_dir.name}"
    df = load_run_df(args.run_dir)
    if df.empty:
        log.error("No data under %s", args.run_dir)
        raise SystemExit(1)

    metrics = metric_columns(df)
    if not metrics:
        log.error("No metric columns present. Has the scorer been run?")
        raise SystemExit(1)

    header = _summary_header(run_name, df, metrics)

    if args.stratify_by is None:
        means = series_metric_means_with_n(df, metrics)
        body = format_means_markdown(means)
    else:
        if args.stratify_by not in df.columns:
            log.error("Stratification column '%s' is not in the loaded data.", args.stratify_by)
            raise SystemExit(1)
        means = series_metric_means_with_n(df, metrics, stratify_by=args.stratify_by)
        sections: list[str] = []
        for stratum_value, stratum_means in means.groupby(level=args.stratify_by, dropna=False):
            sections.append(
                format_means_markdown(
                    stratum_means.droplevel(args.stratify_by),
                    title=f"{args.stratify_by} = {stratum_value}",
                )
            )
        body = "\n\n".join(sections)

    output = f"{header}\n{body}\n"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    log.info("Wrote %s (%d bytes)", out_path, len(output))


def _summary_header(run_name: str, df: pd.DataFrame, metrics: list[str]) -> str:
    n_series = df["series_label"].nunique()
    n_admissions = df["hadm_id"].nunique()
    return (
        f"# Summary: {run_name}\n"
        f"\n"
        f"- Series: {n_series}\n"
        f"- Admissions in the run: {n_admissions}. The `n` column counts the scored "
        f"admissions behind each row.\n"
        f"- Metrics: {len(metrics)}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EDA tables as markdown.")
    subs = parser.add_subparsers(dest="subcommand", required=True)

    summary_p = subs.add_parser(
        "summary", help="Per-series per-metric mean table, optionally stratified."
    )
    summary_p.add_argument(
        "run_dir", type=Path, help="The run: results/<selection>/<prompt>-prompt"
    )
    summary_p.add_argument(
        "--stratify-by",
        default=None,
        help="Cohort column to stratify within (one table per stratum value).",
    )
    summary_p.add_argument(
        "--output", required=True, help="Destination markdown file. Parent dirs are created."
    )
    summary_p.set_defaults(func=export_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
