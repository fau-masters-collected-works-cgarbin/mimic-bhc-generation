"""Quality vs admission size report.

Questions answered:

- How well does each model summarize at each admission-size stratum, on the
  primary clinical-concept metric?
- Does quality hold, rise, or fall as admissions grow? The shape of that
  curve is the all-records result that the working-budget framing rests on.

The primary metric is the scispaCy UMLS-concept F1 (the MEDCON-style metric).
Truncated generations have no score, so they drop out of every mean here.

Usage:

    uv run python scripts/analysis/quality.py <run-folder>
"""

import argparse
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless backend: save figures without a display

import matplotlib.pyplot as plt
import pandas as pd

from bhc_experiment import reporting
from bhc_experiment.metrics import (
    PRIMARY_METRIC,
    SAPBERT_F1,
    SCISPACY_CLINICAL_PRECISION,
    SCISPACY_CLINICAL_RECALL,
)

logger = logging.getLogger(__name__)

REPORT_NAME = "quality"
FIGURE_NAME = "quality_vs_size"

# Fans each model's points sideways so they do not overlap at a shared x position.
JITTER_STEP = 0.12

CONCEPT_PRECISION = SCISPACY_CLINICAL_PRECISION
CONCEPT_RECALL = SCISPACY_CLINICAL_RECALL
# SapBERT is the biomedical BERTScore variant (PubMedBERT tuned on UMLS
# synonyms): it scores clinical-semantic similarity, which two different concepts
# can share.
SEMANTIC_METRIC = SAPBERT_F1


def _pivot_wide(per_cell: pd.DataFrame) -> pd.DataFrame:
    """Pivot a per-cell frame with a ``cell`` string into stratum-by-model form."""
    # Cells are preformatted strings with unique (stratum, model) pairs, so
    # there is nothing to aggregate and pivot_table does not apply.
    wide = per_cell.pivot(index="token_stratum", columns="model", values="cell")  # noqa: PD010
    wide = wide.reindex(reporting.sort_strata(wide.index))
    wide = wide[sorted(wide.columns)]
    return wide.reset_index().rename(columns={"token_stratum": "Admission size"})


def _ci_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Stratum-by-model table whose cells read ``mean [ci_lower, ci_upper] n``."""
    summary = summary.copy()
    summary["cell"] = summary.apply(
        lambda r: f"{r['mean']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] n={r['n']}",
        axis=1,
    )
    return _pivot_wide(summary)


def _precision_recall_table(df: pd.DataFrame) -> pd.DataFrame:
    """Stratum-by-model table of mean concept precision and recall per cell."""
    # Truncated generations have no precision/recall.
    scored_admissions = df.dropna(subset=[CONCEPT_PRECISION, CONCEPT_RECALL])
    means = (
        scored_admissions.groupby(["model", "token_stratum"], dropna=False)
        .agg(precision=(CONCEPT_PRECISION, "mean"), recall=(CONCEPT_RECALL, "mean"))
        .reset_index()
    )
    means["cell"] = means.apply(lambda r: f"P {r['precision']:.3f} / R {r['recall']:.3f}", axis=1)
    return _pivot_wide(means)


def _plot_model_curve(ax: plt.Axes, model_curve: pd.DataFrame, label: str) -> plt.Line2D:
    """Draw a model's mean concept-F1 line and its 95% CI band, return the line."""
    x = list(range(len(model_curve)))
    line = ax.plot(x, model_curve["mean"], marker="o", label=label)[0]
    ax.fill_between(
        x, model_curve["ci_lower"], model_curve["ci_upper"], alpha=0.15, color=line.get_color()
    )
    return line


def _scatter_raw_scores(
    ax: plt.Axes,
    positions: dict[str, int],
    scored_admissions: pd.DataFrame,
    color: str,
    centered_offset: float,
) -> None:
    """Faint per-admission points for one model, colored to match its curve."""
    jitter = centered_offset * JITTER_STEP
    ax.scatter(
        [positions[s] + jitter for s in scored_admissions["token_stratum"]],
        scored_admissions[PRIMARY_METRIC],
        s=12,
        alpha=0.25,
        color=color,
    )


def _quality_figure(df: pd.DataFrame, summary: pd.DataFrame, run_dir: Path) -> str:
    """Line of mean concept F1 per model across strata, with CI band and points."""
    strata = reporting.sort_strata(df["token_stratum"].unique())
    positions = {s: i for i, s in enumerate(strata)}
    models = sorted(df["model"].unique())
    model_count = len(models)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for offset, model in enumerate(models):
        model_curve = summary[summary["model"] == model].set_index("token_stratum").reindex(strata)
        line = _plot_model_curve(ax, model_curve, model)
        scored_admissions = df[(df["model"] == model) & df[PRIMARY_METRIC].notna()]
        centered_offset = offset - (model_count - 1) / 2
        _scatter_raw_scores(ax, positions, scored_admissions, line.get_color(), centered_offset)
    ax.set_xticks(list(range(len(strata))))
    ax.set_xticklabels(strata)
    ax.set_xlabel("Admission size (token stratum)")
    ax.set_ylabel("scispaCy concept F1 (MEDCON-style)")
    ax.set_title("Quality vs admission size")
    ax.legend(title="model")
    return reporting.save_figure(fig, run_dir, FIGURE_NAME)


def build_quality_report(df: pd.DataFrame, run_dir: Path) -> str:
    """Build the markdown for the quality report and save its figure."""
    concept = reporting.stratum_summary(df, PRIMARY_METRIC)
    semantic = reporting.stratum_summary(df, SEMANTIC_METRIC)
    concept_table = reporting.markdown_table(_ci_table(concept))
    pr_table = reporting.markdown_table(_precision_recall_table(df))
    semantic_table = reporting.markdown_table(_ci_table(semantic))
    figure_path = _quality_figure(df, concept, run_dir)
    scored = int(df[PRIMARY_METRIC].notna().sum())
    total = len(df)
    return (
        "# Quality vs admission size\n\n"
        f"_Run: `{run_dir.name}`_\n\n"
        "Primary metric: scispaCy UMLS-concept F1 (the MEDCON-style metric), "
        "the closest proxy for clinical-concept overlap. Each cell is the mean "
        "with a 95% bootstrap confidence interval and the number of scored "
        "admissions. Truncated generations have no score and are excluded, so "
        f"these means cover {scored} of {total} generations.\n\n"
        f"{concept_table}\n\n"
        f"![Quality vs admission size]({figure_path})\n\n"
        "## Precision and recall (concept metric)\n\n"
        "A low F1 can come from low precision (the model reports concepts the "
        "reference does not) or low recall (it misses reference concepts). This "
        "split separates the two. Values are means over the same scored "
        "admissions.\n\n"
        f"{pr_table}\n\n"
        "## Semantic similarity (SapBERT)\n\n"
        "SapBERT is a BERTScore variant with a biomedical encoder (PubMedBERT "
        "tuned on UMLS synonyms), so it rewards clinically equivalent wording "
        "even when the exact concept does not match. Reading it against the "
        "concept F1 shows how much of the low concept overlap is strict "
        "matching rather than wrong content.\n\n"
        f"{semantic_table}\n"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="path to the experiment run folder")
    args = parser.parse_args()

    df = reporting.resolve_run_df(args.run_dir)
    body = build_quality_report(df, args.run_dir)
    report_path = reporting.write_report(args.run_dir, REPORT_NAME, body)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
