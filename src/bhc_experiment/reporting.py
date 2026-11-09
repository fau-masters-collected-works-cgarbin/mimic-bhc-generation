"""Render an experiment run into per-aspect markdown reports with figures.

Each analysis script under ``scripts/analysis/`` owns one report in the run
folder. This module has what they share: loading a run folder into a
DataFrame, report writing, per-stratum aggregation, table formatting, figure
saving, and the typeface and per-model styling every figure draws with.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bhc_experiment.results_reader import load_run_df, run_cli_parameters
from bhc_experiment.stats import bootstrap_mean_ci
from mimic_data_prep.models import model_by_key
from mimic_data_prep.sampling import TOKEN_STRATA

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from matplotlib.figure import Figure

ASSETS_DIRNAME = "assets"

# The typeface the paper's pics/ diagrams set, so the two kinds of figure print the same.
# Arial comes first because macOS ships Helvetica as a .ttc collection that matplotlib
# resolves to the same face at every weight, which loses bold. The two are metrically
# compatible, so the printed result is the same.
FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans"]

# Applies to every figure the repo draws: each script that draws one imports this module,
# and several save without going through save_figure.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = FONT_FAMILY

# Okabe-Ito, colorblind-safe, ordered alphabetically by display name, the order the result
# tables use. Shared so a model reads the same color in every figure that draws models
# individually.
_MODEL_COLORS = {"gemma_4_e4b": "#0072B2", "llama_3_2_3b": "#D55E00", "qwen_3_5_4b": "#009E73"}
MODEL_STYLE = {key: (model_by_key(key).display_name, color) for key, color in _MODEL_COLORS.items()}


def require_full_input(run_dir: Path) -> None:
    """Raise when the run fed the model an oracle selection.

    A script that measures the ceiling over every prepared record would describe
    records an oracle run's model never saw.
    """
    if run_cli_parameters(run_dir).oracle_n is not None:
        msg = f"{run_dir} is an oracle run. This script measures over every record."
        raise ValueError(msg)


def require_oracle_selection(run_dir: Path) -> None:
    """Raise when the run fed the model every record instead of an oracle selection."""
    if run_cli_parameters(run_dir).oracle_n is None:
        msg = f"{run_dir} is not an oracle run."
        raise ValueError(msg)


def resolve_run_df(run_dir: Path) -> pd.DataFrame:
    """Load a run folder into the per-admission, cohort-joined DataFrame.

    Raises ``FileNotFoundError`` with the path when the run produces no rows,
    so a mistyped folder fails loudly instead of writing an empty report.
    """
    df = load_run_df(Path(run_dir))
    if df.empty:
        msg = f"no results found for run folder {run_dir}"
        raise FileNotFoundError(msg)
    return df


def write_report(run_dir: Path, name: str, body: str) -> Path:
    """Write one analysis report into the run folder and return its path.

    The report is ``<run_dir>/<name>.md``, and ``body`` is the whole file, so
    each script owns its own title and prose. Re-running a script rewrites
    only its own report.
    """
    report_path = Path(run_dir) / f"{name}.md"
    report_path.write_text(body)
    return report_path


def stratum_sort_key(stratum: str) -> int:
    """Rank a stratum by its position in the cohort's stratum order.

    Raises ``ValueError`` on a label outside ``TOKEN_STRATA``.
    """
    return TOKEN_STRATA.index(stratum)


def sort_strata(values: Iterable[str]) -> list[str]:
    """Return the present strata in ``TOKEN_STRATA`` order, smallest first.

    Strata not in ``TOKEN_STRATA`` are dropped.
    """
    present = set(values)
    return [s for s in TOKEN_STRATA if s in present]


def _format_cell(value: object, float_format: str) -> str:
    """Render one table cell: blank for missing, fixed places for floats."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return float_format.format(float(value))
    return str(value)


def _table_row(cells: Iterable[str]) -> str:
    """Join cells into one markdown table row, wrapped and separated by pipes."""
    return "| " + " | ".join(cells) + " |"


def markdown_table(
    df: pd.DataFrame, *, float_format: str = "{:.3f}", right_align: Collection[str] = ()
) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table.

    Columns become the header. Floats use ``float_format``. Everything else
    is stringified. Missing values render blank. The index is not emitted.
    Columns named in ``right_align`` get a right-aligned separator (``---:``)
    so their values line up on the last digit. The rest stay left-aligned.
    """
    right = set(right_align)
    header = _table_row(str(c) for c in df.columns)
    separator = _table_row("---:" if c in right else "---" for c in df.columns)
    rows = [
        _table_row(_format_cell(v, float_format) for v in record)
        for record in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def stratum_summary(
    df: pd.DataFrame,
    metric: str,
    *,
    group_col: str = "model",
    stratum_col: str = "token_stratum",
    seed: int = 0,
) -> pd.DataFrame:
    """Per group and stratum: metric mean, bootstrap 95% CI, and n.

    Rows with a null metric are dropped before aggregating, so n reflects the
    scored count (a truncated generation has no score). Rows are ordered by
    group, then by ``TOKEN_STRATA`` order.
    """
    scored_rows = df[[group_col, stratum_col, metric]].dropna(subset=[metric])
    per_cell_summaries = []
    for (group_value, stratum), cell_scores in scored_rows.groupby(
        [group_col, stratum_col], dropna=False
    ):
        # Reuse the same seed across cells so every cell's CI is reproducible.
        mean_ci = bootstrap_mean_ci(cell_scores[metric].to_numpy(), seed=seed)
        per_cell_summaries.append(
            {
                group_col: group_value,
                stratum_col: stratum,
                "mean": mean_ci["mean"],
                "ci_lower": mean_ci["ci_lower"],
                "ci_upper": mean_ci["ci_upper"],
                "n": mean_ci["n"],
            }
        )
    out = pd.DataFrame.from_records(
        per_cell_summaries,
        columns=[group_col, stratum_col, "mean", "ci_lower", "ci_upper", "n"],
    )
    # Order strata by token-count position.
    out["_rank"] = out[stratum_col].map(stratum_sort_key)
    ordered = out.sort_values([group_col, "_rank"])
    return ordered.drop(columns="_rank").reset_index(drop=True)


def save_figure(fig: Figure, run_dir: Path, name: str) -> str:
    """Save a figure as ``assets/<name>.png`` in the run folder.

    Returns the path relative to the run folder.
    """
    assets_dir = Path(run_dir) / ASSETS_DIRNAME
    assets_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"{ASSETS_DIRNAME}/{name}.png"
    fig.savefig(assets_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rel_path
