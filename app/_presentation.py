"""Shared presentation helpers for the Streamlit apps."""

import html

import pandas as pd

# Submetric column order: recall sits beside precision so the coverage gap
# (recall below precision means missing content) reads at a glance, f1 last.
_SUBMETRIC_ORDER = ("precision", "recall", "f1")

# pre-wrap preserves newlines from the model output and the reference BHC.
NOTE_STYLE = (
    "font-family: monospace; font-size: 13px; white-space: pre-wrap; "
    "line-height: 1.4; "
    "padding: 10px; border: 1px solid #ddd; border-radius: 4px;"
)

DETAIL_COHORT_ATTRS: tuple[str, ...] = (
    "bhc_category",
    "los_quartile",
    "note_category_availability",
)


def html_escape(text: str) -> str:
    """Escape HTML special characters, preserve newlines as <br>."""
    return html.escape(text).replace("\n", "<br>")


def scores_to_table(scores: dict[str, float]) -> pd.DataFrame:
    """Group flat ``family/submetric`` scores into a family-by-submetric table.

    Scores arrive flat ("sapbert/f1" -> 0.79). One row per family, columns
    precision / recall / f1 so a family's whole picture reads on one line
    instead of across a wall of narrow metric cards. Families that report only
    f1 (ROUGE, rescaled, reference-medcon) leave the other cells empty.
    """
    grouped: dict[str, dict[str, float]] = {}
    for key, value in scores.items():
        family, _, submetric = key.rpartition("/")
        grouped.setdefault(family, {})[submetric] = value

    seen = {s for row in grouped.values() for s in row}
    known = [s for s in _SUBMETRIC_ORDER if s in seen]
    extra = sorted(seen - set(_SUBMETRIC_ORDER))
    submetrics = [*known, *extra]

    rows = []
    for family, row in grouped.items():
        cells = {s: row.get(s) for s in submetrics}
        rows.append({"metric": family, **cells})
    return pd.DataFrame(rows, columns=["metric", *submetrics])
