"""Aggregation helpers for experiment EDA and dissertation table exports.

Scope: pure functions over a per-admission DataFrame. Series label composition,
metric column detection, per-series aggregates.
"""

from collections.abc import Iterable, Sequence

import pandas as pd

from bhc_experiment.metrics import SAPBERT_F1

_SERIES_LABEL_SEP = " \u00b7 "

_DEFAULT_METRIC = SAPBERT_F1


def compose_series_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add display-friendly series label columns to a per-admission DataFrame.

    A series is one line in a figure or one row in a table, which takes both
    the run and the model to identify.

    Reads ``run`` and ``model`` and writes:

    - ``run_short``: ``run`` without the path parts every run in the DataFrame
      shares, so two runs that differ only in the prompt are labeled by their prompts.
    - ``series_label``: ``run_short`` + separator + ``model``.

    Mutates ``df`` and returns it.
    """
    unique_runs = list(dict.fromkeys(df["run"].tolist()))
    run_short = dict(zip(unique_runs, _differing_parts(unique_runs), strict=True))
    df["run_short"] = df["run"].map(run_short)
    df["series_label"] = df["run_short"] + _SERIES_LABEL_SEP + df["model"]
    return df


def metric_columns(df: pd.DataFrame) -> list[str]:
    """Return the metric column names in insertion order.

    A metric column is a scorer output, keyed ``{name}/{submetric}`` (e.g.
    ``sapbert/f1``). Run metadata, token and timing fields, cohort
    attributes, and derived display columns have plain names, so the ``/`` tells
    them apart.
    """
    return [c for c in df.columns if "/" in c]


def default_metric(metrics: list[str], *, preferred: str = _DEFAULT_METRIC) -> str:
    """Pick a default metric: the preferred one if present, else alphabetical first.

    Returns an empty string for an empty metrics list.
    """
    if preferred in metrics:
        return preferred
    return min(metrics) if metrics else ""


def order_metrics(present: Iterable[str], *, priority: Sequence[str]) -> list[str]:
    """Order metric keys with the priority metrics first, then the rest sorted.

    ``present`` is the set of metric keys actually scored (any iterable, deduped).
    ``priority`` lists the metrics that come first, in that order. A priority
    metric absent from ``present`` is skipped. Every remaining present key follows
    in alphabetical order.
    """
    present_set = set(present)
    priority_set = set(priority)
    lead = [metric for metric in priority if metric in present_set]
    rest = sorted(metric for metric in present_set if metric not in priority_set)
    return lead + rest


def series_metric_means(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    stratify_by: str | None = None,
) -> pd.DataFrame:
    """Mean of each metric per series, optionally within a stratum.

    Groups by ``series_label`` (plus ``stratify_by`` when given). One row per
    resulting group, columns are the requested metrics.
    """
    group_cols = ["series_label"] if stratify_by is None else ["series_label", stratify_by]
    return df.groupby(group_cols, dropna=False)[metrics].mean(numeric_only=True)


def series_metric_means_with_n(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    stratify_by: str | None = None,
) -> pd.DataFrame:
    """Mean of each metric per series, plus the admissions behind each mean as column ``n``.

    A generation that did not finish (truncated or errored) has no score, so it is
    in neither the means nor ``n``. Counting on one metric is enough because an
    admission is scored on every metric or on none.
    """
    means = series_metric_means(df, metrics, stratify_by=stratify_by)
    group_cols = ["series_label"] if stratify_by is None else ["series_label", stratify_by]
    scored = df.groupby(group_cols, dropna=False)[default_metric(metrics)].count()
    means.insert(0, "n", scored)
    return means


def _differing_parts(runs: list[str]) -> list[str]:
    """Each run name without the path parts every run shares.

    One run has nothing to tell apart, so it keeps its whole name.
    """
    if len(runs) <= 1:
        return list(runs)
    parts = [run.split("/") for run in runs]
    differing = [i for i in range(len(parts[0])) if len({p[i] for p in parts}) > 1]
    return ["/".join(p[i] for i in differing) for p in parts]
