"""EDA page for experiment results.

Loads one or two runs, joins each with its cohort, and renders
interactive tabs with a shared sidebar of filters and stratification.

Entry: uv run streamlit run app/results_eda.py
"""

from collections.abc import Callable
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from bhc_experiment.analysis import (
    compose_series_labels,
    default_metric,
    metric_columns,
    order_metrics,
    series_metric_means_with_n,
)
from bhc_experiment.metrics import REFERENCE_MEDCON_F1, SAPBERT_F1, SCISPACY_CLINICAL_F1
from bhc_experiment.results_reader import (
    discover_runs,
    load_run_df,
    run_cli_parameters,
)
from mimic_data_prep.mimic import RESULTS_DIR

st.set_page_config(page_title="Experiment EDA", layout="wide")

# Metrics charted by default, in this order. Concept F1 leads because the
# study's claims rest on it, the QuickUMLS reference implementation
# follows as the literature-units check, and the embedding metric corroborates.
_DEFAULT_CHART_METRICS: tuple[str, ...] = (
    SCISPACY_CLINICAL_F1,
    REFERENCE_MEDCON_F1,
    SAPBERT_F1,
)

# Categorical cohort columns eligible as the stratification dropdown and as
# sidebar multi-select filters. Order here drives display order. The cohort
# sampler balances per-stratum counts for the first three only, so the rest can
# facet into sparse panels, and `service` alone has around 20 levels.
_CATEGORICAL_AXES: tuple[str, ...] = (
    "bhc_category",
    "los_quartile",
    "note_category_availability",
    "numbering_convention",
    "gender",
    "service",
    "mesh_age_group",
)

# Continuous cohort attributes exposed as sidebar range sliders. They filter but
# never stratify, because nothing bins them at runtime. Their effect on a metric
# is read off the correlation scatter instead.
_CONTINUOUS_AXES: tuple[str, ...] = (
    "los_days",
    "age",
    "record_token_count",
    "bhc_word_count",
    "note_count",
    "lab_count",
    "prescription_count",
    "charting_days",
)

# Minimum group size below which per-group distribution summaries are
# unreliable. Five is the conventional floor for quartile-based summaries.
_UNRELIABLE_GROUP_N = 5

# Dot-opacity bounds for strip/scatter overlays. Small groups render vivid,
# large groups fade, so overlapping dots read as density instead of a solid mass.
_FULLY_OPAQUE_GROUP_N = 15.0
_MIN_DOT_ALPHA = 0.05

# Display label for None/NaN values in categorical multi-selects. Keeps null
# admissions visible as an explicit, toggleable filter option.
_NULL_SENTINEL = "(unset)"

# Separator between the metric and the series in composite column labels.
_LABEL_SEP = " · "


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def _load_run_df(run_dir: str) -> pd.DataFrame:
    """Streamlit-cached wrapper over the shared loader."""
    return load_run_df(Path(run_dir))


@st.cache_data
def _load_cohort_file(run_dir: str) -> str:
    """Streamlit-cached read of the cohort a run was generated on."""
    return run_cli_parameters(Path(run_dir)).cohort_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chart_facet_kwargs(*, by_run: bool, by_strat: str | None) -> dict:
    """Return the Altair facet kwargs for the active facets.

    When both are active, run is the column and stratum is the row, so
    runs stay a left-right comparison.
    """
    kwargs: dict = {}
    if by_run:
        kwargs["column"] = alt.Column("run:N", title="Run")
    if by_strat is not None:
        strat_facet = alt.Column(f"{by_strat}:N", title=by_strat)
        if by_run:
            kwargs["row"] = strat_facet
        else:
            kwargs["column"] = strat_facet
    return kwargs


def _apply_filters(
    df: pd.DataFrame,
    *,
    series_labels: list[str],
    categorical_selections: dict[str, list[str]],
    continuous_ranges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Keep the rows matching every selection, intersecting all three filters.

    Clearing the series selection keeps nothing, since no series matches. Clearing a
    categorical selection instead leaves that column unfiltered. Admissions with
    a null categorical value survive only while that column's selection still
    includes the null sentinel.
    """
    filtered = df[df["series_label"].isin(series_labels)]
    for col, allowed in categorical_selections.items():
        if col not in filtered.columns or not allowed:
            continue
        non_null_allowed = [v for v in allowed if v != _NULL_SENTINEL]
        mask = filtered[col].isin(non_null_allowed)
        if _NULL_SENTINEL in allowed:
            mask = mask | filtered[col].isna()
        filtered = filtered[mask]
    for col, (lo, hi) in continuous_ranges.items():
        if col in filtered.columns:
            filtered = filtered[(filtered[col] >= lo) & (filtered[col] <= hi)]
    return filtered


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.title("Experiment EDA")

st.sidebar.header("Run Selection")
runs = discover_runs(RESULTS_DIR)
if not runs:
    st.sidebar.error(f"No runs found in {RESULTS_DIR}")
    st.stop()

# The label matches the run column the loader writes, so it also filters rows.
run_labels = [str(run_dir.relative_to(RESULTS_DIR)) for run_dir in runs]
selected_run_idx = st.sidebar.selectbox(
    "Run",
    options=list(range(len(runs))),
    format_func=lambda i: run_labels[i],
    key="run_selector",
)

compare_mode = st.sidebar.toggle("Compare to another run", key="compare_mode")
run2_idx: int | None = None
if compare_mode:
    remaining = [i for i in range(len(runs)) if i != selected_run_idx]
    if not remaining:
        st.sidebar.info("Only one run discovered. Nothing to compare against.")
        compare_mode = False
    else:
        run2_idx = st.sidebar.selectbox(
            "Second run",
            options=remaining,
            format_func=lambda i: run_labels[i],
            key="run2_selector",
        )

run_a = run_labels[selected_run_idx]
df_a = _load_run_df(str(runs[selected_run_idx]))
if df_a.empty:
    st.warning(
        "No results loaded. Either the run has no run_record.yaml files, or "
        "the cohort file pointed to by its run records is missing."
    )
    st.stop()

run_b: str | None = None
if compare_mode and run2_idx is not None:
    run_b = run_labels[run2_idx]
    df_b = _load_run_df(str(runs[run2_idx]))
    if df_b.empty:
        st.warning(f"Second run {run_b} has no loadable results. Comparison disabled.")
        compare_mode = False
        run_b = None
        df = df_a
    else:
        # load_run_df labels each run alone, which leaves the full run names.
        df = compose_series_labels(pd.concat([df_a, df_b], ignore_index=True))
else:
    df = df_a

cohort_a = _load_cohort_file(str(runs[selected_run_idx]))
caption_lines = [
    f"Run A **{run_a}**  |  cohort: `{cohort_a}`",
]
if compare_mode and run_b is not None:
    cohort_b = _load_cohort_file(str(runs[run2_idx]))
    caption_lines.append(f"Run B **{run_b}**  |  cohort: `{cohort_b}`")
st.caption("  \n".join(caption_lines))

# Widget keys that name the loaded run reset their selection when the run
# changes. Otherwise Streamlit keeps a selection whose values do not exist in
# the new run, which silently filters out every admission.
run_scope = run_a if run_b is None else f"{run_a}||{run_b}"

# In compare mode, intersect per-run metric sets so drill-down tabs
# always have the metric on both sides.
if compare_mode and run_b is not None:
    metrics_a = set(metric_columns(df_a))
    metrics_b = set(metric_columns(df_b))
    metrics = sorted(metrics_a & metrics_b)
else:
    metrics = metric_columns(df)
if not metrics:
    st.error("No metric columns present. Did the scorer run?")
    st.stop()
metric_options = order_metrics(metrics, priority=_DEFAULT_CHART_METRICS)
preselected_metrics = [m for m in _DEFAULT_CHART_METRICS if m in metrics]
selected_metrics = st.sidebar.multiselect(
    "Metrics charted",
    options=metric_options,
    default=preselected_metrics or [default_metric(metrics)],
    key=f"metric_filter::{run_scope}",
)
missing_defaults = [m for m in _DEFAULT_CHART_METRICS if m not in metrics]
if missing_defaults:
    st.sidebar.caption(f"Not scored in this run: {', '.join(missing_defaults)}")

available_strata = [c for c in _CATEGORICAL_AXES if c in df.columns]
strat_options = ["none", *available_strata]
strat_choice = st.sidebar.selectbox(
    "Stratification axis",
    options=strat_options,
    key="strat_selector",
)
strat_column: str | None = None if strat_choice == "none" else strat_choice

# Categorical multi-selects, skipping the column used for stratification.
st.sidebar.subheader("Categorical filters")
categorical_selections: dict[str, list[str]] = {}
for col in available_strata:
    if col == strat_column:
        continue
    non_null_vals = sorted(set(df[col].dropna().unique()))
    has_null = bool(df[col].isna().any())
    unique_vals = non_null_vals + ([_NULL_SENTINEL] if has_null else [])
    if not unique_vals:
        continue
    categorical_selections[col] = st.sidebar.multiselect(
        col,
        options=unique_vals,
        default=unique_vals,
        key=f"cat_{col}",
    )

st.sidebar.subheader("Continuous filters")
continuous_ranges: dict[str, tuple[float, float]] = {}
for col in _CONTINUOUS_AXES:
    if col not in df.columns:
        continue
    series = df[col].dropna()
    if series.empty:
        continue
    lo, hi = float(series.min()), float(series.max())
    if lo == hi:
        continuous_ranges[col] = (lo, hi)
        continue
    continuous_ranges[col] = st.sidebar.slider(
        col,
        min_value=lo,
        max_value=hi,
        value=(lo, hi),
        key=f"cont_{col}",
    )

st.sidebar.subheader("Series")
series_options = sorted(df["series_label"].unique())
selected_series = st.sidebar.multiselect(
    "Series included",
    options=series_options,
    default=series_options,
    key=f"series_filter::{run_scope}",
)

filtered = _apply_filters(
    df,
    series_labels=selected_series,
    categorical_selections=categorical_selections,
    continuous_ranges=continuous_ranges,
)

if filtered.empty:
    st.warning("No admissions match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

(
    tab_summary,
    tab_distribution,
    tab_per_admission,
    tab_correlation,
) = st.tabs(
    [
        "Summary",
        "Distribution",
        "Per-admission",
        "Attribute Correlation",
    ]
)


def _require_metrics() -> bool:
    """True when the sidebar has at least one metric, else says so on the tab."""
    if selected_metrics:
        return True
    st.info("No metrics selected. Pick at least one in the sidebar.")
    return False


def _series_means_with_n(frame: pd.DataFrame) -> pd.DataFrame:
    """Uses every metric in the run, so this table stays the one place to see all of
    them at once whatever the sidebar has selected."""
    return series_metric_means_with_n(frame, metrics)


def _render_series_means(means_with_n: pd.DataFrame) -> None:
    """Render the per-series means table, coloring the metric columns only.

    ``n`` is left uncolored so the gradient reads across the metric scores,
    not the sample size.
    """
    metric_cols = [c for c in means_with_n.columns if c != "n"]
    formats = dict.fromkeys(metric_cols, "{:.3f}")
    formats["n"] = "{:.0f}"
    styled = means_with_n.style.background_gradient(axis=0, cmap="YlGn", subset=metric_cols).format(
        formats
    )
    st.dataframe(styled, width="stretch")


def _render_summary_for(frame: pd.DataFrame) -> None:
    if strat_column is None:
        _render_series_means(_series_means_with_n(frame))
        return
    st.caption(f"Faceted by **{strat_column}**")
    for value, group in frame.groupby(strat_column, dropna=False):
        st.markdown(f"**{strat_column} = {value!r}**")
        _render_series_means(_series_means_with_n(group))


def _render_side_by_side(renderer: Callable[[pd.DataFrame], None]) -> None:
    """Render a tabular view once, or side-by-side when compare mode is active."""
    if not compare_mode:
        renderer(filtered)
        return
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{run_a}**")
        renderer(filtered[filtered["run"] == run_a])
    with col_b:
        st.markdown(f"**{run_b}**")
        renderer(filtered[filtered["run"] == run_b])


def _show_summary() -> None:
    st.subheader("Series by metric means")
    _render_side_by_side(_render_summary_for)


def _with_active_facets(cols: list[str]) -> list[str]:
    """Add the stratification axis and, in compare mode, the run column."""
    with_facets = list(cols)
    if strat_column is not None and strat_column not in with_facets:
        with_facets.append(strat_column)
    if compare_mode and "run" not in with_facets:
        with_facets.append("run")
    return with_facets


def _dot_alpha(group_sizes: pd.Series) -> float:
    """Dot opacity for the given per-group counts, fading as the median grows."""
    return max(_MIN_DOT_ALPHA, min(1.0, _FULLY_OPAQUE_GROUP_N / float(group_sizes.median())))


def _render_distribution_for(metric: str) -> None:
    cols = _with_active_facets(["run_short", "model", "hadm_id", metric])
    plot_df = filtered[cols].dropna(subset=[metric])
    if plot_df.empty:
        st.info(f"No non-null values for `{metric}` under current filters.")
        return

    group_cols = _with_active_facets(["run_short", "model"])
    n_by_group = plot_df.groupby(group_cols).size()
    dot_alpha = _dot_alpha(n_by_group)

    min_n = int(n_by_group.min())
    if min_n < _UNRELIABLE_GROUP_N:
        st.caption(
            f"Smallest group has N={min_n} admissions. Treat the distribution as unreliable."
        )

    models_sorted = sorted(plot_df["model"].unique())
    runs_sorted = sorted(plot_df["run_short"].unique())

    # Model on x with run as the offset keeps one visual dimension per
    # experiment dimension. Collapsing the two onto a single per-series axis
    # buries the interaction between them.
    #
    # Ticks show the model's initial: full names collide once the chart is
    # faceted, and the initials are unique across the models in a run. The key
    # under the tab subheader spells them out, and the tooltip shows the name.
    base = alt.Chart(plot_df).encode(
        x=alt.X(
            "model:N",
            # No axis title: it would repeat once per facet, and the key above
            # the charts already says the initials are models.
            title=None,
            sort=models_sorted,
            axis=alt.Axis(
                labelAngle=0,
                labelOverlap=False,
                labelExpr="substring(datum.label, 0, 1)",
            ),
        ),
        y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=False)),
        xOffset=alt.XOffset("run_short:N", sort=runs_sorted),
        color=alt.Color("run_short:N", title="Run", sort=runs_sorted),
    )
    dots = base.mark_circle(size=30, opacity=dot_alpha).encode(
        tooltip=[
            alt.Tooltip("hadm_id:N"),
            alt.Tooltip("run_short:N", title="run"),
            alt.Tooltip("model:N", title="model"),
            alt.Tooltip(f"{metric}:Q", format=".3f"),
        ],
    )
    boxes = base.mark_boxplot(size=20, extent="min-max", opacity=0.8)

    chart = dots + boxes
    facet_kwargs = _chart_facet_kwargs(by_run=compare_mode, by_strat=strat_column)
    if facet_kwargs:
        chart = chart.facet(**facet_kwargs)

    st.altair_chart(chart, width="stretch")


def _show_distribution() -> None:
    st.subheader("Distribution by model")
    if not _require_metrics():
        return
    models = sorted(filtered["model"].unique())
    st.caption("Model axis:  " + "   ".join(f"**{m[0]}** {m}" for m in models))
    for metric in selected_metrics:
        st.markdown(f"**`{metric}`**")
        _render_distribution_for(metric)


def _render_per_admission_for(frame: pd.DataFrame) -> None:
    pivot = frame.pivot_table(
        index="hadm_id",
        columns="series_label",
        values=selected_metrics,
        aggfunc="mean",
    )
    # pivot_table sorts the metric level alphabetically. Restore the sidebar
    # order, then flatten so the score columns join the single-level cohort ones.
    pivot = pivot.reindex(columns=selected_metrics, level=0)
    pivot.columns = [
        f"{metric}{_LABEL_SEP}{series_label}" for metric, series_label in pivot.columns
    ]
    # Cohort attributes are constant per admission, so one value per hadm_id
    # suffices alongside the score matrix.
    cohort_cols = [c for c in _CATEGORICAL_AXES + _CONTINUOUS_AXES if c in frame.columns]
    cohort_view = frame.groupby("hadm_id")[cohort_cols].first()
    combined = cohort_view.join(pivot, how="right")
    st.dataframe(combined, width="stretch")


def _show_per_admission() -> None:
    st.subheader("Per-admission scores across series")
    if not _require_metrics():
        return
    _render_side_by_side(_render_per_admission_for)


def _render_correlation_for(attr: str, metric: str) -> None:
    cols = _with_active_facets(["hadm_id", "series_label", "run_short", "model", attr, metric])
    plot_df = filtered[cols].dropna(subset=[attr, metric])
    if plot_df.empty:
        st.info(f"No rows with both `{attr}` and `{metric}` present.")
        return

    group_cols = _with_active_facets(["series_label"])
    dot_alpha = _dot_alpha(plot_df.groupby(group_cols).size())

    base = alt.Chart(plot_df).encode(
        x=alt.X(f"{attr}:Q", title=attr, scale=alt.Scale(zero=False)),
        y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=False)),
        color=alt.Color("series_label:N", title="Series"),
        tooltip=[
            alt.Tooltip("hadm_id:N"),
            alt.Tooltip("series_label:N", title="series"),
            alt.Tooltip(f"{attr}:Q", format=".2f"),
            alt.Tooltip(f"{metric}:Q", format=".3f"),
        ],
    )
    dots = base.mark_circle(size=40, opacity=dot_alpha)

    # LOESS trend per series makes the correlation visible when the dot cloud is
    # dense. Grouping matches the facets so each panel gets its own fitted trend.
    trends = (
        base.transform_loess(attr, metric, groupby=group_cols)
        .mark_line(opacity=0.6)
        .encode(size=alt.value(1.5))
    )

    chart = dots + trends
    facet_kwargs = _chart_facet_kwargs(by_run=compare_mode, by_strat=strat_column)
    if facet_kwargs:
        chart = chart.facet(**facet_kwargs)

    st.altair_chart(chart, width="stretch")


def _show_attribute_correlation() -> None:
    st.subheader("Metrics vs continuous cohort attribute")
    if not _require_metrics():
        return

    available_attrs = [c for c in _CONTINUOUS_AXES if c in filtered.columns]
    if not available_attrs:
        st.info("No continuous cohort attributes available under current filters.")
        return

    attr = st.selectbox(
        "Continuous attribute (x-axis)",
        options=available_attrs,
        key="correlation_attr",
    )

    for metric in selected_metrics:
        st.markdown(f"**`{metric}`**")
        _render_correlation_for(attr, metric)


with tab_summary:
    _show_summary()

with tab_distribution:
    _show_distribution()

with tab_per_admission:
    _show_per_admission()

with tab_correlation:
    _show_attribute_correlation()
