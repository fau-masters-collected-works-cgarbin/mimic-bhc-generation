"""Statistical tools for turning per-admission scored results into defensible claims.

``paired_bootstrap_ci`` tests whether condition B differs from condition A on
a metric for each group. ``paired_delta_ci`` does the same for two conditions
already on one row. The ``bootstrap_*_ci`` functions give intervals for a
single mean, median, or slope.

How to read a confidence interval: if it excludes zero, we can claim the
effect. If it crosses zero, the data is consistent with zero or with an effect
too small to detect at this sample size. That is a failure to detect, and says
nothing about whether an effect exists.

These functions require complete data: a null in a requested metric raises,
naming the first offending admissions.

I'm not great with statistics. These are some references that helped me
understand the concepts and apply them correctly.

    Percentile bootstrap CIs, from scratch: OpenIntro, Introduction to Modern
        Statistics (2e), Ch. 12.
        https://openintro-ims.netlify.app/foundations-bootstrapping.html
    Paired bootstrap for comparing two systems: Koehn 2004.
        https://aclanthology.org/W04-3250/
    Why resampling correlated observations as if independent gives false
        precision (pseudoreplication): Lazic 2010.
        https://pmc.ncbi.nlm.nih.gov/articles/PMC2817684/
    Reading a CI that crosses zero: Altman and Bland 1995 and Greenland et al. 2016.
        https://pmc.ncbi.nlm.nih.gov/articles/PMC2550545/
        https://escholarship.org/uc/item/6sg3f506
"""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

_BOOTSTRAP_COLUMNS = ["mean_a", "mean_b", "delta", "ci_lower", "ci_upper", "n_pairs"]


def _format_hadm_list(hadm_ids: Iterable[object]) -> str:
    """Render up to 5 hadm_ids for inclusion in a diagnostic error message."""
    items = list(hadm_ids)[:5]
    return ", ".join(str(h) for h in items)


def _raise_if_metric_has_nulls(df: pd.DataFrame, metric: str, pair_col: str) -> None:
    """Raise ValueError listing the first 5 admissions with null ``metric``.

    A null metric means scoring did not complete for that admission. Fix the
    gap in the scoring step before calling.
    """
    null_mask = df[metric].isna()
    if not null_mask.any():
        return
    offenders = df.loc[null_mask, pair_col].unique()
    msg = (
        f"metric '{metric}' has {int(null_mask.sum())} null values across "
        f"{len(offenders)} admissions. First 5: [{_format_hadm_list(offenders)}]. "
        "Re-run scoring for these admissions, or drop them from the input before calling."
    )
    raise ValueError(msg)


def _raise_if_pairs_incomplete(
    pivot: pd.DataFrame,
    contrast_col: str,
    contrast_a: str,
    contrast_b: str,
    group_value: object,
    *,
    is_grouped: bool,
) -> None:
    """Raise if a whole variant is missing or an admission ran only one variant.

    Both point to incomplete scoring.
    """
    if contrast_a not in pivot.columns or contrast_b not in pivot.columns:
        msg = (
            f"contrast values '{contrast_a}' and/or '{contrast_b}' missing from "
            f"contrast column '{contrast_col}'. Found: {sorted(pivot.columns.tolist())}."
        )
        raise ValueError(msg)
    unpaired = pivot[contrast_a].isna() | pivot[contrast_b].isna()
    if not unpaired.any():
        return
    offenders = pivot.index[unpaired].tolist()
    group_label = f" in group '{group_value}'" if is_grouped else ""
    msg = (
        f"{len(offenders)} admissions{group_label} are missing from contrast "
        f"'{contrast_a}' or '{contrast_b}'. First 5: [{_format_hadm_list(offenders)}]. "
        "Either re-run the missing model or drop these admissions from the input "
        "before calling."
    )
    raise ValueError(msg)


def _two_sided_percentiles(confidence: float) -> tuple[float, float]:
    """Return the (lower, upper) percentile cut points for an equal-tailed CI.

    An equal-tailed interval leaves alpha/2 of the probability in each tail. The
    100x factor maps those proportions onto ``numpy.percentile``'s 0-100 scale.
    """
    alpha = 1.0 - confidence
    return 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)


def _bootstrap_one_group(
    pivot: pd.DataFrame,
    contrast_a: str,
    contrast_b: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    lower_q: float,
    upper_q: float,
) -> dict[str, float | int]:
    """Compute point estimate and confidence interval for one group's delta.

    Input is a paired table for one group (e.g. one model): rows are
    admissions, columns are the two variants being compared.
    """
    deltas = (pivot[contrast_b] - pivot[contrast_a]).to_numpy()
    n_pairs = len(deltas)
    # Each row is one draw of the paired deltas, with replacement. The spread
    # of their means is the CI.
    resample_indices = rng.integers(0, n_pairs, size=(n_bootstrap, n_pairs))
    resampled_deltas = deltas[resample_indices]
    # One mean delta per resample. The spread of these is the CI.
    bootstrap_delta_means = resampled_deltas.mean(axis=1)
    observed_delta_mean = float(deltas.mean())
    ci_lower_bound = float(np.percentile(bootstrap_delta_means, lower_q))
    ci_upper_bound = float(np.percentile(bootstrap_delta_means, upper_q))
    return {
        "mean_a": float(pivot[contrast_a].mean()),
        "mean_b": float(pivot[contrast_b].mean()),
        "delta": observed_delta_mean,
        "ci_lower": ci_lower_bound,
        "ci_upper": ci_upper_bound,
        "n_pairs": int(n_pairs),
    }


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Percentile bootstrap confidence interval for the mean of one group.

    The single-group counterpart to ``paired_bootstrap_ci``. Resamples with
    replacement for a distribution-free CI, useful when a normal
    approximation would be fragile (small, uneven samples).

    A single observation collapses the CI to a point, since every resample
    is that one value.
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        msg = "cannot bootstrap the mean of an empty sequence"
        raise ValueError(msg)
    lower_q, upper_q = _two_sided_percentiles(confidence)
    rng = np.random.default_rng(seed)
    n = data.size
    # Each row of resample_indices is one re-draw of the values WITH REPLACEMENT.
    resample_indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_values = data[resample_indices]
    # One mean per resample. The spread of these is the CI.
    bootstrap_means = resampled_values.mean(axis=1)
    observed_mean = float(data.mean())
    ci_lower_bound = float(np.percentile(bootstrap_means, lower_q))
    ci_upper_bound = float(np.percentile(bootstrap_means, upper_q))
    return {
        "mean": observed_mean,
        "ci_lower": ci_lower_bound,
        "ci_upper": ci_upper_bound,
        "n": int(n),
    }


MIN_SLOPE_POSITIONS = 2


def _ols_slopes(index: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Least-squares slope of each row of ``values`` on the matching row of ``index``.

    Vectorized over rows so one call covers every bootstrap resample. A row whose
    positions are all identical returns nan.
    """
    index_centered = index - index.mean(axis=1, keepdims=True)
    values_centered = values - values.mean(axis=1, keepdims=True)
    spread = (index_centered**2).sum(axis=1)
    covariance = (index_centered * values_centered).sum(axis=1)
    return np.divide(covariance, spread, out=np.full_like(covariance, np.nan), where=spread > 0)


def bootstrap_slope_ci(
    index: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Percentile bootstrap confidence interval for a least-squares slope.

    ``index`` is an ordinal position, so the slope is a step per position.

    Resamples observations with replacement and refits the slope on each draw,
    so the interval reflects how much the trend depends on which admissions the
    cohort happens to have.

    A noiseless line collapses the CI to a point, since every resample fits it.
    """
    positions = np.asarray(index, dtype=float)
    data = np.asarray(values, dtype=float)
    if positions.size == 0:
        msg = "cannot fit a slope through an empty sequence"
        raise ValueError(msg)
    if positions.size != data.size:
        msg = f"index and values must be the same length, got {positions.size} and {data.size}"
        raise ValueError(msg)
    if np.unique(positions).size < MIN_SLOPE_POSITIONS:
        msg = "a slope needs at least two distinct index positions"
        raise ValueError(msg)
    null_at = np.flatnonzero(np.isnan(positions) | np.isnan(data))
    if null_at.size:
        first = _format_hadm_list(null_at[:5])
        msg = f"cannot fit a slope through nulls, first at positions {first}"
        raise ValueError(msg)

    lower_q, upper_q = _two_sided_percentiles(confidence)
    rng = np.random.default_rng(seed)
    n = positions.size
    # Each row of resample_indices is one re-draw of the observations WITH REPLACEMENT.
    resample_indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_slopes = _ols_slopes(positions[resample_indices], data[resample_indices])
    observed_slope = float(_ols_slopes(positions[None, :], data[None, :])[0])
    ci_lower_bound = float(np.nanpercentile(bootstrap_slopes, lower_q))
    ci_upper_bound = float(np.nanpercentile(bootstrap_slopes, upper_q))
    return {
        "slope": observed_slope,
        "ci_lower": ci_lower_bound,
        "ci_upper": ci_upper_bound,
        "n": int(n),
    }


def bootstrap_median_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Percentile bootstrap confidence interval for the median of one group.

    The median counterpart to ``bootstrap_mean_ci``, for skewed data where a few
    extreme values would drag the mean.

    A single observation collapses the CI to a point, since every resample is that
    one value.
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        msg = "cannot bootstrap the median of an empty sequence"
        raise ValueError(msg)
    lower_q, upper_q = _two_sided_percentiles(confidence)
    rng = np.random.default_rng(seed)
    n = data.size
    resample_indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_medians = np.median(data[resample_indices], axis=1)
    return {
        "median": float(np.median(data)),
        "ci_lower": float(np.percentile(bootstrap_medians, lower_q)),
        "ci_upper": float(np.percentile(bootstrap_medians, upper_q)),
        "n": int(n),
    }


def paired_bootstrap_ci(
    df: pd.DataFrame,
    metric: str,
    *,
    contrast_col: str,
    contrast_a: str,
    contrast_b: str,
    group_col: str | None = None,
    pair_col: str | list[str] = "hadm_id",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """For each group, test whether variant b differs from variant a on the metric.

    A group is usually a model (e.g., gpt-5.4-mini) or a cohort stratum
    (e.g., bhc_category=narrative). The output has one row per group with
    mean of a, mean of b, the observed delta (b minus a), and a 95 percent
    confidence interval on the delta. See the module docstring for how to
    read the interval.

    Pairs observations by admission so admission-level difficulty cancels
    out of the comparison. Resamples admissions with replacement (the
    bootstrap) to build the CI without assuming a normal distribution
    of deltas.

    Requires complete paired input: raises if ``metric`` has null values, or
    if an admission has a score on only one side of the contrast within a
    group. ``contrast_col`` and ``group_col`` take discrete values. See
    ``compose_series_labels`` in the analysis module for the standard labels.

    ``pair_col`` can be a single column or a list of columns. Use a list
    when ``hadm_id`` alone does not uniquely identify a row within the
    group. For example, if each admission has been scored under multiple
    models and the group is not the model column, pass
    ``pair_col=["hadm_id", "model"]`` so each (admission, model)
    pair contributes its own delta.

    WARNING: Do not mix models and other groupings in the same analysis.
    With ``model`` in ``pair_col`` the group spans several models,
    so each admission contributes one delta per model and the bootstrap
    resamples those deltas as if independent. They are not, since a
    single admission's deltas share its difficulty, so the interval comes
    out narrower than the data warrant. Read such an interval as
    unclustered and optimistic. For a clustered read, make the model the
    group and pair on ``hadm_id`` alone so each admission is a single
    delta.
    """
    if df.empty:
        return pd.DataFrame(columns=_BOOTSTRAP_COLUMNS)

    # Both pair_col (possibly a single name) and its list form pair_cols are used
    # below: pair_cols drives the column-set checks, the pivot uses raw pair_col.
    pair_cols = [pair_col] if isinstance(pair_col, str) else list(pair_col)
    required = set(pair_cols) | {contrast_col, metric}
    if group_col is not None:
        required.add(group_col)

    working = df[sorted(required)].copy()
    _raise_if_metric_has_nulls(working, metric, pair_cols[0])

    lower_q, upper_q = _two_sided_percentiles(confidence)

    if group_col is None:
        groups = [(None, working)]
    else:
        groups = list(working.groupby(group_col, dropna=False, sort=True))

    results = []
    for group_value, group_df in groups:
        pivot = group_df.pivot(index=pair_col, columns=contrast_col, values=metric)
        _raise_if_pairs_incomplete(
            pivot,
            contrast_col,
            contrast_a,
            contrast_b,
            group_value,
            is_grouped=group_col is not None,
        )
        # Fresh generator per group so a group's interval depends only on the seed
        # and its own paired deltas, whichever other groups share the call and in
        # whatever order. Keeps a grouped call in agreement with per-group calls elsewhere.
        rng = np.random.default_rng(seed)
        row = _bootstrap_one_group(
            pivot, contrast_a, contrast_b, rng, n_bootstrap, lower_q, upper_q
        )
        if group_col is not None:
            row[group_col] = group_value
        results.append(row)

    out = pd.DataFrame(results)
    if group_col is not None:
        out = out[[group_col, *_BOOTSTRAP_COLUMNS]]
    return out


def paired_delta_ci(
    df: pd.DataFrame,
    *,
    base_col: str,
    treat_col: str,
    group_col: str,
    pair_cols: str | list[str],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Per group, the paired base and treatment means, their delta, and a bootstrap CI.

    Wide-form wrapper over ``paired_bootstrap_ci`` for callers that already have
    both conditions on the same row (``base_col`` and ``treat_col``), such as two
    runs merged per admission. It melts to the long form the bootstrap expects,
    so the interval logic lives in one place. Rows missing either value are
    dropped, so the means and the CI cover the same paired set.

    ``pair_cols`` identifies a pair within a group: a single column when it is
    unique there, or a list when it is not. A list that spans several models
    gives each admission one delta per model, which the bootstrap resamples as
    if independent. See the warning in ``paired_bootstrap_ci``.

    The point estimates are plain means of the two columns and their
    difference. Only the CI comes from resampling.
    """
    pair_list = [pair_cols] if isinstance(pair_cols, str) else list(pair_cols)
    complete = df.dropna(subset=[base_col, treat_col])
    long = complete.melt(
        id_vars=[*pair_list, group_col],
        value_vars=[base_col, treat_col],
        var_name="_side",
        value_name="_value",
    )
    long["_side"] = long["_side"].map({base_col: "base", treat_col: "treat"})
    out = paired_bootstrap_ci(
        long,
        "_value",
        contrast_col="_side",
        contrast_a="base",
        contrast_b="treat",
        group_col=group_col,
        pair_col=pair_list,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    return out.rename(columns={"mean_a": "base_mean", "mean_b": "treat_mean", "n_pairs": "n"})


def annotate_ci_excludes_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean ``excludes_zero`` column to a CI table.

    The flag is true when the interval does not cross zero,
    i.e. when the sign of the effect is distinguishable from sampling
    noise.
    """
    out = df.copy()
    out["excludes_zero"] = (out["ci_lower"] > 0) | (out["ci_upper"] < 0)
    return out
