"""Tests for bhc_experiment.stats.

Tests operate on synthetic DataFrames, with no disk I/O.
"""

import numpy as np
import pandas as pd
import pytest

from bhc_experiment.stats import (
    annotate_ci_excludes_zero,
    bootstrap_mean_ci,
    bootstrap_median_ci,
    bootstrap_slope_ci,
    paired_bootstrap_ci,
    paired_delta_ci,
)


def _paired_df(
    n_admissions: int,
    delta: float,
    seed: int,
    *,
    groups: dict | None = None,
) -> pd.DataFrame:
    """Build synthetic paired data with a known per-group variant effect.

    Each admission contributes one (minimal, structured) pair per group. Minimal
    scores are drawn around a per-admission baseline. Structured scores add
    ``delta`` (or the group-specific delta when ``groups`` is given) plus small
    noise. The per-admission baseline simulates admission-level difficulty.
    """
    rng = np.random.default_rng(seed)
    baselines = rng.normal(0.75, 0.05, size=n_admissions)
    group_deltas = groups if groups is not None else {"_all": delta}
    rows = []
    for i, base in enumerate(baselines):
        for group, group_delta in group_deltas.items():
            rows.append(
                {
                    "hadm_id": i,
                    "variant": "unformatted",
                    "model": group,
                    "score": base + rng.normal(0, 0.01),
                }
            )
            rows.append(
                {
                    "hadm_id": i,
                    "variant": "brief",
                    "model": group,
                    "score": base + group_delta + rng.normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


class TestPairedBootstrapCi:
    def test_returns_one_row_without_group(self):
        df = _paired_df(50, delta=0.02, seed=1)
        out = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
        )
        assert len(out) == 1

    def test_returns_one_row_per_group(self):
        df = _paired_df(50, delta=0.0, seed=2, groups={"gemma_4_e4b": 0.02, "qwen_3_5_4b": -0.01})
        out = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
            group_col="model",
        )
        assert len(out) == 2
        assert set(out["model"]) == {"gemma_4_e4b", "qwen_3_5_4b"}

    def test_point_estimate_matches_raw_mean_delta(self):
        df = _paired_df(100, delta=0.03, seed=4)
        out = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
        )
        pivot = df.pivot(index="hadm_id", columns="variant", values="score")
        expected = (pivot["brief"] - pivot["unformatted"]).mean()
        assert out.iloc[0]["delta"] == pytest.approx(expected)

    def test_ci_brackets_point_estimate(self):
        df = _paired_df(100, delta=0.02, seed=5)
        out = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
        )
        row = out.iloc[0]
        assert row["ci_lower"] <= row["delta"] <= row["ci_upper"]

    def test_identical_variants_yield_zero_delta_and_tight_ci(self):
        df = _paired_df(100, delta=0.0, seed=6)
        out = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
        )
        row = out.iloc[0]
        assert abs(row["delta"]) < 0.005
        assert row["ci_upper"] - row["ci_lower"] < 0.02

    def test_deterministic_with_same_seed(self):
        df = _paired_df(80, delta=0.02, seed=7)
        out_a = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
            seed=123,
        )
        out_b = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
            seed=123,
        )
        assert out_a.iloc[0]["ci_lower"] == out_b.iloc[0]["ci_lower"]
        assert out_a.iloc[0]["ci_upper"] == out_b.iloc[0]["ci_upper"]

    def test_different_seed_produces_different_ci(self):
        df = _paired_df(80, delta=0.02, seed=8)
        out_a = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
            seed=123,
        )
        out_b = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
            seed=456,
        )
        assert out_a.iloc[0]["ci_lower"] != out_b.iloc[0]["ci_lower"]

    def test_group_ci_does_not_depend_on_other_groups(self):
        # A group's interval must depend only on the seed and its own data, not
        # on which other groups share the call. "qwen_3_5_4b" sorts after
        # "gemma_4_e4b", so a single shared generator would give it a different
        # stream here than when it is computed alone.
        df = _paired_df(40, delta=0.0, seed=3, groups={"gemma_4_e4b": 0.02, "qwen_3_5_4b": -0.01})
        common = {
            "metric": "score",
            "contrast_col": "variant",
            "contrast_a": "unformatted",
            "contrast_b": "brief",
            "seed": 123,
        }
        both = paired_bootstrap_ci(df, group_col="model", **common)
        alone = paired_bootstrap_ci(df[df["model"] == "qwen_3_5_4b"], group_col="model", **common)

        qwen_both = both[both["model"] == "qwen_3_5_4b"].iloc[0]
        qwen_alone = alone.iloc[0]
        assert qwen_both["ci_lower"] == qwen_alone["ci_lower"]
        assert qwen_both["ci_upper"] == qwen_alone["ci_upper"]

    def test_null_metric_raises(self):
        df = _paired_df(20, delta=0.02, seed=9)
        df.loc[df.index[0], "score"] = None
        with pytest.raises(ValueError, match="null values"):
            paired_bootstrap_ci(
                df,
                metric="score",
                contrast_col="variant",
                contrast_a="unformatted",
                contrast_b="brief",
            )

    def test_unpaired_admission_raises(self):
        df = _paired_df(20, delta=0.02, seed=11)
        df = df[~((df["hadm_id"] == 3) & (df["variant"] == "brief"))]
        with pytest.raises(ValueError, match="missing from contrast"):
            paired_bootstrap_ci(
                df,
                metric="score",
                contrast_col="variant",
                contrast_a="unformatted",
                contrast_b="brief",
            )

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["hadm_id", "variant", "model", "score"])
        out = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
        )
        assert out.empty
        assert list(out.columns) == ["mean_a", "mean_b", "delta", "ci_lower", "ci_upper", "n_pairs"]

    def test_ci_width_shrinks_with_larger_sample(self):
        small = _paired_df(30, delta=0.02, seed=14)
        large = _paired_df(300, delta=0.02, seed=15)
        out_small = paired_bootstrap_ci(
            small,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
        )
        out_large = paired_bootstrap_ci(
            large,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
        )
        width_small = out_small.iloc[0]["ci_upper"] - out_small.iloc[0]["ci_lower"]
        width_large = out_large.iloc[0]["ci_upper"] - out_large.iloc[0]["ci_lower"]
        assert width_large < width_small


class TestPairedBootstrapCiAdditional:
    def test_composite_pair_col_handles_replicate_dimension(self):
        # Each admission has 3 scores per variant, one per model. Pairing on
        # hadm_id alone would raise a pivot duplicate error.
        # pair_col=["hadm_id", "model"] pairs on (admission, model).
        df = _paired_df(
            30,
            delta=0.02,
            seed=200,
            groups={"gemma_4_e4b": 0.0, "qwen_3_5_4b": 0.0, "llama_3_2_3b": 0.0},
        )
        # 30 admissions x 2 variants x 3 models.
        assert len(df) == 30 * 2 * 3  # admissions * variants * models
        out = paired_bootstrap_ci(
            df,
            metric="score",
            contrast_col="variant",
            contrast_a="unformatted",
            contrast_b="brief",
            pair_col=["hadm_id", "model"],
        )
        assert len(out) == 1
        assert out.iloc[0]["n_pairs"] == 90


class TestPairedDeltaCi:
    """Wide-form wrapper: two conditions on the same row, plus a bootstrap CI."""

    def _wide(self) -> pd.DataFrame:
        # base and treat on one row per (admission, model). Group "g" chosen so
        # its base mean is a round 0.50.
        return pd.DataFrame(
            [
                {"hadm_id": 1, "model": "g", "base": 0.50, "treat": 0.55},
                {"hadm_id": 2, "model": "g", "base": 0.40, "treat": 0.48},
                {"hadm_id": 3, "model": "g", "base": 0.60, "treat": 0.60},
                {"hadm_id": 1, "model": "q", "base": 0.70, "treat": 0.66},
                {"hadm_id": 2, "model": "q", "base": 0.65, "treat": 0.66},
                {"hadm_id": 3, "model": "q", "base": 0.62, "treat": 0.60},
            ]
        )

    def test_means_and_delta_match_manual(self):
        out = paired_delta_ci(
            self._wide(),
            base_col="base",
            treat_col="treat",
            group_col="model",
            pair_cols=["hadm_id"],
        ).set_index("model")
        assert out.loc["g", "base_mean"] == pytest.approx(0.50)
        assert out.loc["g", "treat_mean"] == pytest.approx((0.55 + 0.48 + 0.60) / 3)
        assert out.loc["g", "delta"] == pytest.approx(out.loc["g", "treat_mean"] - 0.50)
        assert out.loc["g", "n"] == 3

    def test_ci_brackets_delta(self):
        out = paired_delta_ci(
            self._wide(),
            base_col="base",
            treat_col="treat",
            group_col="model",
            pair_cols=["hadm_id"],
        ).set_index("model")
        for model in ("g", "q"):
            assert (
                out.loc[model, "ci_lower"] <= out.loc[model, "delta"] <= out.loc[model, "ci_upper"]
            )

    def test_one_row_per_group(self):
        out = paired_delta_ci(
            self._wide(),
            base_col="base",
            treat_col="treat",
            group_col="model",
            pair_cols=["hadm_id"],
        )
        assert set(out["model"]) == {"g", "q"}

    def test_incomplete_rows_drop_from_pairs(self):
        # A row missing one condition does not enter the paired set, so n counts
        # only complete pairs.
        df = pd.concat(
            [
                self._wide(),
                pd.DataFrame([{"hadm_id": 4, "model": "g", "base": None, "treat": 0.9}]),
            ],
            ignore_index=True,
        )
        out = paired_delta_ci(
            df,
            base_col="base",
            treat_col="treat",
            group_col="model",
            pair_cols=["hadm_id"],
        ).set_index("model")
        assert out.loc["g", "n"] == 3

    def test_pooled_with_composite_pair_cols(self):
        # Pooling across models means hadm_id alone is not unique in the group, so
        # the pair is (hadm_id, model).
        df = self._wide()
        df["_pool"] = "all"
        out = paired_delta_ci(
            df,
            base_col="base",
            treat_col="treat",
            group_col="_pool",
            pair_cols=["hadm_id", "model"],
        )
        assert len(out) == 1
        assert out.iloc[0]["n"] == 6


class TestAnnotateCIExcludesZero:
    @pytest.mark.parametrize(
        ("lower", "upper", "verdict"),
        [
            (0.01, 0.05, "excludes"),  # entirely positive
            (-0.05, -0.01, "excludes"),  # entirely negative
            (-0.02, 0.02, "includes"),  # crosses zero
        ],
    )
    def test_classification(self, lower: float, upper: float, verdict: str):
        df = pd.DataFrame({"ci_lower": [lower], "ci_upper": [upper]})
        out = annotate_ci_excludes_zero(df)
        assert out["excludes_zero"].tolist() == [verdict == "excludes"]


class TestBootstrapMeanCi:
    def test_ci_brackets_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.5, 0.1, size=50)
        out = bootstrap_mean_ci(values, seed=7)
        assert out["ci_lower"] <= out["mean"] <= out["ci_upper"]

    def test_deterministic_under_seed(self):
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        first = bootstrap_mean_ci(values, seed=42)
        second = bootstrap_mean_ci(values, seed=42)
        assert first == second

    def test_single_value_collapses_ci_to_point(self):
        # With one observation every resample is that value, so the CI is a point.
        out = bootstrap_mean_ci([0.42], seed=1)
        assert out["mean"] == pytest.approx(0.42)
        assert out["ci_lower"] == pytest.approx(0.42)
        assert out["ci_upper"] == pytest.approx(0.42)
        assert out["n"] == 1

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_mean_ci([], seed=1)


class TestBootstrapMedianCi:
    def test_ci_brackets_median_and_stays_in_range(self):
        rng = np.random.default_rng(0)
        values = rng.uniform(0.0, 1.0, size=40)
        out = bootstrap_median_ci(values, seed=7)
        assert out["ci_lower"] <= out["median"] <= out["ci_upper"]
        assert out["ci_lower"] >= values.min()
        assert out["ci_upper"] <= values.max()

    def test_deterministic_under_seed(self):
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        assert bootstrap_median_ci(values, seed=42) == bootstrap_median_ci(values, seed=42)

    def test_single_value_collapses_ci_to_point(self):
        # With one observation every resample is that value, so the CI is a point.
        out = bootstrap_median_ci([0.42], seed=1)
        assert out["median"] == pytest.approx(0.42)
        assert out["ci_lower"] == pytest.approx(0.42)
        assert out["ci_upper"] == pytest.approx(0.42)
        assert out["n"] == 1

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_median_ci([], seed=1)


class TestBootstrapSlopeCi:
    def test_noiseless_line_recovers_slope_exactly(self):
        # Every resample holding two distinct positions fits the same line, so the CI is a point.
        index = [0, 1, 2, 3, 4, 5, 6] * 4
        values = [0.1 + 0.02 * position for position in index]
        out = bootstrap_slope_ci(index, values, seed=3)
        assert out["slope"] == pytest.approx(0.02)
        assert out["ci_lower"] == pytest.approx(0.02)
        assert out["ci_upper"] == pytest.approx(0.02)
        assert out["n"] == 28

    def test_ci_brackets_slope_under_noise(self):
        rng = np.random.default_rng(0)
        index = np.repeat(np.arange(7), 20)
        values = 0.5 + 0.01 * index + rng.normal(0.0, 0.05, size=index.size)
        out = bootstrap_slope_ci(index, values, seed=7)
        assert out["ci_lower"] <= out["slope"] <= out["ci_upper"]

    def test_flat_data_gives_ci_spanning_zero(self):
        rng = np.random.default_rng(1)
        index = np.repeat(np.arange(7), 20)
        values = 0.5 + rng.normal(0.0, 0.05, size=index.size)
        out = bootstrap_slope_ci(index, values, seed=7)
        assert out["ci_lower"] < 0.0 < out["ci_upper"]

    def test_deterministic_under_seed(self):
        index = [0, 1, 2, 3, 4, 5, 6]
        values = [0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.6]
        first = bootstrap_slope_ci(index, values, seed=42)
        second = bootstrap_slope_ci(index, values, seed=42)
        assert first == second

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            bootstrap_slope_ci([0, 1, 2], [0.1, 0.2], seed=1)

    def test_single_distinct_position_raises(self):
        # A slope is undefined without two positions to measure the rise between.
        with pytest.raises(ValueError, match="distinct"):
            bootstrap_slope_ci([3, 3, 3], [0.1, 0.2, 0.3], seed=1)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_slope_ci([], [], seed=1)

    def test_null_value_raises(self):
        # Without this the point estimate is nan while the interval is computed on
        # whichever resamples happened to miss the null, which reads as a real answer.
        with pytest.raises(ValueError, match="nulls"):
            bootstrap_slope_ci([0, 1, 2, 3], [0.1, np.nan, 0.3, 0.4], seed=1)
