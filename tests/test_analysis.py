"""Tests for bhc_experiment.analysis.

Tests operate on synthetic DataFrames, no disk I/O.
"""

import pandas as pd
import pytest

from bhc_experiment.analysis import (
    compose_series_labels,
    metric_columns,
    series_metric_means,
    series_metric_means_with_n,
)


class TestComposeSeriesLabels:
    @pytest.mark.parametrize(
        ("runs", "short"),
        [
            # Same selection, different prompts: the prompt tells them apart.
            (
                ["all-records/unformatted-prompt", "all-records/brief-prompt"],
                ["unformatted-prompt", "brief-prompt"],
            ),
            # Same prompt, different selections: the selection tells them apart.
            (
                ["all-records/brief-prompt", "oracle-n-0/brief-prompt"],
                ["all-records", "oracle-n-0"],
            ),
            (
                ["all-records/unformatted-prompt", "oracle-n-0/brief-prompt"],
                ["all-records/unformatted-prompt", "oracle-n-0/brief-prompt"],
            ),
            # One run has nothing to tell apart, so it keeps its whole name.
            (["all-records/brief-prompt"], ["all-records/brief-prompt"]),
        ],
    )
    def test_labels_each_run_by_what_differs(self, runs: list[str], short: list[str]):
        df = pd.DataFrame({"run": runs, "model": ["gemma_4_e4b"] * len(runs)})
        out = compose_series_labels(df)
        assert out["run_short"].tolist() == short
        assert out["series_label"].tolist() == [f"{s} \u00b7 gemma_4_e4b" for s in short]


class TestMetricColumns:
    def test_keeps_slash_named_metrics_in_order_excludes_the_rest(self):
        # A metric is a `{name}/{submetric}` scorer key. token_stratum and batch
        # are not metrics, even though batch's value contains " / ".
        df = pd.DataFrame(
            {
                "token_stratum": ["8K"],
                "batch": ["exp / run"],
                # Real metrics, deliberately not in alphabetical order.
                "sapbert/f1": [0.8],
                "rouge1/f1": [0.4],
            }
        )
        assert metric_columns(df) == ["sapbert/f1", "rouge1/f1"]


class TestSeriesMetricMeans:
    def _sample(self) -> pd.DataFrame:
        # Two series, two admissions each, with both metric and cohort columns.
        return pd.DataFrame(
            {
                "series_label": [
                    "unformatted \u00b7 gemma_4_e4b",
                    "unformatted \u00b7 gemma_4_e4b",
                    "brief \u00b7 gemma_4_e4b",
                    "brief \u00b7 gemma_4_e4b",
                ],
                "sapbert/f1": [0.8, 0.6, 0.9, 0.7],
                "rouge1/f1": [0.4, 0.2, 0.5, 0.3],
                "bhc_category": ["narrative", "domain_organized", "narrative", "domain_organized"],
            }
        )

    def test_returns_one_row_per_series(self):
        out = series_metric_means(self._sample(), ["sapbert/f1", "rouge1/f1"])
        assert sorted(out.index.tolist()) == [
            "brief \u00b7 gemma_4_e4b",
            "unformatted \u00b7 gemma_4_e4b",
        ]

    def test_mean_per_series(self):
        out = series_metric_means(self._sample(), ["sapbert/f1"])
        assert out.loc["unformatted \u00b7 gemma_4_e4b", "sapbert/f1"] == pytest.approx(0.7)
        assert out.loc["brief \u00b7 gemma_4_e4b", "sapbert/f1"] == pytest.approx(0.8)

    def test_stratified_multi_index(self):
        out = series_metric_means(self._sample(), ["sapbert/f1"], stratify_by="bhc_category")
        assert out.index.names == ["series_label", "bhc_category"]
        # Each of 2 series has both strata, so 4 rows total.
        assert len(out) == 4

    def test_stratified_mean(self):
        out = series_metric_means(self._sample(), ["sapbert/f1"], stratify_by="bhc_category")
        assert out.loc[
            ("unformatted \u00b7 gemma_4_e4b", "narrative"), "sapbert/f1"
        ] == pytest.approx(0.8)
        assert out.loc[
            ("unformatted \u00b7 gemma_4_e4b", "domain_organized"), "sapbert/f1"
        ] == pytest.approx(0.6)


class TestSeriesMetricMeansWithN:
    def _sample(self) -> pd.DataFrame:
        # Series "a" scored both its admissions. Series "b" has a generation that
        # did not finish, so that admission has no score.
        return pd.DataFrame(
            {
                "series_label": ["a", "a", "b", "b"],
                "sapbert/f1": [0.8, 0.6, 0.9, None],
                "bhc_category": ["narrative", "domain_organized"] * 2,
            }
        )

    def test_n_counts_the_scored_admissions_behind_each_mean(self):
        out = series_metric_means_with_n(self._sample(), ["sapbert/f1"])
        assert out.loc["a", "n"] == 2
        assert out.loc["b", "n"] == 1

    def test_stratified_n_belongs_to_its_own_stratum(self):
        out = series_metric_means_with_n(self._sample(), ["sapbert/f1"], stratify_by="bhc_category")
        assert out.loc[("b", "narrative"), "n"] == 1
        # The stratum with only the unfinished generation scored nothing.
        assert out.loc[("b", "domain_organized"), "n"] == 0


def test_timing_breakdown_columns_are_not_metrics() -> None:
    df = pd.DataFrame(
        {
            "hadm_id": [1],
            "rouge1/f1": [0.5],
            "prefill_ms": [200.0],
            "decode_ms": [40.0],
            "prefill_tok_s": [100.0],
            "decode_tok_s": [50.0],
        }
    )
    metrics = metric_columns(df)
    assert "rouge1/f1" in metrics
    assert "prefill_ms" not in metrics
    assert "decode_ms" not in metrics
    assert "prefill_tok_s" not in metrics
    assert "decode_tok_s" not in metrics
