"""Tests for the scripts/analysis summary tables and figures.

The tests run against a synthetic DataFrame, so no run folder has to be
fixtured on disk.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS_DIR))

import completion as completion_script  # noqa: E402
import cost as cost_script  # noqa: E402
import length as length_script  # noqa: E402
import output_checks as output_checks_script  # noqa: E402
import quality as quality_script  # noqa: E402
import sapbert_window as sapbert_window_script  # noqa: E402

from bhc_experiment import metrics, reporting  # noqa: E402


def _capture_figures(monkeypatch):
    """Spy on reporting.save_figure so a test can inspect the built figure.

    The scripts hand their finished Figure to reporting.save_figure and never
    expose it otherwise, so intercepting that call is the only way to assert
    on the plotted artists (line data, scatter offsets).
    """
    figures: dict[str, object] = {}
    original = reporting.save_figure

    def spy(fig, run_dir, name):
        figures[name] = fig
        return original(fig, run_dir, name)

    monkeypatch.setattr(reporting, "save_figure", spy)
    return figures


def _synthetic_run_df() -> pd.DataFrame:
    """Two models across two strata, with one truncated qwen generation.

    The truncated row has a null score and an output count at the cap,
    so it exercises the completed-only filtering and the cap inference.
    """
    rows = []
    for model in ["gemma", "qwen"]:
        for stratum, base, in_tok in [("4K", 0.40, 4000), ("8K", 0.50, 8000)]:
            for i in range(3):
                truncated = model == "qwen" and stratum == "8K" and i == 0
                rows.append(
                    {
                        "model": model,
                        "token_stratum": stratum,
                        "scispacy-clinical/f1": None if truncated else base + 0.02 * i,
                        "scispacy-clinical/precision": None if truncated else base + 0.05,
                        "scispacy-clinical/recall": None if truncated else base - 0.05,
                        "sapbert/f1": None if truncated else base + 0.40,
                        "truncated": truncated,
                        "input_tokens": in_tok,
                        "output_tokens": 4096 if truncated else 300 + 10 * i,
                        "latency_ms": 50_000.0 + 1_000 * i,
                        "prefill_ms": 40_000.0 + 1_000 * i,
                        "decode_ms": 9_000.0,
                        "prefill_tok_s": 400.0,
                        "decode_tok_s": 30.0,
                    }
                )
    return pd.DataFrame(rows)


class TestCompletionSummaryGolden:
    def test_summary_frame_is_exact(self):
        summary = completion_script._completion_summary(_synthetic_run_df())  # noqa: SLF001
        expected = pd.DataFrame(
            {
                "model": ["gemma", "gemma", "qwen", "qwen"],
                "token_stratum": ["4K", "8K", "4K", "8K"],
                "n": [3, 3, 3, 3],
                "truncated": [0, 0, 0, 1],
                "output_median": [310.0, 310.0, 310.0, 320.0],
                "output_max": [320, 320, 320, 4096],
                "completion_rate": [1.0, 1.0, 1.0, 1.0 - 1.0 / 3.0],
            }
        )
        assert_frame_equal(summary, expected, check_exact=True)

    def test_rate_figure_lines_follow_canonical_stratum_axis(self, monkeypatch, tmp_path: Path):
        figures = _capture_figures(monkeypatch)
        summary = completion_script._completion_summary(_synthetic_run_df())  # noqa: SLF001
        completion_script._rate_figure(summary, tmp_path)  # noqa: SLF001
        ax = figures["completion_rate"].axes[0]
        lines = {line.get_label(): list(line.get_ydata()) for line in ax.get_lines()}
        assert lines == {
            "gemma": [100.0, 100.0],
            "qwen": [100.0, 100.0 * (1.0 - 1.0 / 3.0)],
        }


class TestCostSummaryGolden:
    def test_summary_frame_is_exact(self):
        completed = _synthetic_run_df()
        completed = completed[~completed["truncated"]]
        summary = cost_script._cost_summary(completed)  # noqa: SLF001
        expected = pd.DataFrame(
            {
                "model": ["gemma", "gemma", "qwen", "qwen"],
                "token_stratum": ["4K", "8K", "4K", "8K"],
                "latency_s": [51.0, 51.0, 51.0, 51.5],
                "prefill_s": [41.0, 41.0, 41.0, 41.5],
                "decode_s": [9.0, 9.0, 9.0, 9.0],
                "input_tokens": [4000.0, 8000.0, 4000.0, 8000.0],
                "output_tokens": [310.0, 310.0, 310.0, 315.0],
                "quality": [0.42, 0.52, 0.42, 0.53],
                "n": [3, 3, 3, 2],
            }
        )
        assert_frame_equal(summary, expected, check_exact=True)


class TestQualityFigureGolden:
    def test_scatter_points_drop_null_scores(self, monkeypatch, tmp_path: Path):
        figures = _capture_figures(monkeypatch)
        df = _synthetic_run_df()
        concept = reporting.stratum_summary(df, metrics.PRIMARY_METRIC)
        quality_script._quality_figure(df, concept, tmp_path)  # noqa: SLF001
        ax = figures["quality_vs_size"].axes[0]

        curves = {line.get_label(): list(line.get_ydata()) for line in ax.get_lines()}
        assert curves == {"gemma": [0.42, 0.52], "qwen": [0.42, 0.53]}

        # Collections alternate fill_between band then jittered scatter per model,
        # in model order (gemma, qwen), so odd indices are the scatters.
        gemma_scatter = ax.collections[1]
        qwen_scatter = ax.collections[3]

        assert np.round(gemma_scatter.get_offsets(), 6).tolist() == [
            [-0.06, 0.4],
            [-0.06, 0.42],
            [-0.06, 0.44],
            [0.94, 0.5],
            [0.94, 0.52],
            [0.94, 0.54],
        ]
        # The truncated qwen 8K row has a null score and must drop out.
        assert np.round(qwen_scatter.get_offsets(), 6).tolist() == [
            [0.06, 0.4],
            [0.06, 0.42],
            [0.06, 0.44],
            [1.06, 0.52],
            [1.06, 0.54],
        ]


class TestLengthSummaryGolden:
    """A wrong n, a miscounted target miss, or a row under the wrong stratum would
    read as a normal length gate.
    """

    def test_per_stratum_and_overall_rows(self) -> None:
        lengths = pd.DataFrame(
            {
                "model": ["llama", "gemma", "gemma", "gemma", "gemma"],
                "token_stratum": ["4K", "8K", "4K", "4K", "4K"],
                "generated_words": [300, 700, 200, 240, 400],
                "ratio": [0.6, 1.0, 0.9, 0.2, 0.4],
            }
        )
        expected = pd.DataFrame(
            {
                "model": ["gemma", "gemma", "gemma", "llama", "llama"],
                "token_stratum": ["4K", "8K", "All", "4K", "All"],
                "n": [3, 1, 4, 1, 1],
                "median_words": [240.0, 700.0, 320.0, 300.0, 300.0],
                "under_target": [2, 0, 2, 0, 0],
                "over_target": [0, 1, 1, 0, 0],
                "median_ratio": [0.4, 1.0, 0.65, 0.6, 0.6],
                "under_half": [2, 0, 2, 0, 0],
            }
        )
        assert_frame_equal(length_script.length_summary(lengths), expected, check_dtype=False)


class TestOutputChecksGolden:
    """A miscounted echo or missing-information line would read as a passed or failed gate."""

    def test_title_echo_reads_the_first_nonblank_line_only(self) -> None:
        assert output_checks_script.title_echo("\n**Brief Hospital Course:**\nThe patient...")
        assert output_checks_script.title_echo("# Brief hospital course\nText")
        assert not output_checks_script.title_echo(
            "The patient...\nBrief Hospital Course: see above"
        )

    def test_missing_information_lines_count_short_lines_only(self) -> None:
        text = (
            "Disposition: not documented.\n"
            "Follow-up plans were not specified\n"
            "Oxygen needs were not documented in the final two days of the long stay.\n"
            "Discharged home."
        )
        # The third line has 14 words, so it counts as inline phrasing.
        assert output_checks_script.missing_information_lines(text) == 2

    def test_header_lines_count_hash_and_bold_only_lines(self) -> None:
        text = "# Sepsis:\nTreated.\n**Plan**\n**Sepsis**: treated with antibiotics.\n## AKI"
        assert output_checks_script.header_lines(text) == 3


class TestSapbertWindowGolden:
    """A wrong share over the window would misstate how much of a text SapBERT never scores."""

    def test_share_over_the_content_limit(self) -> None:
        counts = pd.DataFrame(
            {
                "source": ["reference", "reference", "reference", "gemma", "gemma"],
                "bhc_category": ["narrative"] * 5,
                "wordpieces": [400, 510, 511, 300, 900],
            }
        )
        expected = pd.DataFrame(
            {
                "source": ["gemma", "reference"],
                "bhc_category": ["narrative", "narrative"],
                "n": [2, 3],
                "median_wordpieces": [600.0, 510.0],
                "over_window": [1, 1],
                "share_over_window": [0.5, 1 / 3],
            }
        )
        result = sapbert_window_script.window_summary(counts)
        assert_frame_equal(result, expected, check_dtype=False)
