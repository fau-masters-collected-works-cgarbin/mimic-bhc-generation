"""Tests for the evaluation pipeline (bhc_experiment.evaluator)."""

import pytest

from bhc_experiment import evaluator
from bhc_experiment.evaluator import build_scorers, evaluate_batch

from .conftest import install_fake_scispacy


def _evaluate_one(generated: str, reference: str, scorers: dict) -> dict:
    """Score a single (cand, ref) pair."""
    return evaluate_batch([generated], [reference], scorers)[0]


def _concept_scorers(space: str = "scispacy") -> dict:
    """The concept metric of one concept space, alone."""
    return build_scorers([{"umls_concept": {"space": space}}])


class TestReferenceMedcon:
    def test_passes_reference_before_prediction(self, monkeypatch) -> None:
        # Yim's scorer is asymmetric: its precision divides by the prediction's
        # term count, so a swapped pair still returns a plausible number.
        seen: list[tuple[str, str]] = []

        def _fake_bridge(pairs: list[tuple[str, str]]) -> list[float]:
            seen.extend(pairs)
            return [0.25] * len(pairs)

        monkeypatch.setattr(evaluator, "run_medcon", _fake_bridge)
        scorers = build_scorers([{"reference_medcon": {"name": "reference-medcon"}}])

        scores = evaluate_batch(["generated text"], ["reference text"], scorers)

        assert seen == [("reference text", "generated text")]
        assert scores == [{"reference-medcon/f1": 0.25}]


class TestEvaluateBatch:
    """Tests for the batch scoring contract."""

    def test_scores_multiple_pairs(self) -> None:
        """Each input pair gets one output dict with all metric keys."""

        def metric_a(cands: list[str], _refs: list[str]) -> list[dict[str, float]]:
            return [{"a/f1": 0.1 * i} for i in range(len(cands))]

        def metric_b(cands: list[str], _refs: list[str]) -> list[dict[str, float]]:
            return [{"b/f1": 0.5} for _ in cands]

        scorers = {"a": metric_a, "b": metric_b}
        results = evaluate_batch(["g1", "g2", "g3"], ["r1", "r2", "r3"], scorers)
        assert len(results) == 3
        for i, scores in enumerate(results):
            assert scores["a/f1"] == pytest.approx(0.1 * i)
            assert scores["b/f1"] == pytest.approx(0.5)

    def test_a_failing_metric_stops_scoring(self) -> None:
        # Skipping it would leave the metric's column all-NaN in every analysis.
        def failing(_cands: list[str], _refs: list[str]) -> list[dict[str, float]]:
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            evaluate_batch(["a"], ["b"], {"fail": failing})

    def test_a_metric_returning_too_few_rows_stops_scoring(self) -> None:
        # Pairing what came back would leave the last admissions without the metric.
        def too_short(_cands: list[str], _refs: list[str]) -> list[dict[str, float]]:
            return [{"short/f1": 1.0}]

        with pytest.raises(ValueError, match="shorter"):
            evaluate_batch(["a", "b"], ["c", "d"], {"short": too_short})


class TestUmlsConcept:
    """The concept metric over a stubbed scispaCy pipeline, so no linker bundle loads."""

    def test_scores_are_keyed_by_the_space_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_scispacy(monkeypatch, {"gen": [("A", ["T047"])], "ref": [("A", ["T047"])]})

        scores = _evaluate_one("gen", "ref", _concept_scorers())

        assert scores == {
            "scispacy-clinical/f1": 1.0,
            "scispacy-clinical/precision": 1.0,
            "scispacy-clinical/recall": 1.0,
        }

    def test_plus_procedures_keeps_a_procedure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A is T060, a Procedures type outside the MEDCON default.
        install_fake_scispacy(
            monkeypatch,
            {"gen": [("A", ["T060"])], "ref": [("A", ["T060"]), ("B", ["T047"])]},
        )

        narrow = _evaluate_one("gen", "ref", _concept_scorers("scispacy"))
        wide = _evaluate_one("gen", "ref", _concept_scorers("scispacy-plus-procedures"))

        assert narrow["scispacy-clinical/recall"] == pytest.approx(0.0)
        assert wide["scispacy-plus-procedures/recall"] == pytest.approx(0.5)

    def test_concept_f1_filters_out_of_scope_tuis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Concepts outside the TUI filter are dropped before F1."""
        install_fake_scispacy(
            monkeypatch,
            {
                "gen": [("C0011849", ["T047"]), ("C9999999", ["T999"])],
                "ref": [("C0011849", ["T047"])],
            },
        )
        scorers = _concept_scorers()
        scores = _evaluate_one("gen", "ref", scorers)
        assert scores["scispacy-clinical/f1"] == pytest.approx(1.0)

    def test_concept_f1_partial_overlap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One shared concept of two on each side gives 0.5 on all three scores."""
        install_fake_scispacy(
            monkeypatch,
            {
                "gen": [("A", ["T047"]), ("B", ["T047"])],
                "ref": [("A", ["T047"]), ("C", ["T047"])],
            },
        )
        scorers = _concept_scorers()
        scores = _evaluate_one("gen", "ref", scorers)
        assert scores["scispacy-clinical/f1"] == pytest.approx(0.5)
        assert scores["scispacy-clinical/precision"] == pytest.approx(0.5)
        assert scores["scispacy-clinical/recall"] == pytest.approx(0.5)

    def test_concept_f1_one_side_empty_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One empty side -> F1 = 0 (not undefined)."""
        install_fake_scispacy(
            monkeypatch,
            {"gen": [], "ref": [("A", ["T047"])]},
        )
        scorers = _concept_scorers()
        scores = _evaluate_one("gen", "ref", scorers)
        assert scores["scispacy-clinical/f1"] == pytest.approx(0.0)

    def test_concept_f1_both_empty_is_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both empty -> F1 = 1 (no concepts to miss)."""
        install_fake_scispacy(monkeypatch, {"gen": [], "ref": []})
        scorers = _concept_scorers()
        scores = _evaluate_one("gen", "ref", scorers)
        assert scores["scispacy-clinical/f1"] == pytest.approx(1.0)

    def test_batch_dedupes_repeated_texts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A batch with repeated texts pipes each unique text exactly once."""
        fake_nlp = install_fake_scispacy(
            monkeypatch,
            {
                "gen1": [("A", ["T047"])],
                "gen2": [("B", ["T047"])],
                "ref_shared": [("A", ["T047"])],
            },
        )
        scorers = _concept_scorers()
        fn = scorers["scispacy-clinical"]
        # Same reference text used twice. Only 3 unique strings in this batch.
        scores_batch = fn(["gen1", "gen2"], ["ref_shared", "ref_shared"])
        assert len(scores_batch) == 2
        # Scores align with inputs: gen1=={A} ref=={A} → F1 1. gen2=={B} ref=={A} → F1 0.
        assert scores_batch[0]["scispacy-clinical/f1"] == pytest.approx(1.0)
        assert scores_batch[1]["scispacy-clinical/f1"] == pytest.approx(0.0)
        # The shared reference text is linked once.
        assert fake_nlp.piped_texts.count("ref_shared") == 1
