"""Tests for bhc_experiment.coverage."""

from math import isclose

from bhc_experiment.coverage import (
    ConceptScores,
    InputCoverage,
    concept_prf,
    input_coverage,
    partition_generated,
)


class TestConceptPrf:
    """Set precision, recall, F1 over CUI sets, with the empty-set convention."""

    def test_partial_overlap(self):
        # gen {A,B,C} vs ref {B,C,D}: two hits, three on each side.
        result = concept_prf(frozenset({"A", "B", "C"}), frozenset({"B", "C", "D"}))
        assert isclose(result.precision, 2 / 3)
        assert isclose(result.recall, 2 / 3)
        assert isclose(result.f1, 2 / 3)

    def test_both_empty_scores_perfect(self):
        # No concepts to miss, so a generation that names none is not penalized.
        assert concept_prf(frozenset(), frozenset()) == ConceptScores(1.0, 1.0, 1.0)

    def test_one_side_empty_scores_zero(self):
        assert concept_prf(frozenset({"A"}), frozenset()) == ConceptScores(0.0, 0.0, 0.0)
        assert concept_prf(frozenset(), frozenset({"A"})) == ConceptScores(0.0, 0.0, 0.0)

    def test_disjoint_sets_score_zero_rather_than_dividing_by_zero(self):
        # Both sets are non-empty, so the empty-set branch does not fire, and
        # precision + recall is 0 in the F1 denominator.
        assert concept_prf(frozenset({"A"}), frozenset({"B"})) == ConceptScores(0.0, 0.0, 0.0)


class TestInputCoverage:
    def test_fraction_of_reference_present_in_input(self):
        # B and C of the three reference concepts are in the input, D is not.
        result = input_coverage(frozenset({"A", "B", "C"}), frozenset({"B", "C", "D"}))
        assert isclose(result.ceiling, 2 / 3)
        assert result.n_ref == 3
        assert result.n_overlap == 2

    def test_reference_fully_covered_is_one(self):
        result = input_coverage(frozenset({"A", "B", "C", "D"}), frozenset({"B", "C"}))
        assert result == InputCoverage(ceiling=1.0, n_ref=2, n_overlap=2)

    def test_no_overlap_is_zero(self):
        result = input_coverage(frozenset({"A"}), frozenset({"B"}))
        assert result.ceiling == 0.0
        assert result.n_overlap == 0

    def test_empty_input_is_zero(self):
        result = input_coverage(frozenset(), frozenset({"B"}))
        assert result.ceiling == 0.0
        assert result.n_overlap == 0

    def test_empty_reference_is_none(self):
        # The ceiling is undefined with no reference concepts, so the admission
        # drops out of aggregates.
        assert input_coverage(frozenset({"A"}), frozenset()) is None


class TestPartitionGenerated:
    """The generated set split by input and reference membership.

    A concept assigned the wrong pair of memberships reads as a normal result, so
    each assignment is pinned with golden sets.
    """

    # A is in both, B is in the reference only, E is in the input only, X is in
    # neither. D is a reference concept the generation missed and F is input
    # content nobody used, so neither appears in the partition.
    GEN = frozenset({"A", "B", "E", "X"})
    REF = frozenset({"A", "B", "C", "D"})
    INPUT = frozenset({"A", "E", "F"})

    def test_each_concept_lands_by_its_memberships(self):
        result = partition_generated(self.GEN, self.REF, self.INPUT)
        assert result.in_input_in_ref == frozenset({"A"})
        assert result.not_input_in_ref == frozenset({"B"})
        assert result.in_input_not_ref == frozenset({"E"})
        assert result.not_input_not_ref == frozenset({"X"})

    def test_matched_sets_reconcile_with_concept_prf_precision(self):
        # Checks that partition_generated's matched counts agree with
        # concept_prf's precision.
        result = partition_generated(self.GEN, self.REF, self.INPUT)
        n_matched = len(result.in_input_in_ref) + len(result.not_input_in_ref)
        assert isclose(n_matched / len(self.GEN), concept_prf(self.GEN, self.REF).precision)

    def test_empty_generation_is_none(self):
        # Undefined, so the admission drops out of aggregates.
        assert partition_generated(frozenset(), self.REF, self.INPUT) is None
