"""Input-coverage ceiling for concept-recall evaluation.

The ceiling is the share of reference concepts present in the assembled model
input. It bounds achievable concept recall for almost every admission, but it
is not a hard limit: a model can still generate a concept the extractor did not
find in the input, by inference or by wording that links where the raw input
did not. Recall reads above the ceiling for some admissions.

Holds the set formulas over CUI sets for the ceiling, for the concept
precision, recall, and F1 that the MEDCON metric reports, and for the
partition of the generated concepts that splits precision into its parts.

Reusing these formulas elsewhere takes one concept extractor, run the same way
over three texts: the reference summary, the input it was written from with the
reference's own document excluded, and the generated summary.
"""

from dataclasses import dataclass
from typing import NamedTuple


class ConceptScores(NamedTuple):
    """Set precision, recall, and F1 over a generated and a reference CUI set."""

    precision: float
    recall: float
    f1: float


def concept_prf(gen_cuis: frozenset[str], ref_cuis: frozenset[str]) -> ConceptScores:
    """Score a generated CUI set against a reference CUI set.

    Empty-set convention: both empty scores perfect, since there were no concepts
    to miss. One empty scores zero. Disjoint non-empty sets score zero without
    dividing by zero in the F1 denominator.
    """
    if not gen_cuis and not ref_cuis:
        return ConceptScores(1.0, 1.0, 1.0)
    if not gen_cuis or not ref_cuis:
        return ConceptScores(0.0, 0.0, 0.0)
    n_shared = len(gen_cuis & ref_cuis)
    precision = n_shared / len(gen_cuis)
    recall = n_shared / len(ref_cuis)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return ConceptScores(precision, recall, f1)


@dataclass(frozen=True)
class InputCoverage:
    """The input-coverage ceiling for one admission, with its raw counts.

    ``ceiling`` is ``|input concepts & reference concepts| / |reference
    concepts|``. ``n_ref`` is the reference concept count (the recall
    denominator) and ``n_overlap`` is the count present in the input (the
    recoverable count). The counts are kept alongside the ratio so analysis
    can pool concepts across admissions (a concept-weighted view)
    without re-extracting, and so the ceiling can be audited from the record.
    """

    ceiling: float
    n_ref: int
    n_overlap: int


def input_coverage(input_cuis: frozenset[str], ref_cuis: frozenset[str]) -> InputCoverage | None:
    """Compute the input-coverage ceiling from input and reference CUI sets.

    Mirrors the evaluator's recall (intersection over reference size) with the
    input substituted for the generated text, so the ceiling is directly
    comparable to the scored concept recall. Returns ``None`` when the reference
    has no concepts, where the ceiling is undefined, so the admission is excluded
    from aggregates.
    """
    if not ref_cuis:
        return None
    n_overlap = len(input_cuis & ref_cuis)
    return InputCoverage(
        ceiling=n_overlap / len(ref_cuis),
        n_ref=len(ref_cuis),
        n_overlap=n_overlap,
    )


@dataclass(frozen=True)
class GeneratedPartition:
    """One admission's generated concepts split by input and reference membership.

    Field names read ``<input membership>_<reference membership>``. The four are
    disjoint and their union is the generated set. In the coverage framework's
    cell names, they are the recovered set (``in_input_in_ref``), the untraced
    match (``not_input_in_ref``), over-inclusion (``in_input_not_ref``), and
    unsupported (``not_input_not_ref``).

    ``in_input_in_ref`` and ``not_input_in_ref`` are precision's numerator.

    ``not_input_not_ref`` is exact-CUI, so it is an upper bound: the input can
    have a concept in wording the extractor did not link there.
    """

    in_input_in_ref: frozenset[str]
    in_input_not_ref: frozenset[str]
    not_input_in_ref: frozenset[str]
    not_input_not_ref: frozenset[str]


def partition_generated(
    gen_cuis: frozenset[str], ref_cuis: frozenset[str], input_cuis: frozenset[str]
) -> GeneratedPartition | None:
    """Partition a generated CUI set by input and reference membership.

    Returns ``None`` on an empty generation, where every share is undefined, so the
    admission drops out of aggregates. Same convention as ``input_coverage`` on an empty
    reference.
    """
    if not gen_cuis:
        return None
    in_ref = gen_cuis & ref_cuis
    not_ref = gen_cuis - ref_cuis
    return GeneratedPartition(
        in_input_in_ref=in_ref & input_cuis,
        in_input_not_ref=not_ref & input_cuis,
        not_input_in_ref=in_ref - input_cuis,
        not_input_not_ref=not_ref - input_cuis,
    )
