"""Load the concept-viewer review gate's prompt records, each paired with its concepts.

Reads the frozen prepared-input and concept artifacts through their stores,
so a reviewer sees what generation and scoring read.
"""

from dataclasses import dataclass

from bhc_experiment.concept_store import ConceptStore
from bhc_experiment.concepts import Concept, concept_text
from bhc_experiment.oracle import attribute_records
from bhc_experiment.prepared_store import PreparedStore


@dataclass(frozen=True)
class RecordConcepts:
    """One record the prompt would feed the model, alongside its extracted concepts."""

    formatted_text: str
    concepts: list[Concept]


@dataclass(frozen=True)
class AdmissionConceptReview:
    """One admission's per-record concepts plus the reference BHC's concepts."""

    records: list[RecordConcepts]
    reference_concepts: list[Concept]


def load_concept_review(
    prepared_store: PreparedStore,
    concept_store: ConceptStore,
    hadm_id: int,
) -> AdmissionConceptReview:
    """The concepts behind one admission's prompt: each surviving record's, and the reference BHC's.

    Reuses ``attribute_records`` for record selection, so the reviewed set matches
    what the prompt builder would feed the model.
    """
    bundle = prepared_store.bundle(hadm_id)
    attributed = attribute_records(bundle, concept_store)
    concept_texts = [concept_text(ar.record) for ar in attributed]
    concepts_by_text = concept_store.extract_concepts(concept_texts)
    records = [
        RecordConcepts(formatted_text=ar.formatted_text, concepts=concepts_by_text[text])
        for ar, text in zip(attributed, concept_texts, strict=True)
    ]
    reference = bundle.ground_truth_bhc
    reference_concepts = concept_store.extract_concepts([reference])[reference]
    return AdmissionConceptReview(records=records, reference_concepts=reference_concepts)
