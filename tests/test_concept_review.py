"""Tests for bhc_experiment.concept_review: the concept-viewer gate's loader."""

from bhc_experiment.concept_review import load_concept_review
from bhc_experiment.concepts import Concept, concept_text
from bhc_experiment.loader import AdmissionBundle
from bhc_experiment.prompt_builder import format_note

from .conftest import make_bundle, make_lab, make_note, make_prescription

HADM_ID = 99999

NOTE = make_note(1, "fake note about a fake infection")
LAB = make_lab(LABEL="Fakelyte", FLAG="abnormal")
RX = make_prescription(DRUG="Fakedrugoxacin")
REFERENCE_BHC = "Patient treated with fakedrugoxacin for a fake infection."


NOTE_CONCEPT = Concept(
    cui="C9999001", term="fake infection", semtypes=("T047",), score=0.99, start=18, end=32
)
LAB_CONCEPT = Concept(
    cui="C9999002", term="Fakelyte", semtypes=("T059",), score=0.95, start=0, end=8
)
RX_CONCEPT = Concept(
    cui="C9999003", term="Fakedrugoxacin", semtypes=("T121",), score=0.98, start=0, end=14
)
REFERENCE_CONCEPT = Concept(
    cui="C9999003", term="fakedrugoxacin", semtypes=("T121",), score=0.97, start=22, end=36
)


class _FakeStore:
    """Duck-types ConceptStore: canned concepts and their CUI projection, keyed by text."""

    def __init__(self, concepts_by_text: dict[str, list[Concept]]):
        self._concepts_by_text = concepts_by_text

    def extract_concepts(self, texts: list[str]) -> dict[str, list[Concept]]:
        return {text: self._concepts_by_text.get(text, []) for text in texts}

    def extract_cuis(self, texts: list[str]) -> dict[str, frozenset[str]]:
        by_concepts = self.extract_concepts(texts)
        return {text: frozenset(c.cui for c in cs) for text, cs in by_concepts.items()}


class _FakePreparedStore:
    """Duck-types PreparedStore: one canned bundle regardless of the hadm_id passed in."""

    def __init__(self, bundle: AdmissionBundle):
        self._bundle = bundle

    def bundle(self, hadm_id: int) -> AdmissionBundle:
        assert hadm_id == self._bundle.admission.HADM_ID
        return self._bundle


def test_loader_surfaces_record_concepts_and_reference_concepts():
    concepts_by_text = {
        concept_text(NOTE): [NOTE_CONCEPT],
        concept_text(LAB): [LAB_CONCEPT],
        concept_text(RX): [RX_CONCEPT],
        REFERENCE_BHC: [REFERENCE_CONCEPT],
    }
    review = load_concept_review(
        _FakePreparedStore(
            make_bundle(
                [NOTE], hadm_id=HADM_ID, labs=[LAB], prescriptions=[RX], reference=REFERENCE_BHC
            )
        ),
        _FakeStore(concepts_by_text),
        HADM_ID,
    )

    assert review.reference_concepts == [REFERENCE_CONCEPT]
    by_formatted_text = {record.formatted_text: record.concepts for record in review.records}
    assert by_formatted_text[format_note(NOTE)] == [NOTE_CONCEPT]
