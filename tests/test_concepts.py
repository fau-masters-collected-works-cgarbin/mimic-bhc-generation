"""Tests for bhc_experiment.concepts: Concept, ConceptExtractor, and concept_text.

This module (and most of these tests) must import cleanly in the base test
environment, which has no ``umls_concept`` extra installed. A plain import is the
test: if ``concepts.py`` pulled in scispaCy, directly or transitively, importing it
here would raise. ``TestConceptTextRealCuis`` is the deliberate exception: it is
marked ``slow`` and asserts real CUIs, so it is the one place in this module that
needs scispaCy installed.
"""

import pytest

from .conftest import make_lab, make_note, make_prescription


def _sample_params() -> dict:
    """A fresh params dict with the same keys as ``params_from_extractor``'s return value.

    Every call returns new nested containers, so callers can mutate the
    result without affecting other tests.
    """
    return {
        "extractor": "scispacy",
        "extractor_version": 1,
        "extractor_params": {
            "spacy_model": "en_core_sci_md",
            "spacy_model_version": "0.5.4",
            "scispacy_version": "0.6.2",
            "spacy_version": "3.7.5",
            "linker_name": "umls",
            "threshold": 0.7,
            "resolve_abbreviations": True,
            "tuis": ["T017", "T047"],
            "linking_artifact_cuis": ["C0032207", "C0168634"],
        },
    }


class TestConceptExtractorProtocol:
    """UmlsExtractor satisfies ConceptExtractor."""

    def test_umls_extractor_satisfies_the_protocol(self) -> None:
        # UmlsExtractor's __init__ does the scispaCy import, and issubclass
        # never runs it.
        from bhc_experiment.concepts import ConceptExtractor
        from bhc_experiment.umls_extractor import UmlsExtractor

        assert issubclass(UmlsExtractor, ConceptExtractor)


class TestExtractorConfigSha256:
    """Sorted-key JSON hash of the extractor's fingerprinted parameters."""

    @pytest.mark.parametrize(
        ("field", "new_value"),
        [
            ("spacy_model", "en_core_sci_lg"),
            ("spacy_model_version", "0.5.3"),
            ("scispacy_version", "0.6.1"),
            ("spacy_version", "3.7.4"),
            ("linker_name", "mesh"),
            ("threshold", 0.9),
            ("resolve_abbreviations", False),
            ("tuis", ["T017"]),
            ("linking_artifact_cuis", ["C0032207"]),
        ],
    )
    def test_changing_any_extractor_param_field_changes_the_hash(
        self, field: str, new_value: object
    ) -> None:
        from bhc_experiment.concepts import extractor_config_sha256

        base = _sample_params()
        changed = _sample_params()
        changed["extractor_params"][field] = new_value

        assert extractor_config_sha256(base) != extractor_config_sha256(changed)

    @pytest.mark.parametrize(
        ("field", "new_value"),
        [("extractor", "quickumls"), ("extractor_version", 2)],
    )
    def test_changing_a_top_level_field_changes_the_hash(
        self, field: str, new_value: object
    ) -> None:
        from bhc_experiment.concepts import extractor_config_sha256

        base = _sample_params()
        changed = _sample_params()
        changed[field] = new_value

        assert extractor_config_sha256(base) != extractor_config_sha256(changed)

    def test_dict_key_order_does_not_affect_the_hash(self) -> None:
        # sort_keys=True canonicalizes dict key order, so the same content
        # built in a different insertion order must hash identically.
        from bhc_experiment.concepts import extractor_config_sha256

        in_order = _sample_params()
        reordered = {
            "extractor_version": in_order["extractor_version"],
            "extractor": in_order["extractor"],
            "extractor_params": {
                "tuis": in_order["extractor_params"]["tuis"],
                "spacy_model": in_order["extractor_params"]["spacy_model"],
                "spacy_model_version": in_order["extractor_params"]["spacy_model_version"],
                "scispacy_version": in_order["extractor_params"]["scispacy_version"],
                "spacy_version": in_order["extractor_params"]["spacy_version"],
                "linker_name": in_order["extractor_params"]["linker_name"],
                "threshold": in_order["extractor_params"]["threshold"],
                "resolve_abbreviations": in_order["extractor_params"]["resolve_abbreviations"],
                "linking_artifact_cuis": in_order["extractor_params"]["linking_artifact_cuis"],
            },
        }

        assert extractor_config_sha256(in_order) == extractor_config_sha256(reordered)


class TestConfigStem:
    """The artifact filename stem: a pure function of the hashed params."""

    def test_stem_is_extractor_name_and_hash_prefix(self) -> None:
        from bhc_experiment.concepts import config_stem, extractor_config_sha256

        params = _sample_params()
        expected = f"scispacy-{extractor_config_sha256(params)[:12]}"

        assert config_stem(params) == expected


class TestConceptText:
    """concept_text dispatches by record type to the raw MIMIC field, never a
    formatted string."""

    def test_note_event_returns_text(self) -> None:
        from bhc_experiment.concepts import concept_text

        note = make_note(1, "synthetic note text")
        assert concept_text(note) == note.TEXT

    def test_lab_event_returns_label(self) -> None:
        from bhc_experiment.concepts import concept_text

        lab = make_lab(LABEL="synthetic lab label")
        assert concept_text(lab) == lab.LABEL

    def test_prescription_returns_drug(self) -> None:
        from bhc_experiment.concepts import concept_text

        rx = make_prescription(DRUG="synthetic drug name")
        assert concept_text(rx) == rx.DRUG


class TestSpansByCui:
    def test_spans_are_sliced_at_the_match_offsets(self):
        from bhc_experiment.concepts import Concept, spans_by_cui

        # An off-by-one slice changes which spans a caller finds in the input, and
        # nothing in the result shows it. Repeated CUIs collect.
        text = "Acute kidney injury and hypotension. AKI again."
        matches = [
            Concept(cui="C0022660", term="AKI", semtypes=(), score=1.0, start=0, end=19),
            Concept(cui="C0020649", term="hypotension", semtypes=(), score=1.0, start=24, end=35),
            Concept(cui="C0022660", term="AKI", semtypes=(), score=1.0, start=37, end=40),
        ]
        assert spans_by_cui(matches, text) == {
            "C0022660": {"acute kidney injury", "aki"},
            "C0020649": {"hypotension"},
        }


@pytest.fixture(scope="module")
def real_extractor():
    """One real UmlsExtractor, shared by every test in TestConceptTextRealCuis.

    Construction loads scispaCy, en_core_sci_md, and the UMLS linker, which
    is slow.
    """
    from bhc_experiment.umls_extractor import UmlsExtractor

    return UmlsExtractor()


class TestConceptTextRealCuis:
    """concept_text against a real extractor. Raw field text links to the right
    concept. The dropped fields (FLAG, CATEGORY) never leak a concept into the
    output."""

    pytestmark = pytest.mark.slow

    def test_white_blood_cells_label_links_to_leukocytes_not_blood_cells(
        self, real_extractor
    ) -> None:
        from bhc_experiment.concepts import concept_text

        lab = make_lab(LABEL="White Blood Cells")
        text = concept_text(lab)
        cuis = real_extractor.extract_cuis([text])[text]

        assert "C0023516" in cuis  # Leukocytes: the raw LABEL's real concept.
        assert "C0005773" not in cuis  # Blood Cells: the generic look-alike.

    def test_flag_does_not_leak_into_the_extracted_text(self, real_extractor) -> None:
        from bhc_experiment.concepts import concept_text

        # LABEL contains no form of the word "abnormal". If concept_text
        # returned formatted text that embeds FLAG (as the prompt formatter
        # does), the extractor would find "Abnormal" in it.
        lab = make_lab(LABEL="Potassium", FLAG="abnormal")
        text = concept_text(lab)
        cuis = real_extractor.extract_cuis([text])[text]

        assert "C0205161" not in cuis  # Abnormal
        # Positive control: "abnormal" DOES link to C0205161 when it is in the
        # text, so the absence above proves concept_text dropped the FLAG, not
        # that the concept is unreachable. Both break together if T033 ever
        # leaves the TUI filter, keeping this test discriminating.
        assert "C0205161" in real_extractor.extract_cuis(["abnormal"])["abnormal"]

    def test_category_does_not_leak_into_the_extracted_text(self, real_extractor) -> None:
        from bhc_experiment.concepts import concept_text

        # TEXT mentions neither nursing nor breast feeding. If concept_text
        # returned formatted text that embeds CATEGORY (as the prompt
        # formatter does), the extractor would find "Nursing" in it, which
        # UMLS links to Breast Feeding instead of the nursing profession.
        note = make_note(
            12, "Patient reports mild headache overnight, vital signs stable.", CATEGORY="Nursing"
        )
        text = concept_text(note)
        cuis = real_extractor.extract_cuis([text])[text]

        assert "C0006147" not in cuis  # Breast Feeding, aliased "Nursing"
        # Positive control: "Nursing" DOES link to C0006147 when it is in the
        # text, so the absence above proves concept_text dropped the CATEGORY,
        # not that the concept is unreachable. Both break together if T040 ever
        # leaves the TUI filter, keeping this test discriminating.
        assert "C0006147" in real_extractor.extract_cuis(["Nursing"])["Nursing"]
