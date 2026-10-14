"""Tests for bhc_experiment.concept_store: ConceptStore, the artifact reader.

ConceptStore stands in for a live extractor once the concept artifact exists:
it satisfies ConceptExtractor by reading only ``text_sha256`` and ``matches``
from the on-disk file, never running scispaCy. A miss is an error, never an
empty list or a fallback to another fingerprint.
"""

import json

import pytest

from bhc_experiment.concept_store import ConceptStore
from bhc_experiment.concepts import Concept, config_stem, text_sha256


def _params() -> dict:
    """An arbitrary fingerprinted params dict, used only here to derive a stem."""
    return {
        "extractor": "scispacy",
        "extractor_version": 1,
        "extractor_params": {"threshold": 0.7},
    }


def _match(cui: str, term: str, semtypes: list[str], score: float, start: int, end: int) -> dict:
    """One ``matches[]`` entry, with exactly the on-disk schema's fields."""
    return {
        "cui": cui,
        "term": term,
        "semtypes": semtypes,
        "score": score,
        "start": start,
        "end": end,
    }


def _artifact_path(tmp_path):
    return tmp_path / f"{config_stem(_params())}.jsonl"


def _write_artifact(path, entries: list[dict]) -> None:
    """Write synthetic entries as the artifact's one-JSON-object-per-line file."""
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


class TestConceptStore:
    """A ConceptStore built over a synthetic artifact file in tmp_path."""

    def test_extract_cuis_is_the_cui_projection_of_extract_concepts(self, tmp_path) -> None:
        # Two mentions of the same CUI plus one distinct CUI: extract_cuis must
        # collapse the repeat and drop mention order.
        text = "synthetic note with repeated concept mentions"
        entry = {
            "text_sha256": text_sha256(text),
            "matches": [
                _match("C0000001", "Fever", ["T184"], 0.99, 0, 5),
                _match("C0000002", "Cough", ["T184"], 0.95, 10, 15),
                _match("C0000001", "Fever", ["T184"], 0.90, 20, 25),
            ],
        }
        _write_artifact(_artifact_path(tmp_path), [entry])
        store = ConceptStore(tmp_path, _params())

        concepts = store.extract_concepts([text])[text]
        expected_cuis = frozenset(c.cui for c in concepts)

        assert expected_cuis == frozenset({"C0000001", "C0000002"})
        assert store.extract_cuis([text]) == {text: expected_cuis}

    def test_text_absent_from_artifact_raises_not_empty_list(self, tmp_path) -> None:
        known_text = "a text present in the artifact"
        entry = {
            "text_sha256": text_sha256(known_text),
            "matches": [_match("C0000001", "Fever", ["T184"], 0.99, 0, 5)],
        }
        _write_artifact(_artifact_path(tmp_path), [entry])
        store = ConceptStore(tmp_path, _params())

        with pytest.raises(KeyError):
            store.extract_concepts(["a synthetic text never extracted into this artifact"])

    def test_miss_error_carries_only_hash_and_length_never_the_text(self, tmp_path) -> None:
        _write_artifact(_artifact_path(tmp_path), [])
        store = ConceptStore(tmp_path, _params())

        # MIMIC safety: this string stands in for real patient text. It must
        # never appear in the exception message, only its hash and length.
        missing_text = "sensitive-patient-identifiable-string-must-not-leak"
        expected_hash = text_sha256(missing_text)

        with pytest.raises(KeyError) as exc_info:
            store.extract_concepts([missing_text])

        message = str(exc_info.value)
        assert missing_text not in message
        assert expected_hash in message
        assert str(len(missing_text)) in message

    def test_extract_concepts_matches_a_fake_extractor_mention_for_mention(self, tmp_path) -> None:
        text = "synthetic note used to compare store output against a fake extractor"
        expected = [
            Concept(cui="C0000001", term="Fever", semtypes=("T184",), score=0.99, start=0, end=5),
            Concept(
                cui="C0000002",
                term="Cough",
                semtypes=("T033", "T184"),
                score=0.87,
                start=20,
                end=25,
            ),
        ]

        class _FakeExtractor:
            """Minimal ConceptExtractor test double returning fixed Concepts."""

            def extract_concepts(self, texts: list[str]) -> dict[str, list[Concept]]:
                return dict.fromkeys(texts, expected)

            def extract_cuis(self, texts: list[str]) -> dict[str, frozenset[str]]:
                return {t: frozenset(c.cui for c in expected) for t in texts}

        fake = _FakeExtractor()

        entry = {
            "text_sha256": text_sha256(text),
            "matches": [
                _match(c.cui, c.term, list(c.semtypes), c.score, c.start, c.end) for c in expected
            ],
        }
        _write_artifact(_artifact_path(tmp_path), [entry])
        store = ConceptStore(tmp_path, _params())

        assert store.extract_concepts([text]) == fake.extract_concepts([text])

    def test_extract_cuis_ignores_rejected_entries(self, tmp_path) -> None:
        text = "synthetic note with one accepted and two rejected concepts"
        entry = {
            "text_sha256": text_sha256(text),
            "matches": [_match("C0000001", "Fever", ["T184"], 0.99, 0, 5)],
            "rejected": [
                {"cui": "C0032207", "score": 1.0, "reason": "blocklisted", "start": 10, "end": 12},
                {
                    "cui": "C0030705",
                    "score": 1.0,
                    "reason": "out_of_filter",
                    "start": 20,
                    "end": 27,
                },
            ],
        }
        _write_artifact(_artifact_path(tmp_path), [entry])
        store = ConceptStore(tmp_path, _params())

        cuis = store.extract_cuis([text])[text]

        assert cuis == frozenset({"C0000001"})

    def test_matches_preserve_mention_order(self, tmp_path) -> None:
        text = "synthetic note with three concepts in a fixed, non-alphabetical order"
        entry = {
            "text_sha256": text_sha256(text),
            "matches": [
                _match("C0000003", "Third", ["T184"], 0.5, 30, 35),
                _match("C0000001", "First", ["T184"], 0.9, 0, 5),
                _match("C0000002", "Second", ["T184"], 0.8, 10, 15),
            ],
        }
        _write_artifact(_artifact_path(tmp_path), [entry])
        store = ConceptStore(tmp_path, _params())

        cuis_in_order = [c.cui for c in store.extract_concepts([text])[text]]

        assert cuis_in_order == ["C0000003", "C0000001", "C0000002"]
