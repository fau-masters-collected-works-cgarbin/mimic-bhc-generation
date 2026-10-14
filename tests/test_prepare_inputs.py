"""Tests for scripts/pipeline/prepare_inputs.py: the prepared-input artifact writer.

All identifiers are synthetic (see AGENTS.md MIMIC data safety).
"""

import json
from pathlib import Path

import prepare_inputs
import pytest
from prepare_inputs import _filter_bundle, extract_concepts, prepared_bundles

from bhc_experiment.concepts import ExtractionResult, text_sha256
from bhc_experiment.loader import AdmissionBundle, NoteEvent
from bhc_experiment.prepared_store import prepared_commit, write_prepared

from .conftest import make_bundle, make_lab, make_note, make_prescription


def _bundle(notes: list[NoteEvent]) -> AdmissionBundle:
    """Caller-supplied notes, one lab, one prescription."""
    return make_bundle(notes, labs=[make_lab()], prescriptions=[make_prescription()])


class TestPreparedBundles:
    """prepared_bundles prepares the records for the main concept space only."""

    def test_another_concept_space_reads_the_records_already_prepared(self, tmp_path) -> None:
        # Rewriting the records would stamp a new commit on them, and scoring
        # then refuses every run generated from the earlier preparation.
        bundle = _bundle([make_note(1, "Synthetic nursing note.")])
        write_prepared(
            tmp_path,
            "test_cohort.json",
            "sentence",
            [bundle],
            {},
            git_commit="a" * 40,
            git_dirty=False,
        )
        bundles = prepared_bundles(
            "test_cohort.json", "sentence", "quickumls", prepared_dir=tmp_path
        )
        assert bundles == [bundle]
        assert prepared_commit(tmp_path, "test_cohort.json", "sentence") == "a" * 40


class TestFilterBundle:
    """_filter_bundle applies the note filter, leaving other fields alone."""

    def test_drops_the_discharge_summary_and_keeps_labs_and_prescriptions(self) -> None:
        bundle = _bundle(
            [
                make_note(1, "admission note"),
                make_note(2, "discharge text", is_discharge_summary=True),
            ]
        )

        result = _filter_bundle(bundle)

        assert [n.ROW_ID for n in result.noteevents] == [1]
        assert result.labevents == bundle.labevents
        assert result.prescriptions == bundle.prescriptions


class _FakeExtractor:
    """Test double for the concept extractor: no scispaCy, counts its own calls."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def extract_entries(self, texts: list[str]) -> dict[str, ExtractionResult]:
        self.calls.append(list(texts))
        return {
            t: ExtractionResult(matches=[], rejected=[], n_entities=0, n_unlinked=0) for t in texts
        }


class TestExtractConcepts:
    """extract_concepts wires collect_texts + build_artifact over the PREPARED bundles.

    ``concept_extractor`` and ``concept_params`` are monkeypatched to a stub
    extractor so no test here loads scispaCy.
    """

    def _stub_extractor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: "_FakeExtractor"
    ) -> None:
        monkeypatch.setattr(prepare_inputs, "concept_extractor", lambda _space, **_kwargs: fake)
        monkeypatch.setattr(
            prepare_inputs,
            "concept_params",
            lambda _space: {
                "extractor": "fake",
                "extractor_version": 1,
                "extractor_params": {},
            },
        )
        monkeypatch.setattr(prepare_inputs, "CONCEPTS_DIR", tmp_path)

    def _stored_hashes(self, tmp_path: Path) -> set[str]:
        jsonl_path = next(tmp_path.glob("*.jsonl"))
        return {json.loads(line)["text_sha256"] for line in jsonl_path.read_text().splitlines()}

    def test_writes_every_prepared_texts_hash_into_the_artifact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._stub_extractor(monkeypatch, tmp_path, _FakeExtractor())
        bundle = _bundle([make_note(1, "note one"), make_note(2, "note two")])

        extract_concepts([bundle], "scispacy")

        stored = self._stored_hashes(tmp_path)
        expected = {
            text_sha256("note one"),
            text_sha256("note two"),
            text_sha256(bundle.ground_truth_bhc),
            text_sha256("Potassium"),  # the bundle's one LabEvent, via concept_text
            text_sha256("Aspirin"),  # the bundle's one Prescription, via concept_text
        }
        assert expected <= stored

    def test_second_run_over_the_same_bundles_appends_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake = _FakeExtractor()
        self._stub_extractor(monkeypatch, tmp_path, fake)
        bundle = _bundle([make_note(1, "note one")])
        concept_space = "scispacy"

        extract_concepts([bundle], concept_space)
        calls_after_first_run = len(fake.calls)

        extract_concepts([bundle], concept_space)

        assert len(fake.calls) == calls_after_first_run

    def test_sentence_deduplicated_text_lands_in_the_store(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Simulates the dedup rewrite: the prepared bundle has the
        # post-dedup TEXT, distinct from the pre-dedup (raw) text.
        raw_text = "Patient stable. Patient stable. No new complaints."
        deduplicated_text = "Patient stable. No new complaints."
        bundle = _bundle([make_note(1, deduplicated_text)])

        self._stub_extractor(monkeypatch, tmp_path, _FakeExtractor())
        extract_concepts([bundle], "scispacy")

        stored = self._stored_hashes(tmp_path)
        assert text_sha256(deduplicated_text) in stored
        assert text_sha256(raw_text) not in stored

    def test_names_the_artifact_by_the_named_space_params(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        built: list[str] = []
        seen: list[str] = []

        def _build(space: str, **_kwargs: object) -> _FakeExtractor:
            built.append(space)
            return _FakeExtractor()

        def _params(space: str) -> dict:
            seen.append(space)
            return {"extractor": "quickumls", "extractor_version": 1, "extractor_params": {}}

        monkeypatch.setattr(prepare_inputs, "concept_extractor", _build)
        monkeypatch.setattr(prepare_inputs, "concept_params", _params)
        monkeypatch.setattr(prepare_inputs, "CONCEPTS_DIR", tmp_path / "concepts")
        (tmp_path / "concepts").mkdir()

        params = extract_concepts([_bundle([make_note(1, "note one")])], "quickumls")

        assert built == ["quickumls"]
        assert seen == ["quickumls"]
        assert params["extractor"] == "quickumls"
