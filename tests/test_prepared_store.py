"""Tests for bhc_experiment.prepared_store: the prepared-input artifact writer and reader.

Bundles use synthetic HADM_ID/SUBJECT_ID values outside real MIMIC-III
ranges and fake note/lab/prescription text (see AGENTS.md's MIMIC data
safety section).
"""

import pytest

from bhc_experiment.loader import AdmissionBundle
from bhc_experiment.prepared_store import (
    PreparedStore,
    prepared_commit,
    prepared_stem,
    write_prepared,
)

from .conftest import make_bundle, make_lab, make_note, make_prescription

COHORT = "synthetic_cohort.json"
COMMIT = "c" * 40


def _bundle(hadm_id: int, subject_id: int) -> AdmissionBundle:
    """Two notes (one with a null CHARTTIME and no DESCRIPTION), one lab, one prescription."""
    notes = [
        make_note(1, "synthetic note text"),
        make_note(
            2,
            "second synthetic note",
            CHARTDATE="2150-01-02",
            CHARTTIME=None,
            CATEGORY="Physician",
            DESCRIPTION=None,
        ),
    ]
    return make_bundle(
        notes,
        hadm_id=hadm_id,
        subject_id=subject_id,
        labs=[make_lab()],
        prescriptions=[make_prescription()],
    )


def _bundles() -> list[AdmissionBundle]:
    return [_bundle(99999, 10), _bundle(99998, 11)]


class TestWritePrepared:
    def test_writes_one_line_per_bundle_named_by_cohort_and_dedup(self, tmp_path) -> None:
        bundles = _bundles()

        write_prepared(
            tmp_path, COHORT, "sentence", bundles, {}, git_commit=COMMIT, git_dirty=False
        )

        assert prepared_stem(COHORT, "sentence") == "synthetic_cohort-dedup-sentence"
        jsonl_path = tmp_path / "synthetic_cohort-dedup-sentence.jsonl"
        assert len(jsonl_path.read_text().splitlines()) == len(bundles)


class TestPreparedCommit:
    def test_reads_back_the_commit_preparation_ran_at(self, tmp_path) -> None:
        # Generation copies it into the run record, and scoring refuses a run
        # whose inputs were prepared again since.
        write_prepared(tmp_path, COHORT, "none", _bundles(), {}, git_commit=COMMIT, git_dirty=False)
        assert prepared_commit(tmp_path, COHORT, "none") == COMMIT


class TestPreparedStore:
    """Read-only reconstruction of AdmissionBundles from a written artifact."""

    def test_round_trips_bundles_field_for_field(self, tmp_path) -> None:
        bundles = _bundles()
        write_prepared(
            tmp_path, COHORT, "sentence", bundles, {}, git_commit=COMMIT, git_dirty=False
        )

        store = PreparedStore(tmp_path, COHORT, "sentence")

        for bundle in bundles:
            assert store.bundle(bundle.admission.HADM_ID) == bundle

    def test_missing_artifact_names_the_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="synthetic_cohort-dedup-none"):
            PreparedStore(tmp_path, COHORT, "none")


class TestRemovedSentences:
    """write_prepared has what sentence dedup cut, alongside the bundle."""

    def test_removed_sentences_round_trip_and_bundle_is_unchanged(self, tmp_path) -> None:
        bundle = _bundle(99999, 10)
        removed_sentences = {
            99999: [
                {
                    "row_id": 1,
                    "removed": [{"text": "duplicate sentence excised here", "kind": "within"}],
                }
            ]
        }

        write_prepared(
            tmp_path,
            COHORT,
            "sentence",
            [bundle],
            removed_sentences,
            git_commit=COMMIT,
            git_dirty=False,
        )
        store = PreparedStore(tmp_path, COHORT, "sentence")

        assert store.bundle(99999) == bundle
        assert store.removed_sentences(99999) == {
            1: [{"text": "duplicate sentence excised here", "kind": "within"}]
        }
