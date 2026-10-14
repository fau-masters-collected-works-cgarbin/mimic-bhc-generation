"""Tests for bhc_experiment.note_review: the note-redundancy review gate's loader."""

from bhc_experiment.loader import AdmissionBundle
from bhc_experiment.note_review import load_note_removals

from .conftest import make_bundle, make_note

HADM_ID = 99999

NOTE_WITH_REMOVALS = make_note(1, "fake deduped note text")
NOTE_CLEAN = make_note(2, "fake note with nothing removed", CATEGORY="Physician")

REMOVED_FOR_NOTE_1 = [
    {"text": "fake within-note duplicate sentence", "kind": "within"},
    {"text": "fake between-note duplicate sentence", "kind": "between"},
]


class _FakePreparedStore:
    """Duck-types PreparedStore: a canned bundle and removed-sentences map for one admission."""

    def bundle(self, hadm_id: int) -> AdmissionBundle:
        assert hadm_id == HADM_ID
        return make_bundle([NOTE_WITH_REMOVALS, NOTE_CLEAN], hadm_id=HADM_ID)

    def removed_sentences(self, hadm_id: int) -> dict[int, list[dict]]:
        assert hadm_id == HADM_ID
        return {NOTE_WITH_REMOVALS.ROW_ID: REMOVED_FOR_NOTE_1}


def test_loader_pairs_each_surviving_note_with_its_removed_sentences():
    removals = load_note_removals(_FakePreparedStore(), HADM_ID)

    by_row_id = {entry.note.ROW_ID: entry.removed for entry in removals}
    assert [(s.text, s.kind) for s in by_row_id[NOTE_WITH_REMOVALS.ROW_ID]] == [
        ("fake within-note duplicate sentence", "within"),
        ("fake between-note duplicate sentence", "between"),
    ]
    assert by_row_id[NOTE_CLEAN.ROW_ID] == []
