"""Load the note-redundancy review gate's records: notes paired with what dedup cut.

Reads the frozen prepared-input artifact through its store, so a reviewer
sees what generation and scoring read.
"""

from dataclasses import dataclass

from bhc_experiment.loader import NoteEvent
from bhc_experiment.prepared_store import PreparedStore


@dataclass(frozen=True)
class RemovedSentence:
    """One duplicate sentence removed by sentence dedup, tagged within-note or between-note."""

    text: str
    kind: str


@dataclass(frozen=True)
class NoteRemovals:
    """One surviving note paired with what sentence dedup removed from it."""

    note: NoteEvent
    removed: list[RemovedSentence]


def load_note_removals(prepared_store: PreparedStore, hadm_id: int) -> list[NoteRemovals]:
    """Every surviving note in one admission's bundle, paired with its removed sentences.

    A note with no removals gets an empty list.
    """
    bundle = prepared_store.bundle(hadm_id)
    removed_by_row = prepared_store.removed_sentences(hadm_id)
    return [
        NoteRemovals(
            note=note,
            removed=[RemovedSentence(**s) for s in removed_by_row.get(note.ROW_ID, [])],
        )
        for note in bundle.noteevents
    ]
