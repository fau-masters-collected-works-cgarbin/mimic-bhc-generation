"""Tests for the bundle-level note deduplication transform."""

from bhc_experiment.loader import NoteEvent
from bhc_experiment.note_dedup import (
    deduplicate_bundle,
    deduplicate_bundle_with_removed_sentences,
    make_dedup_loader,
)

from .conftest import make_bundle, make_note

# A 7-word-plus sentence, >= 100 chars, used as the duplicated unit.
SHARED = (
    "Patient admitted overnight with chest pain and shortness of breath "
    "started on heparin drip and serial troponins were ordered"
)
SHARED_B = (
    "Repeat blood cultures were drawn and broad spectrum antibiotics were "
    "continued while awaiting sensitivities from the micro lab today"
)


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Fake segmenter: split on '. ', returning (sentence, start, end)."""
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for piece in text.split(". "):
        stripped = piece.strip()
        if not stripped:
            cursor += len(piece) + 2
            continue
        start = text.index(stripped, cursor)
        out.append((stripped, start, start + len(stripped)))
        cursor = start + len(stripped)
    return out


def _note(
    row_id: int, text: str, *, charttime: str | None = None, **overrides: object
) -> NoteEvent:
    """A note on 2020-01-02. CHARTTIME defaults to null, the order-by-ROW_ID case."""
    return make_note(row_id, text, CHARTDATE="2020-01-02", CHARTTIME=charttime, **overrides)


def test_none_returns_bundle_unchanged():
    bundle = make_bundle(
        [
            _note(1, SHARED, charttime="2020-01-02 02:00:00"),
            _note(2, SHARED, charttime="2020-01-02 05:00:00"),
        ]
    )
    assert deduplicate_bundle(bundle, "none") is bundle


def test_whole_note_keeps_most_recent_drops_older():
    notes = [
        _note(1, SHARED, charttime="2020-01-02 02:00:00"),
        _note(2, SHARED, charttime="2020-01-02 05:00:00"),
    ]
    out = deduplicate_bundle(make_bundle(notes), "whole-note")
    assert [n.ROW_ID for n in out.noteevents] == [2]


def test_whole_note_drops_superseded_pfi():
    final = _note(
        1,
        "FINAL REPORT chest radiograph shows clear lungs no acute process heart size normal",
        CATEGORY="Radiology",
        DESCRIPTION="CHEST PA AND LAT",
        charttime="2020-01-02 05:00:00",
    )
    pfi = _note(
        2,
        "PFI REPORT wet read prelim chest grossly clear pending final read",
        CATEGORY="Radiology",
        DESCRIPTION="CHEST PA AND LAT",
        charttime="2020-01-02 02:00:00",
    )
    out = deduplicate_bundle(make_bundle([pfi, final]), "whole-note")
    assert [n.ROW_ID for n in out.noteevents] == [1]


def test_whole_note_discharge_summary_passes_through():
    notes = [
        _note(1, SHARED, charttime="2020-01-02 02:00:00"),
        _note(2, SHARED, charttime="2020-01-02 05:00:00"),
        _note(3, "discharge summary brief hospital course text", is_discharge_summary=True),
    ]
    out = deduplicate_bundle(make_bundle(notes), "whole-note")
    # Older duplicate (1) dropped, recent (2) kept, discharge summary (3) untouched.
    assert [n.ROW_ID for n in out.noteevents] == [2, 3]


def test_sentence_excises_between_note_duplicate_keeps_earliest():
    notes = [
        _note(1, SHARED + ". Unique tail of the first note here.", charttime="2020-01-02 02:00:00"),
        _note(
            2,
            SHARED + ". Other unique sentence in the second note.",
            charttime="2020-01-02 05:00:00",
        ),
    ]
    out = deduplicate_bundle(make_bundle(notes), "sentence", _split_sentences)
    by_id = {n.ROW_ID: n.TEXT for n in out.noteevents}
    assert SHARED in by_id[1]
    assert SHARED not in by_id[2]
    assert "Unique tail of the first note here" in by_id[1]
    assert "Other unique sentence in the second note" in by_id[2]


def test_sentence_excises_within_note_duplicate():
    notes = [_note(1, SHARED + ". " + SHARED + ". Tail sentence stands alone here.")]
    out = deduplicate_bundle(make_bundle(notes), "sentence", _split_sentences)
    text = out.noteevents[0].TEXT
    assert text.count(SHARED) == 1
    assert "Tail sentence stands alone here" in text


def test_sentence_multiple_excisions_run_last_span_first():
    notes = [
        _note(
            1,
            SHARED + ". " + SHARED_B + ". earliest note tail here.",
            charttime="2020-01-02 02:00:00",
        ),
        _note(
            2,
            SHARED + ". " + SHARED_B + ". later note keeps this tail.",
            charttime="2020-01-02 05:00:00",
        ),
    ]
    out = deduplicate_bundle(make_bundle(notes), "sentence", _split_sentences)
    by_id = {n.ROW_ID: n.TEXT for n in out.noteevents}
    assert SHARED not in by_id[2]
    assert SHARED_B not in by_id[2]
    assert "later note keeps this tail" in by_id[2]
    assert SHARED in by_id[1]
    assert SHARED_B in by_id[1]


def test_sentence_removed_sentences_report_within_and_between():
    notes = [
        _note(
            1,
            SHARED + ". " + SHARED + ". " + SHARED_B + ". Unique tail of the first note here.",
            charttime="2020-01-02 02:00:00",
        ),
        _note(
            2,
            SHARED_B + ". Other unique sentence in the second note.",
            charttime="2020-01-02 05:00:00",
        ),
    ]
    _, removed_sentences = deduplicate_bundle_with_removed_sentences(
        make_bundle(notes), "sentence", _split_sentences
    )
    by_row_id = {entry["row_id"]: entry["removed"] for entry in removed_sentences}
    assert by_row_id[1] == [{"text": SHARED, "kind": "within"}]
    assert by_row_id[2] == [{"text": SHARED_B, "kind": "between"}]


def test_idempotent_whole_note():
    notes = [
        _note(1, SHARED, charttime="2020-01-02 02:00:00"),
        _note(2, SHARED, charttime="2020-01-02 05:00:00"),
    ]
    once = deduplicate_bundle(make_bundle(notes), "whole-note")
    twice = deduplicate_bundle(once, "whole-note")
    assert [n.ROW_ID for n in twice.noteevents] == [n.ROW_ID for n in once.noteevents]


def test_idempotent_sentence():
    notes = [
        _note(1, SHARED + ". Unique tail of the first note here.", charttime="2020-01-02 02:00:00"),
        _note(
            2,
            SHARED + ". Other unique sentence in the second note.",
            charttime="2020-01-02 05:00:00",
        ),
    ]
    once = deduplicate_bundle(make_bundle(notes), "sentence", _split_sentences)
    twice = deduplicate_bundle(once, "sentence", _split_sentences)
    assert [n.TEXT for n in twice.noteevents] == [n.TEXT for n in once.noteevents]


class TestMakeDedupLoader:
    """make_dedup_loader wraps a bundle loader with the dedup transform."""

    def test_whole_note_loader_drops_duplicate(self):
        bundle = make_bundle(
            [
                _note(1, SHARED, charttime="2020-01-02 02:00:00"),
                _note(2, SHARED, charttime="2020-01-02 05:00:00"),
            ]
        )

        def base(_conn, _hadm_id):
            return bundle

        loader = make_dedup_loader(base, "whole-note")
        out = loader(None, 1)
        assert [n.ROW_ID for n in out.noteevents] == [2]
