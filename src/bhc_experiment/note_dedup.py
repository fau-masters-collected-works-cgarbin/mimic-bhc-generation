"""Apply note deduplication to an AdmissionBundle before generation.

The detectors live in mimic_data_prep. This module bridges the bundle's
``NoteEvent`` records to them and returns a deduplicated bundle.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from bhc_experiment.prompt_builder import filter_notes
from mimic_data_prep.note_redundancy import detect_note_redundancy
from mimic_data_prep.sentence_redundancy import (
    classify_admission_sentences,
    load_sentence_segmenter,
    make_segment_fn,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import duckdb

    from bhc_experiment.loader import AdmissionBundle, NoteEvent


def deduplicate_bundle(
    bundle: AdmissionBundle,
    method: str,
    segment_fn: Callable[[str], list[tuple[str, int, int]]] | None = None,
) -> AdmissionBundle:
    """Return a bundle with the method's redundant note content removed.

    ``method`` is ``none`` (return the bundle unchanged), ``whole-note`` (drop
    near-duplicate and superseded-PFI notes), or ``sentence`` (excise verbatim
    duplicate sentences). Only the notes ``filter_notes`` keeps take part. The
    others pass through untouched. ``segment_fn`` is the sentence segmenter,
    required only for ``sentence``.
    """
    return deduplicate_bundle_with_removed_sentences(bundle, method, segment_fn)[0]


def deduplicate_bundle_with_removed_sentences(
    bundle: AdmissionBundle,
    method: str,
    segment_fn: Callable[[str], list[tuple[str, int, int]]] | None = None,
) -> tuple[AdmissionBundle, list[dict]]:
    """Same transform as ``deduplicate_bundle``, plus what ``sentence`` removed.

    The second return value records, per note that lost text, the duplicate
    sentences it cut and why, so a reviewer can inspect them. Only ``sentence``
    cuts sentences, so ``none`` and ``whole-note`` return an empty list.
    """
    if method == "none":
        return bundle, []
    eligible_notes = filter_notes(bundle.noteevents)
    if method == "whole-note":
        return _apply_whole_note(bundle, eligible_notes), []
    if method == "sentence":
        if segment_fn is None:
            msg = "sentence deduplication requires a segment_fn"
            raise ValueError(msg)
        return _apply_sentence(bundle, eligible_notes, segment_fn)
    msg = f"unknown deduplication method: {method!r}"
    raise ValueError(msg)


def make_dedup_loader(
    base_loader: Callable[[duckdb.DuckDBPyConnection, int], AdmissionBundle],
    deduplication: str,
) -> Callable[[duckdb.DuckDBPyConnection, int], AdmissionBundle]:
    """Wrap a bundle loader so it returns the deduplicated bundle.

    Returns ``base_loader`` unchanged for ``none``.
    """
    if deduplication == "none":
        return base_loader

    segment_fn = make_segment_fn(load_sentence_segmenter()) if deduplication == "sentence" else None

    def load(conn: duckdb.DuckDBPyConnection, hadm_id: int) -> AdmissionBundle:
        return deduplicate_bundle(base_loader(conn, hadm_id), deduplication, segment_fn)

    return load


def _to_detector_dict(note: NoteEvent) -> dict:
    """One note as the dict the redundancy detectors take."""
    return {
        "row_id": note.ROW_ID,
        "chartdate": note.CHARTDATE,
        "charttime": note.CHARTTIME,
        "category": note.CATEGORY,
        "description": note.DESCRIPTION,
        "text": note.TEXT,
    }


def _apply_whole_note(bundle: AdmissionBundle, eligible_notes: list[NoteEvent]) -> AdmissionBundle:
    """Drop the whole notes flagged as near-duplicate or superseded PFI.

    Ineligible notes stay in place and every surviving note is unchanged.
    """
    detail = detect_note_redundancy([_to_detector_dict(n) for n in eligible_notes])
    discarded = {eligible_notes[i].ROW_ID for i in detail["discarded_indices"]}
    if not discarded:
        return bundle
    kept = [n for n in bundle.noteevents if n.ROW_ID not in discarded]
    return replace(bundle, noteevents=kept)


def _apply_sentence(
    bundle: AdmissionBundle,
    eligible_notes: list[NoteEvent],
    segment_fn: Callable[[str], list[tuple[str, int, int]]],
) -> tuple[AdmissionBundle, list[dict]]:
    """Excise the duplicate sentence copies from each eligible note, notes kept.

    Only ``within`` and ``between`` spans are cut: those are the redundant
    copies, while the surviving copy (``kept``) and non-duplicates (``unique``)
    stay. Every note is kept, with only its TEXT rewritten. It also returns
    the cut sentences, read from each note before its TEXT is rewritten.

    A sentence's span can include the line breaks and spaces next to it, so a
    note whose every sentence is a duplicate can come out empty or
    whitespace-only. We keep even those notes so the admission timeline still
    records that a note occurred at that point.
    """
    rows = classify_admission_sentences([_to_detector_dict(n) for n in eligible_notes], segment_fn)
    trimmed: dict[int, str] = {}
    removed_sentences: list[dict] = []
    for note, row in zip(eligible_notes, rows, strict=True):
        removed = [s for s in row["sentences"] if s["kind"] in ("within", "between")]
        if not removed:
            continue
        trimmed[note.ROW_ID] = _excise(note.TEXT, [s["span"] for s in removed])
        removed_sentences.append(
            {
                "row_id": note.ROW_ID,
                "removed": [
                    {"text": note.TEXT[s["span"][0] : s["span"][1]], "kind": s["kind"]}
                    for s in removed
                ],
            }
        )
    if not trimmed:
        return bundle, removed_sentences
    new_notes = []
    for n in bundle.noteevents:
        if n.ROW_ID in trimmed:
            new_notes.append(replace(n, TEXT=trimmed[n.ROW_ID]))
        else:
            new_notes.append(n)
    return replace(bundle, noteevents=new_notes), removed_sentences


def _excise(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove each (start, end) span, last span first so earlier offsets stay valid."""
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[end:]
    return text
