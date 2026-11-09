"""Per-admission sentence-level redundancy metrics (Landi/Lee ExactSubstr).

Clinicians copy text forward, so a sentence can appear verbatim further down the
same note or in a later note of the same admission. Finds those repeats and
reports their cost, in sentences and in tokens, for each admission.

Two duplication types are tracked: within-note (extra copies in a note
that already has the sentence) and between-note (the first copy in
each later note).
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import duckdb

from mimic_data_prep.mimic import pull_notes_for_batch
from mimic_data_prep.models import MODEL_KEYS
from mimic_data_prep.tokenizers import count_tokens

if TYPE_CHECKING:
    from spacy.language import Language
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Pinned parameters, adopted from Landi et al. 2023.
SENTENCE_REDUNDANCY_CONFIG: dict = {
    "min_match_chars": 100,
    "min_sentence_chars": 5,
    "segmenter": "en_core_sci_md (scispaCy), sentences via doc.sents",
    # keep-first is Landi's mechanic. Ordering the notes chronologically so that
    # "first" means "earliest" is this project's adaptation: it keeps a deduplicated
    # fact attached to the note that first recorded it.
    "retention": "keep-first, notes in charting-time order (see mimic.note_order_sql)",
    "reference": "Landi et al. 2023 (arXiv:2312.09469); ExactSubstr from Lee et al. 2022",
}

_MIN_MATCH_CHARS: int = SENTENCE_REDUNDANCY_CONFIG["min_match_chars"]
_MIN_SENTENCE_CHARS: int = SENTENCE_REDUNDANCY_CONFIG["min_sentence_chars"]

# How many admissions to pull note text for at a time.
_BATCH_SIZE: int = 500


def _repeated_region_masks(texts: list[str], k: int) -> list[list[bool]]:
    """Mark, per note, the characters belonging to text that repeats elsewhere in the admission.

    k is the minimum match length: a run of k identical characters counts
    as copy-paste, a short recurring phrase like "Patient stable." does not.

    Windows are built per note, so a window can never match across two
    notes, but the tally is pooled across the whole admission, so a passage
    copied from one note into another still counts. The masks come out the
    same regardless of the order ``texts`` arrives in.
    """
    counts: dict[str, int] = {}
    for text in texts:
        for i in range(len(text) - k + 1):
            window = text[i : i + k]
            counts[window] = counts.get(window, 0) + 1

    masks: list[list[bool]] = []
    for text in texts:
        mask = [False] * len(text)
        for i in range(len(text) - k + 1):
            if counts[text[i : i + k]] > 1:
                for j in range(i, i + k):
                    mask[j] = True
        masks.append(mask)
    return masks


class _DuplicateTracker:
    """Assigns each copy of a duplicated sentence its role.

    Copies are offered in note order, then in order within a note. The first copy
    anywhere is the survivor (``kept``). A later copy inside a note that already
    has the sentence is a ``within``-note duplicate. The first copy in a new
    note is a ``between``-note duplicate. Both duplicates point back at the
    survivor's note.
    """

    def __init__(self) -> None:
        self._survivor_note: dict[str, int] = {}
        self._notes_seen: dict[str, set[int]] = {}

    def classify(self, note_index: int, sentence: str) -> tuple[str, int | None]:
        """Return ``(kind, survivor_note_index)`` for this copy of ``sentence``."""
        if sentence not in self._survivor_note:
            self._survivor_note[sentence] = note_index
            self._notes_seen[sentence] = {note_index}
            return "kept", None
        survivor_index = self._survivor_note[sentence]
        if note_index in self._notes_seen[sentence]:
            return "within", survivor_index
        self._notes_seen[sentence].add(note_index)
        return "between", survivor_index


def classify_admission_sentences(
    notes: list[dict],
    segment_fn: Callable[[str], list[tuple[str, int, int]]],
) -> list[dict]:
    """Label every sentence in each note as kept, within, between, or unique.

    Returns one entry per note, ``{"note_index", "sentences"}``, where each
    sentence is ``{"text", "kind", "count", "survivor_note_index", "span"}``. ``kind``
    is ``kept`` (the surviving copy), ``within`` (an extra copy inside a note
    already having the sentence), ``between`` (the first copy in a later
    note), or ``unique``. ``count`` is the sentence's total covered occurrences
    across the admission. ``survivor_note_index`` is the note with the kept
    copy, set for ``within`` and ``between``. It is ``None`` for ``kept`` (the survivor itself) and
    ``unique``. ``span`` is the sentence's note-relative ``(start, end)``
    character offsets from the segmenter, locating the exact text in its note.
    A sentence is a duplicate only when it recurs verbatim and sits inside a
    repeated region of at least ``min_match_chars``.

    Note order changes which copy of a duplicate survives, but not the
    counts. Pass notes in chronological order. See ``mimic.note_order_sql``.
    """
    texts = [note.get("text") or "" for note in notes]
    per_note_spans = [segment_fn(text) for text in texts]
    masks = _repeated_region_masks(texts, _MIN_MATCH_CHARS)

    # Flatten to per-occurrence (note, text, span, covered) in note order.
    #
    # A sentence counts as a copy only when every character of it lies in the
    # repeated region. One that merely overlaps the region's edge is partly
    # text that appears nowhere else.
    occurrences: list[tuple[int, str, tuple[int, int], bool]] = []
    for note_index, spans in enumerate(per_note_spans):
        mask = masks[note_index]
        for sentence, start, end in spans:
            covered = len(sentence) > _MIN_SENTENCE_CHARS and all(mask[start:end])
            occurrences.append((note_index, sentence, (start, end), covered))

    counts: dict[str, int] = {}
    for _, sentence, _span, covered in occurrences:
        if covered:
            counts[sentence] = counts.get(sentence, 0) + 1
    dup_texts = {text for text, count in counts.items() if count > 1}

    tracker = _DuplicateTracker()
    rows: list[dict] = [{"note_index": ni, "sentences": []} for ni in range(len(notes))]
    for note_index, sentence, span, covered in occurrences:
        if covered and sentence in dup_texts:
            kind, survivor_index = tracker.classify(note_index, sentence)
            count = counts[sentence]
        else:
            kind, survivor_index, count = "unique", None, 1
        rows[note_index]["sentences"].append(
            {
                "text": sentence,
                "kind": kind,
                "count": count,
                "survivor_note_index": survivor_index,
                "span": span,
            }
        )
    return rows


def _aggregate_sentence_classification(
    classified: list[dict],
    tokenizers: "dict[str, PreTrainedTokenizerBase]",
) -> dict:
    """Within/between counts and per-model token sums from the classification.

    Each removed sentence is tokenized on its own, so the sums are estimates. The
    same text inside a full prompt tokenizes to a slightly different count, so an
    exact match against a measured reduction is not expected.
    """
    within_count = 0
    between_count = 0
    within_by_text: dict[str, int] = {}
    between_by_text: dict[str, int] = {}
    for row in classified:
        for sentence in row["sentences"]:
            if sentence["kind"] == "within":
                within_count += 1
                within_by_text[sentence["text"]] = within_by_text.get(sentence["text"], 0) + 1
            elif sentence["kind"] == "between":
                between_count += 1
                between_by_text[sentence["text"]] = between_by_text.get(sentence["text"], 0) + 1

    within_tok = dict.fromkeys(tokenizers, 0)
    between_tok = dict.fromkeys(tokenizers, 0)
    for model, tokenizer in tokenizers.items():
        for text, count in within_by_text.items():
            within_tok[model] += count * count_tokens(tokenizer, text)
        for text, count in between_by_text.items():
            between_tok[model] += count * count_tokens(tokenizer, text)

    return {
        "within_note_duplicate_count": within_count,
        "between_note_duplicate_count": between_count,
        "redundant_token_estimates": {"within_note": within_tok, "between_note": between_tok},
    }


def _empty_result() -> dict:
    zero = dict.fromkeys(MODEL_KEYS, 0)
    return {
        "within_note_duplicate_count": 0,
        "between_note_duplicate_count": 0,
        "duplicated_sentence_fraction": {"within_note": 0.0, "between_note": 0.0},
        "redundant_token_estimates": {"within_note": dict(zero), "between_note": dict(zero)},
    }


def _compute_admission_sentence_redundancy(
    notes: list[dict],
    tokenizers: "dict[str, PreTrainedTokenizerBase]",
    segment_fn: Callable[[str], list[tuple[str, int, int]]],
) -> dict:
    """Measure one admission's sentence redundancy, in sentences and in tokens.

    Adds token estimates and the duplicated-sentence fraction to the
    classification from ``classify_admission_sentences``.
    """
    if not notes:
        return _empty_result()
    classified = classify_admission_sentences(notes, segment_fn)
    total_sentences = sum(len(row["sentences"]) for row in classified)
    estimates = _aggregate_sentence_classification(classified, tokenizers)
    denom = total_sentences or 1
    return {
        "within_note_duplicate_count": estimates["within_note_duplicate_count"],
        "between_note_duplicate_count": estimates["between_note_duplicate_count"],
        "duplicated_sentence_fraction": {
            "within_note": estimates["within_note_duplicate_count"] / denom,
            "between_note": estimates["between_note_duplicate_count"] / denom,
        },
        "redundant_token_estimates": estimates["redundant_token_estimates"],
    }


def make_segment_fn(nlp: "Language") -> Callable[[str], list[tuple[str, int, int]]]:
    """Wrap a spaCy Language into a text -> [(sentence, start, end)] function."""

    def segment(text: str) -> list[tuple[str, int, int]]:
        if not text:
            return []
        out: list[tuple[str, int, int]] = []
        for sent in nlp(text).sents:
            stripped = sent.text.strip()
            if stripped:
                out.append((stripped, sent.start_char, sent.end_char))
        return out

    return segment


def load_sentence_segmenter() -> "Language":
    """Load en_core_sci_md for sentence segmentation.

    Its parser sets the sentence boundaries ``doc.sents`` reads.
    """
    import spacy  # noqa: PLC0415

    return spacy.load("en_core_sci_md")


def compute_sentence_redundancy(
    conn: duckdb.DuckDBPyConnection,
    hadm_ids: list[int],
    tokenizers: "dict[str, PreTrainedTokenizerBase]",
    segment_fn: Callable[[str], list[tuple[str, int, int]]],
) -> dict[int, dict]:
    """Per-admission sentence redundancy for the filtered pool.

    Returns ``{hadm_id: {within_note_duplicate_count, between_note_duplicate_count,
    duplicated_sentence_fraction, redundant_token_estimates}}``. Admissions with no
    source notes get all-zero values.
    """
    out: dict[int, dict] = {int(h): _empty_result() for h in hadm_ids}
    for start in range(0, len(hadm_ids), _BATCH_SIZE):
        batch = [int(h) for h in hadm_ids[start : start + _BATCH_SIZE]]
        notes_by_adm = pull_notes_for_batch(conn, batch)
        for hid in batch:
            notes = notes_by_adm.get(hid, [])
            out[hid] = _compute_admission_sentence_redundancy(notes, tokenizers, segment_fn)
        logger.info(
            "    sentence_redundancy: %d/%d admissions",
            min(start + _BATCH_SIZE, len(hadm_ids)),
            len(hadm_ids),
        )
    return out
