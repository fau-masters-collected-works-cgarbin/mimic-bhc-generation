"""Tests for the per-admission sentence_redundancy logic."""

import itertools

from mimic_data_prep.models import MODEL_KEYS
from mimic_data_prep.sentence_redundancy import (
    _aggregate_sentence_classification,
    _compute_admission_sentence_redundancy,
    _repeated_region_masks,
    classify_admission_sentences,
    make_segment_fn,
)


class _FakeTokenizer:
    """One token per whitespace-separated word. Deterministic, no network."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG002, FBT001, FBT002
        return [0] * len(text.split())


def _tokenizers() -> dict[str, _FakeTokenizer]:
    return dict.fromkeys(MODEL_KEYS, _FakeTokenizer())


def test_repeated_region_marks_only_the_repeat():
    block = "the patient was admitted overnight and remained stable "  # 55 chars
    text = block + "UNIQUE MIDDLE SEGMENT XYZ " + block
    mask = _repeated_region_masks([text], k=20)[0]
    # The repeated block is covered at both occurrences.
    first = text.index(block)
    second = text.index(block, first + 1)
    assert all(mask[first : first + len(block)])
    assert all(mask[second : second + len(block)])
    # The unique middle is not covered.
    mid = text.index("UNIQUE")
    assert not any(mask[mid : mid + len("UNIQUE MIDDLE SEGMENT XYZ")])


def test_repeated_region_pools_the_tally_across_notes():
    # A block appearing once in each of two notes repeats across the admission,
    # so it is marked in both.
    block = "the patient was admitted overnight and remained stable "
    texts = [block + "FIRST NOTE TAIL", "SECOND NOTE HEAD " + block]
    masks = _repeated_region_masks(texts, k=20)
    assert all(masks[0][: len(block)])
    second_start = texts[1].index(block)
    assert all(masks[1][second_start : second_start + len(block)])
    # The text unique to each note is untouched.
    assert not any(masks[0][len(block) :])
    assert not any(masks[1][:second_start])


def test_repeated_region_never_matches_across_a_note_boundary():
    # No note repeats anything, inside itself or in another note. But the tail of
    # note 0 and the head of note 1 spell "abcd\nefgh", and so do the tail of note 2
    # and the head of note 3, so joining the notes would find that at both junctions
    # and mark it. It is a string no note contains.
    masks = _repeated_region_masks(["0000abcd", "efgh1111", "2222abcd", "efgh3333"], k=8)
    assert not any(any(mask) for mask in masks)


def test_repeated_region_no_repeat_is_all_false():
    assert not any(_repeated_region_masks(["abcdefghij"], k=4)[0])


def test_repeated_region_below_k_is_all_false():
    # A repeat shorter than k must not be marked.
    assert not any(_repeated_region_masks(["abXabY"], k=4)[0])


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Fake segmenter for tests: split on '. ', returning (sentence, start, end).

    Offsets are the character span of the stripped sentence in ``text``,
    mirroring how the real segmenter reports native span offsets.
    """
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


def test_classify_within_between_and_unique():
    # The shared sentence appears twice in note 0 and once in note 1
    # (T=3, m=2): one kept + one within in note 0, one between in note 1.
    shared = (
        "Patient admitted overnight with chest pain and shortness of breath, "
        "started on heparin drip and serial troponins were ordered"
    )
    notes = [
        {"text": shared + ". " + shared + ". Tail of the first note here now today."},
        {"text": shared + ". Other unique sentence in the second note today."},
    ]
    rows = classify_admission_sentences(notes, _split_sentences)
    assert [s["kind"] for s in rows[0]["sentences"]] == ["kept", "within", "unique"]
    assert [s["kind"] for s in rows[1]["sentences"]] == ["between", "unique"]
    # count is the sentence's total covered occurrences across the admission.
    assert rows[0]["sentences"][0]["count"] == 3
    # within/between point at the note with the surviving copy (note 0,
    # where "kept" was assigned). Kept and unique have None.
    assert rows[0]["sentences"][0]["survivor_note_index"] is None  # the kept copy
    assert rows[0]["sentences"][1]["survivor_note_index"] == 0  # within
    assert rows[1]["sentences"][0]["survivor_note_index"] == 0  # between
    assert rows[1]["sentences"][1]["survivor_note_index"] is None  # unique


def test_classify_surfaces_note_relative_spans():
    # Each classified sentence has its note-relative (start, end). The span
    # must index the sentence's own note,
    # not the concatenation the classifier builds for cross-note detection, so
    # slicing the note text by the span recovers the sentence.
    shared = (
        "Patient admitted overnight with chest pain and shortness of breath, "
        "started on heparin drip and serial troponins were ordered"
    )
    notes = [
        {"text": shared + ". " + shared + ". Tail of the first note here now today."},
        {"text": shared + ". Other unique sentence in the second note today."},
    ]
    rows = classify_admission_sentences(notes, _split_sentences)
    for note_index, row in enumerate(rows):
        text = notes[note_index]["text"]
        for sentence in row["sentences"]:
            start, end = sentence["span"]
            assert text[start:end].strip() == sentence["text"]


def test_classify_short_sentences_are_unique():
    shared = "x" * 200  # one long repeated region, but "Hi" is short after split
    notes = [{"text": "Hi. " + shared}, {"text": "Hi. " + shared}]
    rows = classify_admission_sentences(notes, _split_sentences)
    short = [s["kind"] for row in rows for s in row["sentences"] if s["text"] == "Hi"]
    assert short == ["unique", "unique"]
    long_kinds = [s["kind"] for row in rows for s in row["sentences"] if s["text"] == shared]
    assert long_kinds == ["kept", "between"]


def test_aggregate_counts_labels_and_tokens():
    # Two within + one between of a 3-word sentence. The unique is ignored.
    s = "stable overnight please"
    classified = [
        {
            "note_index": 0,
            "sentences": [
                {"text": s, "kind": "kept", "count": 4},
                {"text": s, "kind": "within", "count": 4},
                {"text": s, "kind": "within", "count": 4},
                {"text": "noted no acute distress", "kind": "unique", "count": 1},
            ],
        },
        {"note_index": 1, "sentences": [{"text": s, "kind": "between", "count": 4}]},
    ]
    out = _aggregate_sentence_classification(classified, _tokenizers())
    assert out["within_note_duplicate_count"] == 2
    assert out["between_note_duplicate_count"] == 1
    for model in MODEL_KEYS:
        # within: 2 * 3 tokens = 6. between: 1 * 3 tokens = 3.
        assert out["redundant_token_estimates"]["within_note"][model] == 6
        assert out["redundant_token_estimates"]["between_note"][model] == 3


def test_compute_admission_duplicated_sentence_fraction():
    # The orchestration's unique job over the units is the fraction:
    # duplicate count divided by total segmented sentences.
    shared = (
        "Patient admitted overnight with chest pain and shortness of breath, "
        "started on heparin drip and serial troponins were ordered"
    )
    notes = [
        {"text": shared + ". First note unique tail sentence goes here now."},
        {"text": shared + ". Second note unique tail sentence goes here now."},
    ]
    out = _compute_admission_sentence_redundancy(notes, _tokenizers(), _split_sentences)
    # shared appears once per note (between count 1) over 4 sentences -> 1/4.
    assert out["duplicated_sentence_fraction"]["between_note"] == 0.25
    assert out["duplicated_sentence_fraction"]["within_note"] == 0.0


def test_compute_admission_no_notes_is_zero():
    out = _compute_admission_sentence_redundancy([], _tokenizers(), _split_sentences)
    assert out["within_note_duplicate_count"] == 0
    assert out["between_note_duplicate_count"] == 0
    assert out["duplicated_sentence_fraction"] == {"within_note": 0.0, "between_note": 0.0}


class _FakeSpan:
    def __init__(self, text, start, end):
        self.text = text
        self.start_char = start
        self.end_char = end


class _FakeDoc:
    def __init__(self, spans):
        self.sents = spans


class _FakeNlp:
    """Stand-in spaCy Language: splits on '. ', exposing span char offsets."""

    def __call__(self, text):
        spans = []
        cursor = 0
        for piece in text.split(". "):
            stripped = piece.strip()
            if not stripped:
                cursor += len(piece) + 2
                continue
            start = text.index(stripped, cursor)
            spans.append(_FakeSpan(stripped, start, start + len(stripped)))
            cursor = start + len(stripped)
        return _FakeDoc(spans)


def test_make_segment_fn_uses_span_offsets():
    seg = make_segment_fn(_FakeNlp())
    assert seg("First sentence here. Second sentence here") == [
        ("First sentence here", 0, 19),
        ("Second sentence here", 21, 41),
    ]


def test_a_sentence_clipped_by_the_repeat_edge_is_not_a_duplicate():
    # A sentence must lie wholly inside the repeated region. Touching its edge
    # is not enough.
    #
    # Notes 0 and 2 both end with the sentence S. Each is preceded by a long block
    # that repeats into the next note, but that next note diverges 15 characters
    # into S, so the repeated region stops mid-sentence and clips S. S is not a
    # duplicate. The two blocks are wholly inside their own repeats, so they are.
    s = "wound clean dry and intact without surrounding erythema"
    block_a = (
        "Patient admitted overnight with chest pain and shortness of breath "
        "started on heparin drip today"
    )
    block_b = (
        "Patient reports gradual improvement in pain control and tolerated "
        "the procedure well overnight ok"
    )
    notes = [
        {"text": block_a + ". " + s + ". Ending unique to the first note."},
        {"text": block_a + ". " + s[:15] + "XX. Ending unique to the second note."},
        {"text": block_b + ". " + s + ". Ending unique to the third note."},
        {"text": block_b + ". " + s[:15] + "YY. Ending unique to the fourth note."},
    ]

    rows = classify_admission_sentences(notes, _split_sentences)
    kinds = {row["note_index"]: [sent["kind"] for sent in row["sentences"]] for row in rows}
    # Note 0: the block is kept (first copy), S is unique (clipped), tail unique.
    assert kinds[0] == ["kept", "unique", "unique"]
    # Note 2: S is still unique here, so no copy of it is ever removed.
    assert kinds[2] == ["kept", "unique", "unique"]

    out = _compute_admission_sentence_redundancy(notes, _tokenizers(), _split_sentences)
    # Two removals, one per block. S contributes none.
    assert out["between_note_duplicate_count"] == 2
    assert out["within_note_duplicate_count"] == 0


class TestNoteOrder:
    """Detection ignores note order. Retention follows it."""

    def test_counts_are_independent_of_note_order(self):
        # The sentence is 55 chars, shorter than the 100-char match window, and
        # sits at the end of its note. Joining the notes used to glue it to the
        # next note's opening text, forming a repeat longer than the window, so
        # whether it counted as a duplicate depended on which notes adjoined.
        sentence = "wound clean dry and intact without surrounding erythema"
        following = "patient tolerated the procedure well and remained afebrile"
        notes = [
            {"text": "Day one progress note. " + sentence},
            {"text": following + ". Plan is to continue current management."},
            {"text": "Day two progress note. " + sentence},
            {"text": following + ". Plan is to reassess in the morning."},
        ]
        counts = set()
        for order in itertools.permutations(range(len(notes))):
            out = _compute_admission_sentence_redundancy(
                [notes[i] for i in order], _tokenizers(), _split_sentences
            )
            counts.add((out["within_note_duplicate_count"], out["between_note_duplicate_count"]))
        # Neither repeat reaches the 100-char window inside a single note, so
        # neither is a duplicate, under every ordering.
        assert counts == {(0, 0)}

    def test_between_note_duplicates_are_found_without_joining(self):
        # Pooling the window counts across notes is what finds a cross-note
        # repeat. The join was never what did it.
        shared = (
            "Patient admitted overnight with chest pain and shortness of breath, "
            "started on heparin drip and serial troponins were ordered"
        )
        notes = [
            {"text": shared + ". Tail of the first note here now today."},
            {"text": shared + ". Other unique sentence in the second note today."},
        ]
        out = _compute_admission_sentence_redundancy(notes, _tokenizers(), _split_sentences)
        assert out["between_note_duplicate_count"] == 1
        assert out["within_note_duplicate_count"] == 0

    def test_within_note_duplicates_are_found_without_joining(self):
        shared = (
            "Patient admitted overnight with chest pain and shortness of breath, "
            "started on heparin drip and serial troponins were ordered"
        )
        notes = [{"text": shared + ". " + shared + ". Tail of the only note here today."}]
        out = _compute_admission_sentence_redundancy(notes, _tokenizers(), _split_sentences)
        assert out["within_note_duplicate_count"] == 1
        assert out["between_note_duplicate_count"] == 0

    def test_the_first_note_holding_a_sentence_keeps_it(self):
        shared = (
            "Patient admitted overnight with chest pain and shortness of breath, "
            "started on heparin drip and serial troponins were ordered"
        )
        first = {"text": shared + ". Tail of the earlier note here now today."}
        second = {"text": shared + ". Other unique sentence in the later note today."}

        rows = classify_admission_sentences([first, second], _split_sentences)
        assert rows[0]["sentences"][0]["kind"] == "kept"

        rows = classify_admission_sentences([second, first], _split_sentences)
        assert rows[0]["sentences"][0]["kind"] == "kept"
        assert rows[1]["sentences"][0]["kind"] == "between"
