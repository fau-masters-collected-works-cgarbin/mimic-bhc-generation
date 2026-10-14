"""Tests for bhc_experiment.oracle: per-record concept attribution, the record-concepts
artifact, and the oracle selection."""

import json
from math import isclose

import pytest

from bhc_experiment.concept_store import ConceptStore
from bhc_experiment.concepts import concept_text, config_stem, text_sha256
from bhc_experiment.loader import AdmissionBundle, NoteEvent
from bhc_experiment.oracle import (
    AdmissionRecordConcepts,
    RecordConcepts,
    admission_input_coverage,
    anchor_ids,
    attribute_records,
    build_admission_record_concepts,
    expand_neighbors,
    load_record_concepts,
    raw_input_concepts,
    raw_input_concepts_and_text,
    record_concepts_path,
    reduce_bundle,
    reference_cuis,
    selected_ids_for_admission,
    write_record_concepts,
)
from bhc_experiment.prompt_builder import (
    chronological_sort_key,
    format_note,
)

from .conftest import make_bundle, make_lab, make_note, make_prescription


class _FakeExtractor:
    """Maps a formatted text to a canned CUI set, empty for anything unmapped.

    Stands in for UmlsExtractor so attribution tests stay fast and deterministic
    without loading scispaCy or UMLS. It extracts each text independently.
    """

    def __init__(self, by_text: dict[str, frozenset[str]]):
        self._by_text = by_text

    def extract_cuis(self, texts: list[str]) -> dict[str, frozenset[str]]:
        return {text: self._by_text.get(text, frozenset()) for text in texts}


def _note(row_id: int, text: str, **overrides: object) -> NoteEvent:
    """A note charted at hour ``row_id``, so ROW_ID order is chronological order."""
    return make_note(row_id, text, CHARTTIME=f"2150-01-01 0{row_id}:00:00", **overrides)


NOTE_KEEP = _note(1, "patient has hypertension")
NOTE_ERROR = _note(2, "charted in error", ISERROR="1")
NOTE_DISCHARGE = _note(3, "discharge summary text", is_discharge_summary=True)
NOTE_PRE = _note(4, "outside hospital note", is_pre_admission=True)
LAB_ABNORMAL = make_lab(10, LABEL="Potassium", VALUE="5.9", FLAG="abnormal")
LAB_NORMAL = make_lab(
    11, CHARTTIME="2150-01-01 03:00:00", ITEMID=50983, LABEL="Sodium", VALUE="140"
)
RX_KEEP = make_prescription(20, DRUG="Aspirin")


def _bundle() -> AdmissionBundle:
    return make_bundle(
        [NOTE_KEEP, NOTE_ERROR, NOTE_DISCHARGE, NOTE_PRE],
        hadm_id=100,
        labs=[LAB_ABNORMAL, LAB_NORMAL],
        prescriptions=[RX_KEEP],
        reference="reference brief hospital course",
    )


def _cuis_by_text() -> dict[str, frozenset[str]]:
    # Keyed by concept_text: the raw note TEXT, lab LABEL, and rx DRUG.
    return {
        concept_text(NOTE_KEEP): frozenset({"C_HTN"}),
        concept_text(NOTE_PRE): frozenset(),
        concept_text(LAB_ABNORMAL): frozenset({"C_HYPERK"}),
        concept_text(LAB_NORMAL): frozenset(),
        concept_text(RX_KEEP): frozenset({"C_ASA"}),
    }


class TestAttributeRecords:
    def test_attributes_each_surviving_record_with_its_cuis(self):
        result = attribute_records(_bundle(), _FakeExtractor(_cuis_by_text()))
        by_id = {ar.record.ROW_ID: ar for ar in result}
        assert set(by_id) == {1, 4, 10, 11, 20}
        assert by_id[1].cui_set == frozenset({"C_HTN"})
        assert by_id[10].cui_set == frozenset({"C_HYPERK"})
        assert by_id[11].cui_set == frozenset()
        assert by_id[20].cui_set == frozenset({"C_ASA"})

    def test_formatted_text_and_sort_key_come_from_prompt_builder(self):
        result = attribute_records(_bundle(), _FakeExtractor(_cuis_by_text()))
        note = next(ar for ar in result if ar.record.ROW_ID == 1)
        assert note.formatted_text == format_note(NOTE_KEEP)
        assert note.sort_key == chronological_sort_key(NOTE_KEEP)

    def test_cui_set_comes_from_concept_text_not_formatted_text(self):
        # NOTE_KEEP's raw TEXT ("patient has hypertension") and its formatted
        # rendering (format_note adds a "[timestamp] NOTE (category): " prefix)
        # are different strings. Map each to a different CUI so a regression
        # back to extracting from the formatted text is caught by a CUI
        # mismatch on the value.
        note = NOTE_KEEP
        assert concept_text(note) != format_note(note)
        by_text = {
            concept_text(note): frozenset({"C_FROM_RAW_FIELD"}),
            format_note(note): frozenset({"C_FROM_FORMATTED_TEXT"}),
        }
        result = attribute_records(_bundle(), _FakeExtractor(by_text))
        attributed = next(ar for ar in result if ar.record.ROW_ID == note.ROW_ID)

        assert attributed.cui_set == frozenset({"C_FROM_RAW_FIELD"})
        assert attributed.formatted_text == format_note(note)


class TestAdmissionInputCoverage:
    def test_ceiling_counts_reference_concepts_present_in_the_raw_input(self):
        # The reference has three concepts: two are in the input (the note's HTN
        # and the abnormal lab's hyperkalemia) and one is not, so the ceiling is
        # 2/3. The input is keyed by raw field text, so if the code went back to
        # reading the formatted prompt it would match neither and score 0.
        bundle = _bundle()
        by_text = {
            concept_text(NOTE_KEEP): frozenset({"C_HTN"}),
            concept_text(LAB_ABNORMAL): frozenset({"C_HYPERK"}),
            bundle.ground_truth_bhc: frozenset({"C_HTN", "C_HYPERK", "C_ABSENT"}),
        }
        result = admission_input_coverage(bundle, _FakeExtractor(by_text))
        assert result.n_ref == 3
        assert result.n_overlap == 2
        assert isclose(result.ceiling, 2 / 3)

    def test_no_reference_concepts_is_none(self):
        # An empty reference leaves the ceiling undefined.
        result = admission_input_coverage(_bundle(), _FakeExtractor({}))
        assert result is None


class TestRawInputConcepts:
    def test_matches_the_text_variant(self):
        # Skipping the join must not skip a concept.
        args = (_bundle(), _FakeExtractor(_cuis_by_text()))
        assert raw_input_concepts(*args) == raw_input_concepts_and_text(*args)[0]


class TestRawInputConceptsAndText:
    def test_concepts_and_text_come_from_raw_fields(self):
        cuis, text = raw_input_concepts_and_text(_bundle(), _FakeExtractor(_cuis_by_text()))
        # The per-record union over the kept records (the normal lab maps to empty).
        assert cuis == frozenset({"C_HTN", "C_HYPERK", "C_ASA"})
        # The text is each kept record's raw field, so the note body is there and
        # the "Nursing" category the formatted prompt would add is not. It is
        # lowercased for the word search.
        assert concept_text(NOTE_KEEP).lower() in text
        assert "nursing" not in text
        assert text == text.lower()

    def test_record_concepts_carry_id_concepts_and_per_model_tokens_in_order(self):
        bundle = _bundle()
        cuis_by_text = {**_cuis_by_text(), bundle.ground_truth_bhc: frozenset({"C_HTN", "C_REF"})}
        counters = {"model_a": len, "model_b": lambda text: 2 * len(text)}

        admission = build_admission_record_concepts(bundle, _FakeExtractor(cuis_by_text), counters)

        assert admission.hadm_id == bundle.admission.HADM_ID
        assert admission.reference_cuis == frozenset({"C_HTN", "C_REF"})
        by_id = {r.record_id: r for r in admission.records}
        assert {("note", 1), ("lab", 10), ("rx", 20)} <= set(by_id)
        note = by_id[("note", 1)]
        assert note.cui_set == frozenset({"C_HTN"})
        assert note.token_counts == {
            "model_a": len(format_note(NOTE_KEEP)),
            "model_b": 2 * len(format_note(NOTE_KEEP)),
        }
        attributed = attribute_records(bundle, _FakeExtractor(cuis_by_text))
        in_order = sorted(attributed, key=lambda ar: ar.sort_key)
        assert [r.record_id[1] for r in admission.records] == [ar.record.ROW_ID for ar in in_order]


def _store_params() -> dict:
    """An arbitrary fingerprinted params dict, used only to derive the artifact's stem."""
    return {"extractor": "scispacy", "extractor_version": 1, "extractor_params": {}}


def _write_store(tmp_path, cuis_by_text: dict[str, frozenset[str]]) -> ConceptStore:
    """A ConceptStore over a synthetic on-disk artifact with exactly cuis_by_text."""
    params = _store_params()
    path = tmp_path / f"{config_stem(params)}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for text, cuis in cuis_by_text.items():
            entry = {
                "text_sha256": text_sha256(text),
                "matches": [
                    {"cui": cui, "term": cui, "semtypes": [], "score": 1.0, "start": 0, "end": 1}
                    for cui in sorted(cuis)
                ],
            }
            handle.write(json.dumps(entry) + "\n")
    return ConceptStore(tmp_path, params)


class TestAttributeRecordsAndReferenceCuisWithConceptStore:
    """attribute_records/reference_cuis behave the same whether the extractor is
    a fake in-memory ConceptExtractor or a ConceptStore reading the concepts
    back from a synthetic on-disk artifact, given the same known concepts.
    """

    def test_attribute_records_matches_a_fake_extractor_via_the_store(self, tmp_path):
        cuis_by_text = _cuis_by_text()
        store = _write_store(tmp_path, cuis_by_text)

        fake_result = attribute_records(_bundle(), _FakeExtractor(cuis_by_text))
        store_result = attribute_records(_bundle(), store)

        fake_by_id = {ar.record.ROW_ID: ar.cui_set for ar in fake_result}
        store_by_id = {ar.record.ROW_ID: ar.cui_set for ar in store_result}
        assert store_by_id == fake_by_id
        assert store_by_id[1] == frozenset({"C_HTN"})

    def test_reference_cuis_matches_a_fake_extractor_via_the_store(self, tmp_path):
        bundle = _bundle()
        cuis_by_text = {bundle.ground_truth_bhc: frozenset({"C_REF_ONE", "C_REF_TWO"})}
        store = _write_store(tmp_path, cuis_by_text)

        from_fake = reference_cuis(bundle, _FakeExtractor(cuis_by_text))
        from_store = reference_cuis(bundle, store)

        assert from_store == from_fake == frozenset({"C_REF_ONE", "C_REF_TWO"})


RECORD_IDS = ["r1", "r2", "r3", "r4", "r5"]


class TestExpandNeighbors:
    def test_zero_radius_returns_the_anchors_in_record_order(self):
        assert expand_neighbors(RECORD_IDS, {"r4", "r2"}, 0) == ["r2", "r4"]

    def test_radius_one_pulls_each_side_and_clips_at_the_ends(self):
        # r1 has no left neighbor and r5 no right, so the windows clip rather than
        # wrap. r3's window is r2..r4.
        assert expand_neighbors(RECORD_IDS, {"r1", "r3", "r5"}, 1) == RECORD_IDS

    def test_overlapping_windows_merge_without_repeats(self):
        assert expand_neighbors(RECORD_IDS, {"r2", "r3"}, 1) == ["r1", "r2", "r3", "r4"]

    def test_no_anchors_selects_nothing(self):
        assert expand_neighbors(RECORD_IDS, set(), 2) == []


REFERENCE_CUIS = frozenset({"A", "C", "D"})


def _admission() -> AdmissionRecordConcepts:
    """A five-record admission where notes 1, 3 and 5 have a reference concept."""
    cuis = [{"A"}, {"Z"}, {"C"}, {"Y"}, {"D"}]
    return AdmissionRecordConcepts(
        hadm_id=100,
        reference_cuis=REFERENCE_CUIS,
        records=[
            RecordConcepts(("note", i), frozenset(c), {"model_a": 100})
            for i, c in enumerate(cuis, start=1)
        ],
    )


class TestSelectedIdsForAdmission:
    def test_zero_expansion_feeds_only_the_anchors(self):
        admission = _admission()
        anchors = {("note", 1), ("note", 3), ("note", 5)}
        assert selected_ids_for_admission(admission, 0) == anchor_ids(admission) == anchors

    def test_expansion_adds_neighboring_records(self):
        assert selected_ids_for_admission(_admission(), 1) == {("note", i) for i in range(1, 6)}


class TestReduceBundle:
    def test_keeps_only_the_selected_records(self):
        reduced = reduce_bundle(_bundle(), {("note", 1), ("lab", 10)})
        assert [n.ROW_ID for n in reduced.noteevents] == [1]
        assert [lab.ROW_ID for lab in reduced.labevents] == [10]
        assert reduced.prescriptions == []


class TestOracleCeiling:
    def test_anchors_alone_keep_the_full_input_ceiling(self):
        # Every record with a reference concept is an anchor, so cutting an
        # admission down to its anchors (N=0) drops no reference concept. Scoring
        # an oracle run measures the ceiling over that reduced input.
        bundle = _bundle()
        extractor = _FakeExtractor(
            {**_cuis_by_text(), bundle.ground_truth_bhc: frozenset({"C_HTN", "C_ASA", "C_GONE"})}
        )
        admission = build_admission_record_concepts(bundle, extractor, {"m": len})

        reduced = reduce_bundle(bundle, selected_ids_for_admission(admission, 0))

        assert [lab.ROW_ID for lab in reduced.labevents] == []  # the lab had no reference concept
        full = admission_input_coverage(bundle, extractor)
        assert admission_input_coverage(reduced, extractor) == full
        assert full.n_overlap == 2


class TestRecordConceptsFile:
    def test_path_combines_the_prepared_name_and_the_extractor_stem(self, tmp_path):
        params = _store_params()
        path = record_concepts_path(tmp_path, "experiment_cohort.json", "sentence", params)
        assert path == tmp_path / (
            f"experiment_cohort-dedup-sentence.record-concepts-{config_stem(params)}.jsonl"
        )

    def test_write_and_load_are_inverse(self, tmp_path):
        # Record-id tuples and concept sets come back as their Python types.
        path = tmp_path / "record-concepts.jsonl"
        write_record_concepts(path, [_admission()])
        assert load_record_concepts(path) == {100: _admission()}

    def test_missing_file_says_how_to_build_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"prepare_inputs\.py"):
            load_record_concepts(tmp_path / "absent.jsonl")
