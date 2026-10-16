"""Tests for scripts/pipeline/score_results.py helper functions."""

import json
from dataclasses import replace

import pytest
from score_results import _compute_coverage, _scorable_model_dirs, stale_preparation

from bhc_experiment.concept_store import ConceptStore
from bhc_experiment.concepts import concept_text, config_stem, text_sha256
from bhc_experiment.loader import AdmissionBundle
from bhc_experiment.prepared_store import PreparedStore, write_prepared
from bhc_experiment.prompt_builder import (
    filter_notes,
)

from .conftest import make_bundle, make_lab, make_note, make_prescription, write_model_dir

RUN_RECORD = {"status": "completed"}


class TestScorableModelDirs:
    """_scorable_model_dirs returns the model directories under a run directory."""

    def test_returns_scorable_model_dirs_sorted(self, tmp_path):
        run_dir = tmp_path / "all-records" / "format-matched-prompt"
        dir_b = write_model_dir(run_dir, "qwen_3_5_4b", RUN_RECORD, [{"hadm_id": 1}])
        dir_a = write_model_dir(run_dir, "gemma_4_e4b", RUN_RECORD, [{"hadm_id": 1}])

        assert _scorable_model_dirs(run_dir) == [dir_a, dir_b]

    def test_skips_dir_with_record_but_no_results(self, tmp_path):
        run_dir = tmp_path / "all-records" / "format-matched-prompt"
        write_model_dir(run_dir, "gemma_4_e4b", RUN_RECORD)

        assert _scorable_model_dirs(run_dir) == []


class TestStalePreparation:
    def test_names_the_model_dirs_generated_from_another_preparation(self, tmp_path):
        current = write_model_dir(tmp_path, "gemma_4_e4b", {"prepared_commit": "a" * 40})
        old = write_model_dir(tmp_path, "qwen_3_5_4b", {"prepared_commit": "b" * 40})

        assert stale_preparation([current, old], "a" * 40) == [old]


class _FakeExtractor:
    """Maps a text to a canned CUI set, empty for anything unmapped.

    Stands in for a ConceptStore/UmlsExtractor so _compute_coverage tests stay
    fast and deterministic. Only extract_cuis is called.
    """

    def __init__(self, by_text: dict[str, frozenset[str]]):
        self._by_text = by_text

    def extract_cuis(self, texts: list[str]) -> dict[str, frozenset[str]]:
        return {text: self._by_text.get(text, frozenset()) for text in texts}

    def extract_concepts(self, texts: list[str]):  # pragma: no cover - unused by this path
        raise NotImplementedError


NOTE_KEEP = make_note(1, "patient has hypertension")
NOTE_ERROR = make_note(2, "charted in error", ISERROR="1")
LAB_ABNORMAL = make_lab(10, LABEL="Potassium", VALUE="5.9", FLAG="abnormal")
LAB_NORMAL = make_lab(11, ITEMID=50983, LABEL="Sodium", VALUE="140")
RX_KEEP = make_prescription(20, DRUG="Aspirin")


def _coverage_bundle(reference: str = "reference text") -> AdmissionBundle:
    return make_bundle(
        [NOTE_KEEP, NOTE_ERROR],
        hadm_id=100,
        labs=[LAB_ABNORMAL, LAB_NORMAL],
        prescriptions=[RX_KEEP],
        reference=reference,
    )


def _prepared_coverage_bundle() -> AdmissionBundle:
    """A coverage bundle already filtered, as preparation would leave it.

    _coverage_bundle() stays unfiltered for TestComputeCoverage, which tests
    that attribute_records applies the filter itself. Anything read back from
    the prepared store must already be filtered, since format_admission_records
    only validates the filter.
    """
    bundle = _coverage_bundle()
    return replace(bundle, noteevents=filter_notes(bundle.noteevents))


class TestComputeCoverage:
    """_compute_coverage unions per-record concept_text CUI sets over exactly
    the records format_admission_records would have serialized.
    """

    def test_unions_only_the_records_the_prompt_would_have_kept(self):
        bundle = _coverage_bundle()
        cuis_by_text = {
            concept_text(NOTE_KEEP): frozenset({"C1"}),
            concept_text(NOTE_ERROR): frozenset({"C_ERR"}),  # dropped: an error note
            concept_text(LAB_ABNORMAL): frozenset({"C2"}),
            concept_text(LAB_NORMAL): frozenset(),
            concept_text(RX_KEEP): frozenset({"C3"}),
            bundle.ground_truth_bhc: frozenset({"C1", "C2", "C3", "C4", "C_ERR"}),
        }
        extractor = _FakeExtractor(cuis_by_text)

        coverage = _compute_coverage({bundle.admission.HADM_ID: bundle}, extractor)

        result = coverage[bundle.admission.HADM_ID]
        # C4 and C_ERR are in the reference, but no surviving record has them.
        # C_ERR would count as overlap if the error note were not excluded.
        assert result["n_ref"] == 5
        assert result["n_overlap"] == 3
        assert result["ceiling"] == pytest.approx(0.6)

    def test_admission_with_no_reference_concepts_is_omitted(self):
        bundle = _coverage_bundle("no concepts here")
        extractor = _FakeExtractor({})  # every text extracts to frozenset()

        coverage = _compute_coverage({bundle.admission.HADM_ID: bundle}, extractor)

        assert coverage == {}

    def test_store_miss_raises_instead_of_being_swallowed(self, tmp_path):
        # A real ConceptStore missing entries for the surviving records: the
        # KeyError attribute_records raises must propagate out of
        # _compute_coverage and reach the caller.
        bundle = _coverage_bundle()
        params = {"extractor": "scispacy", "extractor_version": 1, "extractor_params": {}}
        path = tmp_path / f"{config_stem(params)}.jsonl"
        # Only the reference text is stored. Every record's concept_text is
        # absent, so attribute_records' first lookup misses.
        entry = {"text_sha256": text_sha256(bundle.ground_truth_bhc), "matches": []}
        path.write_text(json.dumps(entry) + "\n")
        store = ConceptStore(tmp_path, params)

        with pytest.raises(KeyError):
            _compute_coverage({bundle.admission.HADM_ID: bundle}, store)


def _match(cui: str) -> dict:
    return {"cui": cui, "term": cui, "semtypes": [], "score": 1.0, "start": 0, "end": 1}


class TestCoverageOverThePreparedStore:
    def test_attributes_and_scores_the_prepared_bundle_without_key_error(self, tmp_path):
        # Regression for the sentence-dedup KeyError: scoring must attribute over
        # exactly the bundle the concept store was built over. Re-deduping first
        # produces a different bundle and the KeyError.
        cohort_file = "synthetic_cohort.json"
        bundle = _prepared_coverage_bundle()
        write_prepared(
            tmp_path, cohort_file, "sentence", [bundle], {}, git_commit="c" * 40, git_dirty=False
        )
        hadm_id = bundle.admission.HADM_ID
        bundles = {hadm_id: PreparedStore(tmp_path, cohort_file, "sentence").bundle(hadm_id)}

        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()
        store_params = {"extractor": "scispacy", "extractor_version": 1, "extractor_params": {}}
        concept_path = concepts_dir / f"{config_stem(store_params)}.jsonl"
        entries = [
            {"text_sha256": text_sha256(concept_text(NOTE_KEEP)), "matches": [_match("C1")]},
            {"text_sha256": text_sha256(concept_text(LAB_ABNORMAL)), "matches": [_match("C2")]},
            {"text_sha256": text_sha256(concept_text(LAB_NORMAL)), "matches": []},
            {"text_sha256": text_sha256(concept_text(RX_KEEP)), "matches": [_match("C3")]},
            {
                "text_sha256": text_sha256(bundle.ground_truth_bhc),
                "matches": [_match("C1"), _match("C2"), _match("C3")],
            },
        ]
        concept_path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
        concept_store = ConceptStore(concepts_dir, store_params)

        coverage = _compute_coverage(bundles, concept_store)

        result = coverage[bundle.admission.HADM_ID]
        assert result["n_ref"] == 3
        assert result["n_overlap"] == 3
        assert result["ceiling"] == pytest.approx(1.0)
