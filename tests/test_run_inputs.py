"""Tests for bhc_experiment.run_inputs: which admissions a run covers, and their bundles.

All identifiers are synthetic (see AGENTS.md MIMIC data safety).
"""

from types import SimpleNamespace

from bhc_experiment import run_inputs
from bhc_experiment.concept_spaces import concept_params
from bhc_experiment.config import (
    EXPERIMENT_COHORT_FILE,
    MAIN_CONCEPT_SPACE,
    ORACLE_STRATA,
    CliParameters,
)
from bhc_experiment.loader import AdmissionBundle
from bhc_experiment.oracle import AdmissionRecordConcepts, RecordConcepts, record_concepts_path
from bhc_experiment.prepared_store import prepared_stem
from bhc_experiment.prompt_builder import build_prompt
from bhc_experiment.results import ResultRecord, append_result
from bhc_experiment.run_inputs import oracle_selections
from mimic_data_prep.mimic import PREPARED_DIR

from .conftest import make_bundle, make_note

REFERENCE = frozenset({"C1"})


def _admission(hadm_id: int, cuis_per_record: list[set[str]]) -> AdmissionRecordConcepts:
    return AdmissionRecordConcepts(
        hadm_id=hadm_id,
        reference_cuis=REFERENCE,
        records=[
            RecordConcepts(("note", i), frozenset(cuis), {"m": 10})
            for i, cuis in enumerate(cuis_per_record, start=1)
        ],
    )


class TestOracleSelections:
    def test_long_strata_admissions_get_their_anchors_and_neighbors(self):
        record_concepts = {100: _admission(100, [set(), {"C1"}, set(), set()])}

        selections = oracle_selections({100: "64K"}, record_concepts, neighbor_expansion=1)

        assert selections == {100: {("note", 1), ("note", 2), ("note", 3)}}

    def test_short_strata_are_left_out(self):
        record_concepts = {100: _admission(100, [{"C1"}])}
        assert oracle_selections({100: "16K"}, record_concepts, neighbor_expansion=0) == {}

    def test_admissions_without_anchors_are_left_out(self):
        record_concepts = {100: _admission(100, [{"C9"}])}
        assert oracle_selections({100: "64K"}, record_concepts, neighbor_expansion=0) == {}


class _FakePreparedStore:
    def __init__(self, bundles: dict) -> None:
        self._bundles = bundles

    def bundle(self, hadm_id: int):
        return self._bundles[hadm_id]


class TestRunInputs:
    def _patch(self, monkeypatch, strata: dict[int, str], record_concepts: dict) -> None:
        admissions = [SimpleNamespace(hadm_id=h, token_stratum=s) for h, s in strata.items()]
        monkeypatch.setattr(
            run_inputs, "load_cohort", lambda _path: SimpleNamespace(admissions=admissions)
        )
        monkeypatch.setattr(
            run_inputs, "PreparedStore", lambda _dir, _cohort, _dedup: _FakePreparedStore({})
        )
        monkeypatch.setattr(run_inputs, "load_record_concepts", lambda _path: record_concepts)
        monkeypatch.setattr(run_inputs, "concept_params", lambda _path: {})
        monkeypatch.setattr(run_inputs, "record_concepts_path", lambda *_args: None)
        monkeypatch.setattr(run_inputs, "reduce_bundle", lambda bundle, ids: (bundle, ids))

    def test_oracle_covers_the_selected_admissions_with_reduced_bundles(self, monkeypatch):
        record_concepts = {200: _admission(200, [{"C1"}, set()])}
        self._patch(monkeypatch, {100: "8K", 200: "64K"}, record_concepts)
        monkeypatch.setattr(
            run_inputs,
            "PreparedStore",
            lambda _dir, _cohort, _dedup: _FakePreparedStore({200: "b200"}),
        )

        hadm_ids, load_bundle = run_inputs.run_inputs(
            CliParameters(prompt="format-matched", oracle_n=0)
        )

        assert hadm_ids == [200]
        assert load_bundle(200) == ("b200", {("note", 1)})


class TestDescribeInputs:
    def test_full_input_names_the_cohort_and_prepared_records(self):
        assert run_inputs.describe_inputs(CliParameters(prompt="format-matched")) == {
            "cohort_file": EXPERIMENT_COHORT_FILE,
            "prepared_file": f"{prepared_stem(EXPERIMENT_COHORT_FILE, 'sentence')}.jsonl",
        }

    def test_oracle_adds_the_strata_and_the_record_concepts_it_selected_from(self):
        inputs = run_inputs.describe_inputs(
            CliParameters(prompt="format-matched", oracle_n=2, deduplication="none")
        )
        expected_path = record_concepts_path(
            PREPARED_DIR, EXPERIMENT_COHORT_FILE, "none", concept_params(MAIN_CONCEPT_SPACE)
        )
        assert inputs == {
            "cohort_file": EXPERIMENT_COHORT_FILE,
            "prepared_file": f"{prepared_stem(EXPERIMENT_COHORT_FILE, 'none')}.jsonl",
            "oracle_strata": list(ORACLE_STRATA),
            "record_concepts_file": expected_path.name,
        }


def _bundle(hadm_id: int, note_text: str) -> AdmissionBundle:
    return make_bundle([make_note(1, note_text)], hadm_id=hadm_id)


class TestMismatchedPrompts:
    """mismatched_prompts finds the admissions whose input differs from what the model saw."""

    def _generate(
        self,
        model_dir,
        bundle: AdmissionBundle,
        *,
        error: str | None = None,
        system: str | None = None,
    ) -> None:
        prompt = build_prompt(bundle, "format-matched")
        append_result(
            model_dir,
            ResultRecord(
                hadm_id=bundle.admission.HADM_ID,
                template_name=prompt.template_name,
                prompt={"system": system or prompt.system, "user": prompt.user},
                llm_result=None,
                error=error,
                timestamp="2150-01-01T00:00:00+00:00",
            ),
        )

    def test_names_an_admission_whose_records_changed_and_one_left_out(self, tmp_path):
        for hadm_id in (100, 200, 300):
            self._generate(tmp_path, _bundle(hadm_id, "Synthetic note."))
        bundles = {
            100: _bundle(100, "Synthetic note."),
            200: _bundle(200, "Another note."),
        }

        assert run_inputs.mismatched_prompts(tmp_path, bundles, "format-matched") == [
            200,
            300,
        ]

    def test_names_an_admission_whose_system_message_changed(self, tmp_path):
        self._generate(
            tmp_path,
            _bundle(100, "Synthetic note."),
            system="An older system message.",
        )
        bundles = {100: _bundle(100, "Synthetic note.")}

        assert run_inputs.mismatched_prompts(tmp_path, bundles, "format-matched") == [100]

    def test_skips_an_admission_that_failed_before_its_prompt(self, tmp_path):
        self._generate(
            tmp_path,
            _bundle(100, "Synthetic note."),
            error="ValueError: boom",
        )
        bundles = {100: _bundle(100, "Another note.")}

        assert run_inputs.mismatched_prompts(tmp_path, bundles, "format-matched") == []
