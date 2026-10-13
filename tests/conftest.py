"""Shared pytest fixtures and helpers."""

import dataclasses
import json
import sys
import typing
from pathlib import Path

import pytest
import yaml

from bhc_experiment.config import CliParameters
from bhc_experiment.loader import (
    AdmissionBundle,
    AdmissionMetadata,
    BhcStructure,
    LabEvent,
    NoteEvent,
    Prescription,
)
from bhc_experiment.results import (
    EVAL_RESULTS_FILENAME,
    RUN_RECORD_FILENAME,
    RUN_RESULTS_FILENAME,
)
from mimic_data_prep.models import MODEL_KEYS


def per_model(value: object) -> dict[str, object]:
    """Spread one value across every model key."""
    return dict.fromkeys(MODEL_KEYS, value)


def make_note(row_id: int = 1, text: str = "Synthetic note.", **overrides: object) -> NoteEvent:
    """A synthetic Nursing note, charted after admission, that preparation keeps."""
    note = NoteEvent(
        ROW_ID=row_id,
        CHARTDATE="2150-01-01",
        CHARTTIME="2150-01-01 01:00:00",
        CATEGORY="Nursing",
        TEXT=text,
        ISERROR=None,
        is_pre_admission=False,
        is_discharge_summary=False,
        DESCRIPTION="Report",
    )
    return dataclasses.replace(note, **overrides)


def make_lab(row_id: int = 10, **overrides: object) -> LabEvent:
    """A synthetic potassium result."""
    lab = LabEvent(
        ROW_ID=row_id,
        CHARTTIME="2150-01-01 02:00:00",
        ITEMID=50971,
        LABEL="Potassium",
        VALUE="4.5",
        VALUEUOM="mEq/L",
        FLAG=None,
    )
    return dataclasses.replace(lab, **overrides)


def make_prescription(row_id: int = 20, **overrides: object) -> Prescription:
    """A synthetic aspirin order."""
    prescription = Prescription(
        ROW_ID=row_id,
        STARTDATE="2150-01-02",
        ENDDATE=None,
        DRUG="Aspirin",
        DOSE_VAL_RX="81",
        DOSE_UNIT_RX="mg",
        ROUTE="PO",
    )
    return dataclasses.replace(prescription, **overrides)


def make_bundle(
    notes: list[NoteEvent] | None = None,
    *,
    hadm_id: int = 99999,
    subject_id: int = 10,
    labs: list[LabEvent] | None = None,
    prescriptions: list[Prescription] | None = None,
    reference: str = "synthetic reference summary",
    bhc_category: str = "narrative",
) -> AdmissionBundle:
    """A synthetic admission. Missing record lists are empty."""
    return AdmissionBundle(
        admission=AdmissionMetadata(
            HADM_ID=hadm_id,
            SUBJECT_ID=subject_id,
            ADMITTIME="2150-01-01 00:00:00",
            DISCHTIME="2150-01-05 00:00:00",
            GENDER="F",
            SERVICE="MED",
            los_days=4,
            age=60,
        ),
        ground_truth_bhc=reference,
        bhc_category=bhc_category,
        bhc_structure=BhcStructure(category=bhc_category, numbering_convention=None, modifiers=[]),
        noteevents=list(notes or []),
        labevents=list(labs or []),
        prescriptions=list(prescriptions or []),
    )


def make_meta(model_key: str = "gemma_4_e4b", prompt: str = "brief") -> dict:
    """A completed run record for one model and prompt variant."""
    return {
        "cli_parameters": dataclasses.asdict(CliParameters(model=model_key, prompt=prompt)),
        "status": "completed",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_model_dir(
    run_dir: Path,
    model_key: str,
    run_record: dict,
    results: list[dict] | None = None,
    evals: list[dict] | None = None,
) -> Path:
    """Write a model directory: the run record, plus the results and eval files when given."""
    model_dir = run_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / RUN_RECORD_FILENAME).write_text(yaml.dump(run_record, sort_keys=False))
    if results is not None:
        _write_jsonl(model_dir / RUN_RESULTS_FILENAME, results)
    if evals is not None:
        _write_jsonl(model_dir / EVAL_RESULTS_FILENAME, evals)
    return model_dir


def pool_admission(hadm_id: int = 100, **overrides) -> dict:
    """Factory for a minimally valid filtered-pool admission dict.

    Matches the schema written by ``write_filtered_pool_file``.
    """
    rtc = {
        "notes": 12_000,
        "labs_all": 0,
        "labs_abnormal": 0,
        "prescriptions": 0,
        "demographics": 0,
    }
    record = {
        "hadm_id": hadm_id,
        "subject_id": 1000 + hadm_id,
        "bhc_char_count": 2500,
        "bhc_word_count": 400,
        "bhc_token_counts": per_model(520),
        "bhc_category": "narrative",
        "bhc_numbering_convention": None,
        "bhc_modifiers": [],
        "los_days": 5,
        "note_count": 7,
        "charting_days": 4,
        "lab_count": 25,
        "lab_abnormal_count": 5,
        "prescription_count": 3,
        "admittime": "2150-01-01 08:00:00",
        "dischtime": "2150-01-06 14:00:00",
        "hospital_expire_flag": 0,
        "gender": "M",
        "age": 60,
        "mesh_age_group": "Middle Aged",
        "service": "MED",
        "record_token_counts": per_model(rtc),
        "max_total_tokens": 12_000,
        "narrative_record_token_counts": per_model(8_500),
        "temporal_density": {
            "first_note_offset_hours": -1.5,
            "last_note_offset_hours": -0.5,
            "clinical_span_hours": 72.0,
            "gap_hours_between_notes": {"p50": 6.0, "p75": 8.0, "p90": 12.0, "max": 24.0},
            "notes_per_day": {"p50": 2.0, "p75": 3.0, "p90": 4.0, "max": 5.0},
        },
        "timestamp_completeness": {
            "null_charttime_note_count": 2,
            "null_charttime_note_categories": {"ECG": 1, "Echo": 1},
        },
        "source_diversity": {
            "distinct_lab_items": 25,
            "distinct_drugs": 8,
            "distinct_services_touching": 1,
        },
        "pre_admission_notes": {
            "count": 0,
            "token_counts": per_model(0),
            "categories": {},
        },
        "note_categories": {"Nursing": 4, "Physician ": 3},
    }
    record.update(overrides)
    return record


def pool_dict(admissions: list[dict], metadata: dict | None = None) -> dict:
    """Wrap an admission list in the pool-file envelope (`metadata` + `admissions`)."""
    return {
        "metadata": metadata or {"pool_size": len(admissions)},
        "admissions": admissions,
    }


@pytest.fixture
def admission_dict():
    """Factory for a minimally valid cohort admission dict.

    Matches the schema written by ``write_cohort_file`` and expected by
    ``Admission.from_dict``.
    """

    def build(hadm_id: int = 100, **overrides) -> dict:
        rtc = {
            "notes": 12000,
            "labs_all": 2000,
            "labs_abnormal": 400,
            "prescriptions": 500,
            "demographics": 500,
        }
        record = {
            "hadm_id": hadm_id,
            "subject_id": 10000 + hadm_id,
            "bhc_category": "narrative",
            "token_stratum": "16K",
            "los_quartile": "Q1",
            "los_days": 4,
            "note_category_availability": "clinical_narrative",
            "bhc_numbering_convention": None,
            "bhc_modifiers": [],
            "note_count": 12,
            "note_categories": {"Nursing": 8, "Radiology": 4},
            "lab_count": 50,
            "lab_abnormal_count": 8,
            "prescription_count": 8,
            "bhc_word_count": 200,
            "bhc_token_counts": per_model(260),
            "charting_days": 5,
            "record_token_counts": per_model(rtc),
            "max_total_tokens": 15_000,
            "narrative_record_token_counts": per_model(9_500),
            "hospital_expire_flag": 0,
            "temporal_density": {
                "first_note_offset_hours": -2.5,
                "last_note_offset_hours": -1.0,
                "clinical_span_hours": 96.0,
                "gap_hours_between_notes": {"p50": 4.0, "p75": 6.0, "p90": 12.0, "max": 24.0},
                "notes_per_day": {"p50": 3.0, "p75": 4.0, "p90": 5.0, "max": 6.0},
            },
            "timestamp_completeness": {
                "null_charttime_note_count": 4,
                "null_charttime_note_categories": {"ECG": 1, "Echo": 1, "Radiology": 2},
            },
            "source_diversity": {
                "distinct_lab_items": 30,
                "distinct_drugs": 12,
                "distinct_services_touching": 2,
            },
            "pre_admission_notes": {
                "count": 1,
                "token_counts": per_model(150),
                "categories": {"Nursing": 1},
            },
            "gender": "M",
            "age": 60,
            "mesh_age_group": "Middle Aged",
            "service": "MED",
            "admittime": "2150-01-01 00:00:00",
            "dischtime": "2150-01-05 00:00:00",
        }
        record.update(overrides)
        return record

    return build


class Mention(typing.NamedTuple):
    """One fake entity's linker output: CUI, its KB semantic types, and the
    span/score fields most tests do not need to control precisely.

    ``linked=False`` models an NER proposal with no linking candidate at
    all (``ent._.kb_ents`` empty): ``cui``/``tuis`` are ignored and no KB
    entry is registered for it.
    """

    cui: str
    tuis: list[str]
    start: int | None = None
    end: int | None = None
    score: float = 0.99
    linked: bool = True


def install_fake_scispacy(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    ent_map: dict[str, list["Mention | tuple[str, list[str]] | None"]],
) -> type:
    """Stub spacy + scispacy.linking with a fake pipeline.

    ``ent_map`` keys are input texts. Each value is a list of mentions the
    fake pipeline should emit as entities for that text: a plain
    ``(cui, tuis)`` pair, a full ``Mention``, or ``None`` for an NER
    proposal with no linking candidate. The fake supplies all attributes
    ``UmlsExtractor`` reads, with deterministic placeholder values for the
    ones a given test does not control (canonical_name, and a
    ``Mention``'s unset start/end/score). Lets any test construct a real
    ``UmlsExtractor`` without the ``umls_concept`` extra installed.

    Returns the FakeNLP class, which records piped texts on ``piped_texts``.
    """

    def _as_mention(item: "Mention | tuple[str, list[str]] | None") -> Mention:
        if item is None:
            return Mention(cui="", tuis=[], linked=False)
        if isinstance(item, Mention):
            return item
        cui, tuis = item
        return Mention(cui, tuis)

    class FakeEntity:
        def __init__(self, cui: str, tuis: list[str]) -> None:
            self.types = tuis
            self.canonical_name = f"name_for_{cui}"

    class FakeEnt:
        def __init__(self, mention: Mention, index: int) -> None:
            kb_ents = [(mention.cui, mention.score)] if mention.linked else []
            self._ = type("U", (), {"kb_ents": kb_ents})()
            start = mention.start if mention.start is not None else index * 10
            self.start_char = start
            self.end_char = mention.end if mention.end is not None else start + 1

    class FakeDoc:
        def __init__(self, ents: list[FakeEnt]) -> None:
            self.ents = ents

    class FakeKB:
        def __init__(self, entries: dict[str, FakeEntity]) -> None:
            self.cui_to_entity = entries

    class FakeLinker:
        def __init__(self, kb: FakeKB) -> None:
            self.kb = kb

    kb_entries: dict[str, FakeEntity] = {}
    for mention_list in ent_map.values():
        for item in mention_list:
            mention = _as_mention(item)
            if not mention.linked:
                continue
            kb_entries[mention.cui] = FakeEntity(mention.cui, mention.tuis)
    linker = FakeLinker(FakeKB(kb_entries))

    class FakeNLP:
        # Class-level so tests can assert across instances. Cleared on
        # every install so each test starts from a clean slate.
        piped_texts: typing.ClassVar[list[str]] = []

        def __init__(self) -> None:
            self._pipe = linker

        def add_pipe(self, *_args, **_kwargs):
            return self._pipe

        def get_pipe(self, _name):
            return self._pipe

        def __call__(self, text: str):
            mentions = [_as_mention(item) for item in ent_map.get(text, [])]
            return FakeDoc([FakeEnt(m, idx) for idx, m in enumerate(mentions)])

        def pipe(self, texts, **_kwargs):
            ts = list(texts)
            FakeNLP.piped_texts.extend(ts)
            return [self.__call__(t) for t in ts]

    FakeNLP.piped_texts = []
    fake_spacy = type("S", (), {"load": staticmethod(lambda _m: FakeNLP())})
    fake_linking = type("L", (), {"EntityLinker": object})
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    monkeypatch.setitem(sys.modules, "scispacy", type("M", (), {}))
    monkeypatch.setitem(sys.modules, "scispacy.linking", fake_linking)
    return FakeNLP
