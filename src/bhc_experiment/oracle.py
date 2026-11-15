"""Per-record concept attribution and the oracle selection.

The oracle sees the reference BHC and keeps every record that mentions a
reference concept (the anchors), plus the records within N positions of an
anchor (the neighbors). It is a measurement device for an upper bound on
retrieval quality, with no deployable equivalent.

Attribution reuses the prompt builder's formatters and filters, so the records
match what a full-input run feeds the model. Preparation freezes each
admission's attributed records in the record-concepts file, which the oracle
selection and the analyses read.
"""

import json
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from bhc_experiment.concepts import ConceptExtractor, concept_text, config_stem
from bhc_experiment.coverage import InputCoverage, input_coverage
from bhc_experiment.loader import AdmissionBundle, LabEvent, NoteEvent, Prescription
from bhc_experiment.prepared_store import prepared_stem
from bhc_experiment.prompt_builder import (
    chronological_sort_key,
    filter_notes,
    format_lab,
    format_note,
    format_prescription,
)


@dataclass(frozen=True)
class AttributedRecord:
    """A surviving record with its formatted text, concepts, and chronological position."""

    record: NoteEvent | LabEvent | Prescription
    formatted_text: str
    cui_set: frozenset[str]
    sort_key: tuple[str, int]


def format_record(record: NoteEvent | LabEvent | Prescription) -> str:
    """The record's prompt line, through the prompt builder's own formatter."""
    if isinstance(record, NoteEvent):
        return format_note(record)
    if isinstance(record, LabEvent):
        return format_lab(record)
    return format_prescription(record)


def record_id(record: NoteEvent | LabEvent | Prescription) -> tuple[str, int]:
    """The record's (kind, ROW_ID) identity, as the record-concepts file stores it."""
    # ROW_ID is table-scoped, so a note and a lab can share one. The kind tag
    # keeps the id unique across record types.
    if isinstance(record, NoteEvent):
        return ("note", record.ROW_ID)
    if isinstance(record, LabEvent):
        return ("lab", record.ROW_ID)
    return ("rx", record.ROW_ID)


def attribute_records(
    bundle: AdmissionBundle, extractor: ConceptExtractor
) -> list[AttributedRecord]:
    """Each record the prompt serializes, paired with the concepts extracted from its raw field.

    Using the raw field keeps the formatted line's timestamps and lab flags from
    adding concepts that were never clinical content.
    """
    records: list[NoteEvent | LabEvent | Prescription] = [
        *filter_notes(bundle.noteevents),
        *bundle.labevents,
        *bundle.prescriptions,
    ]
    concept_texts = [concept_text(record) for record in records]
    cuis_by_text = extractor.extract_cuis(concept_texts)
    return [
        AttributedRecord(
            record=record,
            formatted_text=format_record(record),
            cui_set=cuis_by_text[text],
            sort_key=chronological_sort_key(record),
        )
        for record, text in zip(records, concept_texts, strict=True)
    ]


def reference_cuis(bundle: AdmissionBundle, extractor: ConceptExtractor) -> frozenset[str]:
    """The reference concept set R: the CUIs of the reference BHC."""
    reference = bundle.ground_truth_bhc
    return extractor.extract_cuis([reference])[reference]


def admission_input_coverage(
    bundle: AdmissionBundle, extractor: ConceptExtractor
) -> InputCoverage | None:
    """The input-coverage ceiling for one admission: the share of the reference
    summary's concepts the extractor found in this input, the recoverable concepts.
    """
    attributed = attribute_records(bundle, extractor)
    input_cuis = frozenset().union(*(ar.cui_set for ar in attributed))
    return input_coverage(input_cuis, reference_cuis(bundle, extractor))


def raw_input_concepts(bundle: AdmissionBundle, extractor: ConceptExtractor) -> frozenset[str]:
    """One admission's input as the set of concepts it contains.

    Use this over ``raw_input_concepts_and_text`` unless a caller searches the
    text. That function joins every record into one string, which on a
    112K-token admission runs to hundreds of kilobytes.
    """
    attributed = attribute_records(bundle, extractor)
    return frozenset().union(*(ar.cui_set for ar in attributed))


def raw_input_concepts_and_text(
    bundle: AdmissionBundle, extractor: ConceptExtractor
) -> tuple[frozenset[str], str]:
    """One admission's input as the set of concepts it contains, and its text
    for a plain word search, lowercased.
    """
    attributed = attribute_records(bundle, extractor)
    input_cuis = frozenset().union(*(ar.cui_set for ar in attributed))
    input_text = "\n".join(concept_text(ar.record) for ar in attributed)
    return input_cuis, input_text.lower()


def expand_neighbors(
    ordered_ids: Sequence[Hashable], anchors: set[Hashable], n: int
) -> list[Hashable]:
    """Depth expansion: every id within n positions of an anchor.

    ``ordered_ids`` is the admission's records in chronological order, reduced
    to their ids. Only positions are read, so the expansion is the same for
    every model. n=0 returns exactly the anchor ids, in order. Overlapping and
    adjacent windows merge (each id appears once) and a window clips at either
    end.
    """
    positions = {record_id: i for i, record_id in enumerate(ordered_ids)}
    selected: set[int] = set()
    for anchor in anchors:
        pos = positions[anchor]
        lo = max(0, pos - n)
        hi = min(len(ordered_ids) - 1, pos + n)
        selected.update(range(lo, hi + 1))
    return [ordered_ids[i] for i in sorted(selected)]


# --- The record-concepts artifact ---


@dataclass(frozen=True)
class RecordConcepts:
    """One input record with its concepts and its token count under each model."""

    record_id: tuple[str, int]  # (kind, ROW_ID)
    cui_set: frozenset[str]
    token_counts: dict[str, int]  # model key -> token count of the record's prompt line

    def to_dict(self) -> dict:
        return {
            "record_id": list(self.record_id),
            "cui_set": sorted(self.cui_set),
            "token_counts": self.token_counts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecordConcepts":
        return cls(
            record_id=tuple(data["record_id"]),
            cui_set=frozenset(data["cui_set"]),
            token_counts=dict(data["token_counts"]),
        )


@dataclass(frozen=True)
class AdmissionRecordConcepts:
    """One admission's records in chronological order, and its reference concepts.

    Neighbors are adjacent entries of ``records``.
    """

    hadm_id: int
    reference_cuis: frozenset[str]
    records: list[RecordConcepts]

    def to_dict(self) -> dict:
        return {
            "hadm_id": self.hadm_id,
            "reference_cuis": sorted(self.reference_cuis),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdmissionRecordConcepts":
        return cls(
            hadm_id=data["hadm_id"],
            reference_cuis=frozenset(data["reference_cuis"]),
            records=[RecordConcepts.from_dict(record) for record in data["records"]],
        )


def build_admission_record_concepts(
    bundle: AdmissionBundle,
    extractor: ConceptExtractor,
    token_counters: dict[str, Callable[[str], int]],
) -> AdmissionRecordConcepts:
    """Attribute one admission's records and count each one's tokens per model."""
    attributed = sorted(attribute_records(bundle, extractor), key=lambda ar: ar.sort_key)
    records = [
        RecordConcepts(
            record_id=record_id(ar.record),
            cui_set=ar.cui_set,
            token_counts={
                model: count(ar.formatted_text) for model, count in token_counters.items()
            },
        )
        for ar in attributed
    ]
    return AdmissionRecordConcepts(
        hadm_id=bundle.admission.HADM_ID,
        reference_cuis=reference_cuis(bundle, extractor),
        records=records,
    )


def record_concepts_path(
    prepared_dir: Path, cohort_file: str, deduplication: str, extractor_params: dict
) -> Path:
    """Named by the prepared records it attributes and the concept artifact it read."""
    stem = prepared_stem(cohort_file, deduplication)
    return prepared_dir / f"{stem}.record-concepts-{config_stem(extractor_params)}.jsonl"


def write_record_concepts(path: Path, admissions: list[AdmissionRecordConcepts]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for admission in admissions:
            handle.write(json.dumps(admission.to_dict()) + "\n")


def load_record_concepts(path: Path) -> dict[int, AdmissionRecordConcepts]:
    """Every admission's record concepts, keyed by hadm_id."""
    if not path.exists():
        msg = f"no record-concepts file at {path}. Run scripts/pipeline/prepare_inputs.py first."
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        admissions = [AdmissionRecordConcepts.from_dict(json.loads(line)) for line in handle]
    return {admission.hadm_id: admission for admission in admissions}


# --- The oracle selection ---


def anchor_ids(admission: AdmissionRecordConcepts) -> set[tuple[str, int]]:
    """The anchors: every record that mentions a reference concept."""
    return {
        record.record_id
        for record in admission.records
        if record.cui_set & admission.reference_cuis
    }


def selected_ids_for_admission(
    admission: AdmissionRecordConcepts, neighbor_expansion: int
) -> set[tuple[str, int]]:
    """The record ids one admission feeds to generation.

    ``neighbor_expansion`` 0 selects the anchors alone. A larger value adds every
    record within that many positions of an anchor.
    """
    anchors = anchor_ids(admission)
    ordered_ids = [record.record_id for record in admission.records]
    return set(expand_neighbors(ordered_ids, anchors, neighbor_expansion))


def reduce_bundle(bundle: AdmissionBundle, selected_ids: set[tuple[str, int]]) -> AdmissionBundle:
    """A copy of the bundle with only the selected records.

    Each list is filtered by its own (kind, ROW_ID) pair, so a note and a lab
    that happen to share a ROW_ID stay distinct.
    """
    notes = [n for n in bundle.noteevents if ("note", n.ROW_ID) in selected_ids]
    labs = [lab for lab in bundle.labevents if ("lab", lab.ROW_ID) in selected_ids]
    prescriptions = [rx for rx in bundle.prescriptions if ("rx", rx.ROW_ID) in selected_ids]
    return replace(bundle, noteevents=notes, labevents=labs, prescriptions=prescriptions)
