"""The Concept record and the ConceptExtractor protocol.

``Concept``'s fields are a fixed, serializable set, sorted where order would
otherwise vary, so one record serializes to JSON and back.
"""

import hashlib
import json
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

from bhc_experiment.loader import LabEvent, NoteEvent, Prescription


class Concept(NamedTuple):
    """One linked concept mention in a source text.

    cui: the UMLS Concept Unique Identifier.
    term: the extractor's own label for the concept (scispaCy gives
        ``entity.canonical_name``). Different extractors give different
        strings here, so only ``cui`` compares across them.
    semtypes: every UMLS semantic type the knowledge base gives this CUI,
        sorted for determinism, including the types outside an extractor's
        inclusion filter.
    score: the top linking candidate's confidence.
    start: 0-indexed start offset of the mention in the source text.
    end: exclusive end offset. ``text[start:end]`` is exactly the matched
        span.
    """

    cui: str
    term: str
    semtypes: tuple[str, ...]
    score: float
    start: int
    end: int


class Rejection(NamedTuple):
    """A linking candidate that had a CUI but was dropped, with why.

    cui: the dropped candidate's UMLS Concept Unique Identifier.
    score: the top linking candidate's confidence.
    reason: why the candidate was dropped (e.g. "blocklisted",
        "out_of_filter").
    start: 0-indexed start offset of the mention in the source text.
    end: exclusive end offset.
    """

    cui: str
    score: float
    reason: str
    start: int
    end: int


class ExtractionResult(NamedTuple):
    """The full per-text extraction record: what linked and why the rest didn't.

    matches: concepts that passed every filter.
    rejected: candidates that had a CUI but were dropped, with reason.
    n_entities: total NER proposals in the text, including dropped ones.
    n_unlinked: proposals that never reached a usable entity, either no
        candidate CUI at all or a CUI the knowledge base does not know.
    """

    matches: list[Concept]
    rejected: list[Rejection]
    n_entities: int
    n_unlinked: int


@runtime_checkable
class ConceptExtractor(Protocol):
    """Per-text CUI sets and per-mention concept records.

    ``extract_cuis`` supports set-overlap metrics where only "which concepts
    were present" matters.
    """

    def extract_cuis(self, texts: list[str]) -> dict[str, frozenset[str]]: ...

    def extract_concepts(self, texts: list[str]) -> dict[str, list[Concept]]: ...


@runtime_checkable
class EntryExtractor(ConceptExtractor, Protocol):
    """What writing the concept artifact needs: the full record behind each text."""

    def extract_entries(self, texts: list[str]) -> dict[str, ExtractionResult]: ...


def concept_text(record: NoteEvent | LabEvent | Prescription) -> str:
    """The raw MIMIC field to run concept extraction over, one per record type.

    Raw field text. The record's prompt-formatted string embeds timestamps,
    VALUE, FLAG, and CATEGORY into the rendered text, which injects spurious
    concepts and corrupts entity offsets.
    """
    if isinstance(record, NoteEvent):
        return record.TEXT
    if isinstance(record, LabEvent):
        return record.LABEL
    # Prescription
    return record.DRUG


def spans_by_cui(matches: list[Concept], text: str) -> dict[str, set[str]]:
    """{cui: matched lowercased span texts}, sliced from ``text`` at each match's offsets."""
    spans: dict[str, set[str]] = {}
    for concept in matches:
        spans.setdefault(concept.cui, set()).add(text[concept.start : concept.end].strip().lower())
    return spans


def extractor_config_sha256(params: dict) -> str:
    """SHA-256 hex digest of an extractor's fingerprinted parameters.

    Encodes ``params`` with keys sorted and no extra whitespace, so the same
    content hashes identically regardless of construction order. Lists inside
    ``params`` must already be sorted, since sorting keys does not reach into them.
    """
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    """SHA-256 hex digest of a text's UTF-8 bytes, the concept artifact's per-entry key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_stem(params: dict) -> str:
    """The concept artifact's filename stem, derived from ``params``."""
    return f"{params['extractor']}-{extractor_config_sha256(params)[:12]}"


def concept_artifact_path(concepts_dir: Path, params: dict) -> Path:
    """The concept artifact JSONL that ``params`` fingerprints."""
    return concepts_dir / f"{config_stem(params)}.jsonl"
