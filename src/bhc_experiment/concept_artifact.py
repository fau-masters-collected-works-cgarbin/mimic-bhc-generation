"""Collect a cohort's unique texts, extract concepts, and write the concept artifact.

Extracts and appends texts to the JSONL incrementally, then writes the
.meta.json sidecar.
"""

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from bhc_experiment.concepts import (
    Concept,
    EntryExtractor,
    ExtractionResult,
    Rejection,
    text_sha256,
)
from bhc_experiment.loader import AdmissionBundle, LabEvent, NoteEvent, Prescription
from bhc_experiment.provenance import git_commit_hash

log = logging.getLogger(__name__)

# Version of the JSONL entry and .meta.json schemas.
SCHEMA_VERSION = 1

# Extraction and the on-disk append happen in chunks of this many texts, so a
# crash mid-build keeps everything extracted so far on disk instead of losing
# an entire cohort's worth of work to one long in-memory call, and progress
# can be logged between chunks.
_PROGRESS_CHUNK_SIZE = 500


def _match_dict(concept: Concept) -> dict:
    """One ``matches[]`` entry: the concept's codes, score, and offsets."""
    return {
        "cui": concept.cui,
        "term": concept.term,
        "semtypes": list(concept.semtypes),
        "score": concept.score,
        "start": concept.start,
        "end": concept.end,
    }


def _rejection_dict(rejection: Rejection) -> dict:
    """One ``rejected[]`` entry: the candidate's code, score, reason, and offsets."""
    return {
        "cui": rejection.cui,
        "score": rejection.score,
        "reason": rejection.reason,
        "start": rejection.start,
        "end": rejection.end,
    }


def serialize_entry(text: str, result: ExtractionResult) -> dict:
    """One JSONL line for ``text``'s extraction.

    No patient text reaches the artifact. The line has the text's hash and length.
    Each match has codes, scores, and offsets.
    """
    return {
        "text_sha256": text_sha256(text),
        "text_length": len(text),
        "n_entities": result.n_entities,
        "n_unlinked": result.n_unlinked,
        "matches": [_match_dict(c) for c in result.matches],
        "rejected": [_rejection_dict(r) for r in result.rejected],
    }


def collect_texts(
    bundles: list[AdmissionBundle],
    concept_text: Callable[[NoteEvent | LabEvent | Prescription], str],
) -> list[str]:
    """Every unique text this cohort needs concepts for.

    Discharge summaries are the ground-truth BHC, so they are excluded from the
    note loop. Deduplicates while preserving order.
    """
    texts: list[str] = []
    for bundle in bundles:
        for note in bundle.noteevents:
            if note.is_discharge_summary:
                continue
            texts.append(concept_text(note))
        for lab in bundle.labevents:
            texts.append(concept_text(lab))
        for rx in bundle.prescriptions:
            texts.append(concept_text(rx))
        texts.append(bundle.ground_truth_bhc)
    return list(dict.fromkeys(texts))


def _stored_hashes(jsonl_path: Path) -> set[str]:
    """The text_sha256 values already on disk, or empty if the file does not exist yet."""
    if not jsonl_path.exists():
        return set()
    hashes: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            hashes.add(json.loads(line)["text_sha256"])
    return hashes


def _verify_complete(texts: list[str], jsonl_path: Path) -> None:
    """Raise if any text this cohort needs is absent from the artifact."""
    stored = _stored_hashes(jsonl_path)
    needed = {text_sha256(t) for t in texts}
    missing = needed - stored
    if missing:
        msg = (
            f"concept artifact incomplete: {len(missing)} of {len(needed)} texts "
            f"missing from {jsonl_path}"
        )
        raise RuntimeError(msg)


def build_artifact(texts: list[str], extractor: EntryExtractor, jsonl_path: Path) -> int:
    """Extract every text in ``texts`` not already stored in ``jsonl_path``, appending as it goes.

    Reads the file's stored hashes first, so a resumed run only extracts
    what is missing. Returns the count of newly appended lines.
    """
    stored_hashes = _stored_hashes(jsonl_path)
    missing = [t for t in texts if text_sha256(t) not in stored_hashes]
    if missing:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = sum(len(t) for t in missing)
    done_texts = 0
    done_chars = 0
    n_new = 0
    for start in range(0, len(missing), _PROGRESS_CHUNK_SIZE):
        chunk = missing[start : start + _PROGRESS_CHUNK_SIZE]
        results = extractor.extract_entries(chunk)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            for text in chunk:
                entry = serialize_entry(text, results[text])
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_new += 1
        done_texts += len(chunk)
        done_chars += sum(len(t) for t in chunk)
        # Percentage on characters, since scispaCy time scales with length and
        # note bodies range from one line to half a megabyte.
        pct = (done_chars / total_chars * 100) if total_chars else 100.0
        log.info(
            "    %s/%s texts | %s/%s chars (%.0f%%)",
            f"{done_texts:,}",
            f"{len(missing):,}",
            _fmt_chars(done_chars),
            _fmt_chars(total_chars),
            pct,
        )

    _verify_complete(texts, jsonl_path)
    return n_new


_ONE_MILLION = 1_000_000
_ONE_THOUSAND = 1_000


def _fmt_chars(n: int) -> str:
    """Human-readable character count for progress logs (e.g. '18.2M', '412K')."""
    if n >= _ONE_MILLION:
        return f"{n / _ONE_MILLION:.1f}M"
    if n >= _ONE_THOUSAND:
        return f"{n / _ONE_THOUSAND:.0f}K"
    return str(n)


def write_meta(meta_path: Path, params: dict, *, n_texts: int, n_new: int) -> None:
    """Write the ``.meta.json`` sidecar for the latest build, replacing any earlier one.

    ``params`` is stored so a human can read the fingerprint's meaning and so
    this file's own stem can be recomputed from it.
    """
    meta = {
        "schema_version": SCHEMA_VERSION,
        "params": params,
        "git_commit": git_commit_hash(),
        "created_at": datetime.now(UTC).isoformat(),
        "n_texts": n_texts,
        "n_new": n_new,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
