"""Prepare the input for the pipeline: filtered, deduplicated records, their
concepts, and each admission's record concepts.

Usage:
    uv run python scripts/pipeline/prepare_inputs.py --dedup sentence
    uv run python scripts/pipeline/prepare_inputs.py --dedup none --smoke

Concepts are extracted in the main concept space, the one scoring reads.
--concept-space extracts another for a robustness check, over the records
already prepared.
"""

import argparse
import dataclasses
import functools
import logging
import multiprocessing
import os
import time
from pathlib import Path

import duckdb

from bhc_experiment import config
from bhc_experiment.concept_artifact import build_artifact, collect_texts, write_meta
from bhc_experiment.concept_spaces import concept_extractor, concept_params
from bhc_experiment.concept_store import ConceptStore
from bhc_experiment.concepts import concept_artifact_path, concept_text
from bhc_experiment.config import CONCEPT_SPACES, DEDUPLICATIONS, MAIN_CONCEPT_SPACE
from bhc_experiment.loader import AdmissionBundle, load_admission_bundle, prepare_cohort
from bhc_experiment.note_dedup import deduplicate_bundle_with_removed_sentences
from bhc_experiment.oracle import (
    build_admission_record_concepts,
    record_concepts_path,
    write_record_concepts,
)
from bhc_experiment.prepared_store import PreparedStore, write_prepared
from bhc_experiment.prompt_builder import filter_notes
from bhc_experiment.provenance import git_commit_hash, git_is_dirty, require_clean_tree
from mimic_data_prep.cohort import load_cohort
from mimic_data_prep.mimic import COHORTS_DIR, CONCEPTS_DIR, PREPARED_DIR
from mimic_data_prep.sentence_redundancy import load_sentence_segmenter, make_segment_fn
from mimic_data_prep.tokenizers import count_tokens, load_all_tokenizers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Log preparation progress every this many admissions. Sentence dedup segments
# every note with scispaCy, minutes of otherwise silent work on a full cohort.
_LOG_EVERY = 10


def _filter_bundle(bundle: AdmissionBundle) -> AdmissionBundle:
    """The bundle with the note filter applied."""
    return dataclasses.replace(bundle, noteevents=filter_notes(bundle.noteevents))


def prepare_inputs(
    cohort_file: str, deduplication: str
) -> tuple[list[AdmissionBundle], dict[int, list]]:
    """Every admission in the cohort, deduplicated and filtered.

    Also returns what sentence dedup cut from each admission, keyed by
    hadm_id, empty for every admission under ``none``.
    """
    cohort_path = COHORTS_DIR / cohort_file
    hadm_ids = [a.hadm_id for a in load_cohort(cohort_path).admissions]
    total = len(hadm_ids)

    conn = duckdb.connect()
    prepare_cohort(conn, hadm_ids)

    bundles: list[AdmissionBundle] = []
    removed_sentences: dict[int, list] = {}

    if deduplication == "sentence":
        log.info("Loading sentence segmenter...")
        segment_fn = make_segment_fn(load_sentence_segmenter())

        log.info("Starting sentence deduplication - it will take several minutes...")
        for i, hadm_id in enumerate(hadm_ids, start=1):
            deduped, removed = deduplicate_bundle_with_removed_sentences(
                load_admission_bundle(conn, hadm_id), "sentence", segment_fn
            )
            removed_sentences[hadm_id] = removed
            bundles.append(_filter_bundle(deduped))
            if i % _LOG_EVERY == 0 or i == total:
                log.info("Deduplicated %d/%d admissions", i, total)
    else:  # none: no dedup, no removals
        for i, hadm_id in enumerate(hadm_ids, start=1):
            bundles.append(_filter_bundle(load_admission_bundle(conn, hadm_id)))
            removed_sentences[hadm_id] = []
            if i % _LOG_EVERY == 0 or i == total:
                log.info("Loaded %d/%d admissions", i, total)

    conn.close()
    return bundles, removed_sentences


def prepared_bundles(
    cohort_file: str, deduplication: str, concept_space: str, prepared_dir: Path = PREPARED_DIR
) -> list[AdmissionBundle]:
    """The prepared bundles, prepared and written again for the main concept space.

    Another concept space reads the records already prepared. Writing them again
    stamps a new commit on them, and scoring refuses every run generated before.
    """
    if concept_space != MAIN_CONCEPT_SPACE:
        return PreparedStore(prepared_dir, cohort_file, deduplication).bundles()

    bundles, removed_sentences = prepare_inputs(cohort_file, deduplication)
    write_prepared(
        prepared_dir,
        cohort_file,
        deduplication,
        bundles,
        removed_sentences,
        git_commit=git_commit_hash(),
        git_dirty=git_is_dirty(),
    )
    log.info("Wrote %d prepared admission(s) to %s", len(bundles), prepared_dir)
    return bundles


def extract_concepts(bundles: list[AdmissionBundle], concept_space: str) -> dict:
    """Extract concepts over the prepared bundles into the content-addressed store.

    Returns the concept space's extractor params.
    """
    extractor = concept_extractor(concept_space, n_process=max(1, (os.cpu_count() or 1) - 1))
    params = concept_params(concept_space)
    jsonl_path = concept_artifact_path(CONCEPTS_DIR, params)

    texts = collect_texts(bundles, concept_text)
    n_new = build_artifact(texts, extractor, jsonl_path)
    write_meta(jsonl_path.with_suffix(".meta.json"), params, n_texts=len(texts), n_new=n_new)
    log.info(
        "Extracted concepts for %d unique text(s) (%d new) into %s", len(texts), n_new, jsonl_path
    )
    return params


def write_admission_record_concepts(
    bundles: list[AdmissionBundle],
    cohort_file: str,
    deduplication: str,
    params: dict,
    token_counters: dict,
) -> Path:
    """Write each admission's records with their concepts and per-model token counts."""
    concepts = ConceptStore(CONCEPTS_DIR, params)
    admissions = []
    for i, bundle in enumerate(bundles, start=1):
        admissions.append(build_admission_record_concepts(bundle, concepts, token_counters))
        if i % _LOG_EVERY == 0 or i == len(bundles):
            log.info("Record concepts: %d/%d admissions", i, len(bundles))
    path = record_concepts_path(PREPARED_DIR, cohort_file, deduplication, params)
    write_record_concepts(path, admissions)
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the frozen inputs for the experiments.")
    parser.add_argument(
        "--dedup",
        default="sentence",
        choices=DEDUPLICATIONS,
        help="Note deduplication, computed over the notes the filter keeps (default: sentence).",
    )
    parser.add_argument("--smoke", action="store_true", help="Prepare the smoke cohort.")
    parser.add_argument(
        "--concept-space",
        default=MAIN_CONCEPT_SPACE,
        choices=CONCEPT_SPACES,
        help=f"Concept space to extract in (default: {MAIN_CONCEPT_SPACE}, the one scoring reads).",
    )
    return parser.parse_args(argv)


def main() -> None:
    # Forked children get a memory copy of the parent (including the loaded
    # scispaCy pipeline) without serialization. macOS defaults multiprocessing
    # to "spawn", which fails to re-create the extractor in the child. This must
    # run before anything else touches multiprocessing.
    multiprocessing.set_start_method("fork", force=True)

    args = _parse_args()
    require_clean_tree()
    cohort_file = config.cohort_file(smoke=args.smoke)

    bundles = prepared_bundles(cohort_file, args.dedup, args.concept_space)
    params = extract_concepts(bundles, args.concept_space)

    log.info("Loading tokenizers for the per-record token counts...")
    token_counters = {
        key: functools.partial(count_tokens, tokenizer)
        for key, tokenizer in load_all_tokenizers().items()
    }
    started = time.perf_counter()
    path = write_admission_record_concepts(bundles, cohort_file, args.dedup, params, token_counters)
    log.info("Wrote %s in %.0fs", path, time.perf_counter() - started)


if __name__ == "__main__":
    main()
