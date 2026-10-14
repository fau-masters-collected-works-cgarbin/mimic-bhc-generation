"""The prepared-input artifact: deduplicated AdmissionBundles for one (cohort, dedup).

One JSONL line per admission, named ``<cohort_stem>-dedup-<method>``. The note
filter is fixed, so the cohort and dedup method identify the file completely.
"""

import dataclasses
import json
from pathlib import Path

from bhc_experiment.loader import (
    AdmissionBundle,
    AdmissionMetadata,
    BhcStructure,
    LabEvent,
    NoteEvent,
    Prescription,
)


def prepared_stem(cohort_file: str, deduplication: str) -> str:
    """The artifact's filename stem: ``<cohort_stem>-dedup-<method>``."""
    return f"{Path(cohort_file).stem}-dedup-{deduplication}"


def prepared_path(prepared_dir: Path, cohort_file: str, deduplication: str) -> Path:
    """The prepared-input JSONL for one cohort and dedup method."""
    return prepared_dir / f"{prepared_stem(cohort_file, deduplication)}.jsonl"


def _bundle_from_dict(record: dict) -> AdmissionBundle:
    """Rebuild an AdmissionBundle from its ``asdict`` form."""
    return AdmissionBundle(
        admission=AdmissionMetadata(**record["admission"]),
        ground_truth_bhc=record["ground_truth_bhc"],
        bhc_category=record["bhc_category"],
        bhc_structure=BhcStructure(**record["bhc_structure"]),
        noteevents=[NoteEvent(**note) for note in record["noteevents"]],
        labevents=[LabEvent(**lab) for lab in record["labevents"]],
        prescriptions=[Prescription(**rx) for rx in record["prescriptions"]],
    )


_REMOVED_SENTENCES_KEY = "_removed_sentences"


def write_prepared(  # noqa: PLR0913
    prepared_dir: Path,
    cohort_file: str,
    deduplication: str,
    bundles: list[AdmissionBundle],
    removed_sentences: dict[int, list],
    *,
    git_commit: str | None,
    git_dirty: bool,
) -> None:
    """Write ``bundles`` as one JSONL line each, and the commit preparation ran at
    to the ``.meta.json`` sidecar.

    ``removed_sentences`` is what sentence dedup cut from each admission, keyed
    by hadm_id. An admission it leaves out had nothing removed. It is stored
    under a reserved key beside the bundle's own fields.

    Overwrites any existing artifact for this cohort and dedup method.
    """
    prepared_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = prepared_path(prepared_dir, cohort_file, deduplication)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for bundle in bundles:
            record = dataclasses.asdict(bundle)
            record[_REMOVED_SENTENCES_KEY] = removed_sentences.get(bundle.admission.HADM_ID, [])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta = {"git_commit": git_commit, "git_dirty": git_dirty}
    jsonl_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def prepared_commit(prepared_dir: Path, cohort_file: str, deduplication: str) -> str | None:
    """The commit the prepared records were written at.

    The main concept space's record-concepts file is written in the same
    preparation pass, so this commit covers it too.
    """
    meta_path = prepared_path(prepared_dir, cohort_file, deduplication).with_suffix(".meta.json")
    return json.loads(meta_path.read_text())["git_commit"]


class PreparedStore:
    """Read-only access to one prepared-input artifact, keyed by hadm_id."""

    def __init__(self, prepared_dir: Path, cohort_file: str, deduplication: str) -> None:
        stem = prepared_stem(cohort_file, deduplication)
        jsonl_path = prepared_path(prepared_dir, cohort_file, deduplication)

        if not jsonl_path.exists():
            msg = (
                f"no prepared artifact for {stem}: expected {jsonl_path}. "
                "Run the preparation step first."
            )
            raise FileNotFoundError(msg)

        self._by_hadm_id: dict[int, AdmissionBundle] = {}
        self._removed_sentences: dict[int, list[dict]] = {}
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                bundle = _bundle_from_dict(record)
                self._by_hadm_id[bundle.admission.HADM_ID] = bundle
                self._removed_sentences[bundle.admission.HADM_ID] = record[_REMOVED_SENTENCES_KEY]

    def bundles(self) -> list[AdmissionBundle]:
        """Every bundle, in file order."""
        return list(self._by_hadm_id.values())

    def bundle(self, hadm_id: int) -> AdmissionBundle:
        """The bundle for one admission, or KeyError if hadm_id is absent."""
        return self._by_hadm_id[hadm_id]

    def removed_sentences(self, hadm_id: int) -> dict[int, list[dict]]:
        """Sentence-dedup removals for one admission keyed by row_id.

        Empty when the admission had no removals.
        """
        return {entry["row_id"]: entry["removed"] for entry in self._removed_sentences[hadm_id]}
