"""Experiment result persistence.

Persists per-admission results as append-only JSONL with a YAML sidecar
for run-level metadata, so an interrupted run can resume. Also derives the run
directory layout and stores per-admission review tags and notes.

Design: docs/specs/results-storage-design.md
"""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from bhc_experiment.config import INFERENCE, CliParameters
from bhc_experiment.llm import LlmResult

RUN_RECORD_FILENAME = "run_record.yaml"
RUN_RESULTS_FILENAME = "run_results.jsonl"
EVAL_RESULTS_FILENAME = "eval_results.jsonl"
EVAL_RECORD_FILENAME = "eval_record.yaml"
# Qualitative outlier review notes. They live in the run directory, one level
# above each model, because the notes apply to all the models in the run.
OUTLIER_REVIEW_NOTES_FILENAME = "outlier_review_notes.json"


@dataclass(frozen=True)
class ResultRecord:
    """One admission's generation result, ready for JSONL serialization.

    Fields mirror the JSONL schema in the spec.
    """

    hadm_id: int
    template_name: str
    prompt: dict  # {"system": str, "user": str}
    llm_result: LlmResult | None  # None on LLM failure
    error: str | None  # None on success, error message on failure
    timestamp: str  # ISO 8601 UTC

    @classmethod
    def from_dict(cls, data: dict) -> "ResultRecord":
        """Construct a ResultRecord from a JSON-deserialized dict."""
        llm_result_data = data["llm_result"]
        return cls(
            hadm_id=data["hadm_id"],
            template_name=data["template_name"],
            prompt=data["prompt"],
            llm_result=LlmResult.from_dict(llm_result_data) if llm_result_data else None,
            error=data["error"],
            timestamp=data["timestamp"],
        )


@dataclass(frozen=True)
class EvalRecord:
    """One admission's scores. Scoring rewrites the whole file, one line per admission.

    The commit and metrics that produced them are in ``eval_record.yaml``.
    """

    hadm_id: int
    scores: dict[str, float]
    # Input-coverage ceiling for this admission: {ceiling, n_ref, n_overlap}.
    # None when the reference had no concepts.
    coverage: dict | None

    @classmethod
    def from_dict(cls, data: dict) -> "EvalRecord":
        """Construct an EvalRecord from a JSON-deserialized dict."""
        return cls(hadm_id=data["hadm_id"], scores=data["scores"], coverage=data["coverage"])


@dataclass(frozen=True)
class ScoredResult:
    """A generation result paired with its evaluation scores.

    ``scores`` is an empty dict if no eval file exists or this admission
    has no scored metrics yet.
    """

    generation: ResultRecord
    scores: dict[str, float]

    @property
    def hadm_id(self) -> int:
        return self.generation.hadm_id

    @property
    def llm_result(self) -> LlmResult | None:
        return self.generation.llm_result

    @property
    def error(self) -> str | None:
        return self.generation.error

    @property
    def truncated(self) -> bool:
        return bool(self.llm_result and self.llm_result.truncated)


@dataclass(frozen=True)
class ReviewNote:
    """Qualitative observations recorded during outlier review of one admission.

    Persisted at the run level in ``outlier_review_notes.json`` so the notes
    survive across sessions. ``tags`` uses a controlled vocabulary, while
    ``notes`` is free prose. ``updated_at`` is overwritten on every save.
    """

    hadm_id: int
    tags: list[str]
    notes: str
    updated_at: str


def run_dir_path(results_root: Path, cli_parameters: CliParameters) -> Path:
    """``results_root / <selection> / <prompt>-prompt``, the run's directory."""
    return results_root / cli_parameters.run_name


def model_dir_path(results_root: Path, cli_parameters: CliParameters) -> Path:
    """One model's directory under the run directory, named by its registry key."""
    return run_dir_path(results_root, cli_parameters) / cli_parameters.model


def create_model_dir_and_record(  # noqa: PLR0913
    model_dir: Path,
    *,
    cli_parameters: CliParameters,
    total_admissions: int,
    git_commit: str | None,
    git_dirty: bool,
    prepared_commit: str | None,
    inputs: dict,
    runtime_environment: dict | None = None,
) -> None:
    """Create the model directory and write its initial run record.

    ``prepared_commit`` is the commit the inputs were prepared at, so scoring can
    refuse a run whose inputs were prepared again since. ``inputs`` names the
    files the run read. ``runtime_environment`` is what the server reported
    about itself.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    run_record = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "prepared_commit": prepared_commit,
        "cli_parameters": asdict(cli_parameters),
        "inference": asdict(INFERENCE),
        "inputs": inputs,
        "runtime_environment": runtime_environment,
        "total_admissions": total_admissions,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "completed_count": 0,
        "failed_count": 0,
        "status": "running",
    }
    record_path = model_dir / RUN_RECORD_FILENAME
    record_path.write_text(yaml.dump(run_record, default_flow_style=False, sort_keys=False))


def append_result(model_dir: Path, record: ResultRecord) -> None:
    """Append one JSON line to the results file and flush to disk.

    The file is opened in append mode and flushed immediately so a crash
    loses at most the current write.
    """
    jsonl_path = model_dir / RUN_RESULTS_FILENAME
    line = json.dumps(asdict(record), ensure_ascii=False)
    with jsonl_path.open("a") as f:
        f.write(line + "\n")
        f.flush()


def iter_result_records(model_dir: Path) -> Iterator[ResultRecord]:
    """Stream ResultRecord objects from run_results.jsonl.

    Yields nothing if the file does not exist (fresh run).
    """
    jsonl_path = model_dir / RUN_RESULTS_FILENAME
    if not jsonl_path.exists():
        return
    with jsonl_path.open() as f:
        for line in f:
            yield ResultRecord.from_dict(json.loads(line))


def iter_eval_records(model_dir: Path) -> Iterator[EvalRecord]:
    """Stream EvalRecord objects from eval_results.jsonl.

    Yields nothing if the file does not exist.
    """
    eval_path = model_dir / EVAL_RESULTS_FILENAME
    if not eval_path.exists():
        return
    with eval_path.open() as f:
        for line in f:
            yield EvalRecord.from_dict(json.loads(line))


def load_completed_ids(model_dir: Path) -> set[int]:
    """Read the results file and return the set of hadm_ids already recorded.

    Returns an empty set if the file does not exist (fresh run). Includes
    both successful and failed records.
    """
    return {r.hadm_id for r in iter_result_records(model_dir)}


def _compute_counts(model_dir: Path) -> tuple[int, int]:
    """Count successful and failed records in the results file.

    Returns (completed_count, failed_count). A record is failed if
    its ``error`` field is not None.
    """
    completed = 0
    failed = 0
    for record in iter_result_records(model_dir):
        if record.error is None:
            completed += 1
        else:
            failed += 1
    return completed, failed


def _update_record(model_dir: Path, status: str) -> None:
    """Read the run record, update status/counts/timestamp, write back."""
    record_path = model_dir / RUN_RECORD_FILENAME
    run_record = yaml.safe_load(record_path.read_text())
    completed, failed = _compute_counts(model_dir)
    run_record["status"] = status
    run_record["completed_at"] = datetime.now(UTC).isoformat()
    run_record["completed_count"] = completed
    run_record["failed_count"] = failed
    record_path.write_text(yaml.dump(run_record, default_flow_style=False, sort_keys=False))


def finalize_run(model_dir: Path) -> None:
    """Mark a run as completed with final counts from the results file."""
    _update_record(model_dir, "completed")


def mark_interrupted(model_dir: Path) -> None:
    """Mark a run as interrupted, with counts computed from the results file.

    "Interrupted" means the run stopped before every admission completed, whether
    from a SIGINT/SIGTERM or from ending with no successful generations.
    """
    _update_record(model_dir, "interrupted")


def save_review_note(run_dir: Path, *, hadm_id: int, tags: list[str], notes: str) -> ReviewNote:
    """Write or replace the review note for one admission.

    Writes atomically via a temp-file rename, to avoid a partially written
    notes file.
    """
    note = ReviewNote(
        hadm_id=hadm_id,
        tags=list(tags),
        notes=notes,
        updated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    path = run_dir / OUTLIER_REVIEW_NOTES_FILENAME
    existing: list[dict] = []
    if path.is_file():
        existing = json.loads(path.read_text())
    existing = [e for e in existing if e.get("hadm_id") != hadm_id]
    existing.append(asdict(note))
    existing.sort(key=lambda e: e["hadm_id"])

    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(existing, indent=2) + "\n")
    tmp_path.replace(path)
    return note
