"""Scoring logic for decoupled evaluation of experiment results.

Loads generated texts from completed run directories, scores them against
ground truth using the metric set, and writes the eval results file.
Each pass rewrites the whole file.
"""

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from bhc_experiment.config import METRICS
from bhc_experiment.evaluator import MetricFn, evaluate_batch
from bhc_experiment.results import (
    EVAL_RECORD_FILENAME,
    EVAL_RESULTS_FILENAME,
    EvalRecord,
    iter_result_records,
)

logger = logging.getLogger(__name__)


def load_generated_texts(model_dir: Path) -> tuple[dict[int, str], dict[int, str]]:
    """Read run_results.jsonl and return (texts, errors) keyed by hadm_id.

    Records that cannot be scored go into the errors dict instead of texts.
    A truncated generation (hit max_completion_tokens) is excluded because
    the incomplete text would distort the scores.
    """
    texts: dict[int, str] = {}
    errors: dict[int, str] = {}
    for record in iter_result_records(model_dir):
        if record.error is not None:
            errors[record.hadm_id] = record.error
            continue
        if record.llm_result is None:
            errors[record.hadm_id] = "llm_result missing (no error recorded)"
            continue
        if record.llm_result.truncated:
            errors[record.hadm_id] = "truncated (hit max_completion_tokens limit)"
            continue
        texts[record.hadm_id] = record.llm_result.text
    return texts, errors


def score_model_dir(  # noqa: PLR0913
    model_dir: Path,
    scorers: dict[str, MetricFn],
    ground_truth: dict[int, str],
    *,
    coverage: dict[int, dict],
    git_commit: str | None,
    git_dirty: bool,
) -> None:
    """Score every generated admission in a model directory, replacing earlier scores.

    ``coverage`` is the per-admission input-coverage ceiling, keyed by hadm_id,
    written verbatim onto each EvalRecord. Admissions absent from it get None.
    The commit, the metrics, and the counts go once into
    ``eval_record.yaml``.
    """
    generated_texts, generation_errors = load_generated_texts(model_dir)
    for hadm_id, err in sorted(generation_errors.items()):
        logger.warning("Dropping HADM_ID %d from scoring in %s: %s", hadm_id, model_dir.name, err)

    hadm_ids = list(generated_texts)
    cands = [generated_texts[hadm_id] for hadm_id in hadm_ids]
    refs = [ground_truth[hadm_id] for hadm_id in hadm_ids]
    scores_by_admission = evaluate_batch(cands, refs, scorers)

    records = [
        EvalRecord(hadm_id=hadm_id, scores=scores, coverage=coverage.get(hadm_id))
        for hadm_id, scores in zip(hadm_ids, scores_by_admission, strict=True)
    ]
    # Temp file plus rename, so a crash mid-write leaves the previous scores whole.
    eval_path = model_dir / EVAL_RESULTS_FILENAME
    tmp_path = eval_path.with_suffix(eval_path.suffix + ".tmp")
    tmp_path.write_text(
        "".join(json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in records)
    )
    tmp_path.replace(eval_path)
    eval_record = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "metrics": list(METRICS),
        "n_scored": len(records),
        "n_excluded": len(generation_errors),
        "scored_at": datetime.now(UTC).isoformat(),
    }
    (model_dir / EVAL_RECORD_FILENAME).write_text(yaml.dump(eval_record, sort_keys=False))

    logger.info("Scored %d admissions in %s", len(records), model_dir.name)
    if generation_errors:
        logger.warning(
            "%s: %d admission(s) excluded from scoring (errors or truncated): %s",
            model_dir.name,
            len(generation_errors),
            sorted(generation_errors.keys()),
        )
