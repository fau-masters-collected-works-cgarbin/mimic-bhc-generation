"""Score every model of one run with the metrics in ``config.METRICS``.

Usage:
    RESULTS=../phd-dissertation-experiment-results/results
    uv run python scripts/pipeline/score_results.py $RESULTS/all-records/format-matched-prompt
    uv run python scripts/pipeline/score_results.py $RESULTS/oracle-n-2/format-matched-prompt

Scores each model directory under the run, replacing earlier scores. The run's
recorded CLI parameters name the inputs. The input-coverage ceiling is measured
over the input the model saw, so an oracle run's ceiling is over its selected
records.
"""

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import dotenv

from bhc_experiment.concept_spaces import concept_params
from bhc_experiment.concept_store import ConceptStore
from bhc_experiment.concepts import ConceptExtractor
from bhc_experiment.config import MAIN_CONCEPT_SPACE, METRICS
from bhc_experiment.evaluator import build_scorers
from bhc_experiment.loader import AdmissionBundle
from bhc_experiment.oracle import admission_input_coverage
from bhc_experiment.prepared_store import prepared_commit
from bhc_experiment.provenance import git_commit_hash, git_is_dirty, require_clean_tree
from bhc_experiment.results import RUN_RESULTS_FILENAME
from bhc_experiment.results_reader import load_run_record, model_dirs, run_cli_parameters
from bhc_experiment.run_inputs import mismatched_prompts, run_inputs
from bhc_experiment.scorer import score_model_dir
from mimic_data_prep.mimic import CONCEPTS_DIR, PREPARED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)


def _compute_coverage(
    bundles: dict[int, AdmissionBundle],
    extractor: ConceptExtractor,
) -> dict[int, dict]:
    """Per-admission input-coverage ceiling, keyed by hadm_id.

    Admissions whose reference has no concepts have an undefined ceiling
    and are omitted.
    """
    coverage: dict[int, dict] = {}
    total = len(bundles)
    for i, (hadm_id, bundle) in enumerate(bundles.items(), start=1):
        result = admission_input_coverage(bundle, extractor)
        if result is not None:
            coverage[hadm_id] = asdict(result)
        # Log periodically so a long run does not sit silent between updates.
        if i % 10 == 0 or i == total:
            log.info("Input-coverage ceiling: %d/%d admissions", i, total)
    return coverage


def _scorable_model_dirs(run_dir: Path) -> list[Path]:
    """The model directories under a run that have a run record and results."""
    if not run_dir.is_dir():
        return []
    return [d for d in model_dirs(run_dir) if (d / RUN_RESULTS_FILENAME).exists()]


def stale_preparation(model_dirs: list[Path], current_commit: str | None) -> list[Path]:
    """The model directories generated from inputs prepared at another commit.

    Their generations came from records that may differ from the prepared files
    scoring reads now, so their scores and ceiling would not describe one input.
    """
    return [
        model_dir
        for model_dir in model_dirs
        if load_run_record(model_dir)["prepared_commit"] != current_commit
    ]


def main() -> None:
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Score every model of one run.")
    parser.add_argument("run_dir", type=Path, help="The run: results/<selection>/<prompt>-prompt")
    args = parser.parse_args()

    require_clean_tree()
    scorable = _scorable_model_dirs(args.run_dir)
    if not scorable:
        log.error("No scorable model directories under %s", args.run_dir)
        sys.exit(1)
    cli_parameters = run_cli_parameters(args.run_dir)
    stale = stale_preparation(
        scorable,
        prepared_commit(PREPARED_DIR, cli_parameters.cohort_file, cli_parameters.deduplication),
    )
    if stale:
        log.error(
            "Inputs were prepared again after these runs were generated: %s. "
            "Delete and regenerate them, or prepare at their commit.",
            [str(d) for d in stale],
        )
        sys.exit(1)
    log.info("Scoring %d model(s) under %s", len(scorable), args.run_dir)

    # Every model saw the same inputs, so the references and the ceiling are
    # computed once for the run.
    hadm_ids, load_bundle = run_inputs(cli_parameters)
    bundles = {hadm_id: load_bundle(hadm_id) for hadm_id in hadm_ids}
    mismatched = {d: mismatched_prompts(d, bundles, cli_parameters.prompt) for d in scorable}
    mismatched = {d: ids for d, ids in mismatched.items() if ids}
    if mismatched:
        log.error(
            "The code now builds a different input than these runs fed the model, so the "
            "ceiling would not describe what the model saw: %s. Delete and regenerate them.",
            {str(d): f"{len(ids)} admission(s)" for d, ids in mismatched.items()},
        )
        sys.exit(1)

    # Built once: expensive for BERTScore.
    scorers = build_scorers(list(METRICS))
    log.info("Metrics: %s", list(scorers.keys()))

    ground_truth = {hadm_id: bundle.ground_truth_bhc for hadm_id, bundle in bundles.items()}
    concepts = ConceptStore(CONCEPTS_DIR, concept_params(MAIN_CONCEPT_SPACE))
    coverage = _compute_coverage(bundles, concepts)

    git_commit, git_dirty = git_commit_hash(), git_is_dirty()
    for model_dir in scorable:
        log.info("Scoring %s", model_dir.name)
        status = load_run_record(model_dir)["status"]
        if status != "completed":
            log.warning("%s is %s, so its scores cover a partial run", model_dir.name, status)
        score_model_dir(
            model_dir,
            scorers,
            ground_truth,
            coverage=coverage,
            git_commit=git_commit,
            git_dirty=git_dirty,
        )
    log.info("Scoring complete for %d model(s).", len(scorable))


if __name__ == "__main__":
    main()
