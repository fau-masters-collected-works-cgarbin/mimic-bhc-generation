"""Output length against reference length.

Questions answered:

- How long are reference Brief Hospital Courses outside the experiment cohort?
  The prompts' 250 to 600 word target is this range, rounded.
- Does each model's output meet the prompts' length target, per admission size?
  A prompt change passes the prompt test length gate when every model's median
  generated words fall within the target (see the testing appendix of
  docs/specs/prompt-builder-design.md).
- How long is the output relative to its admission's reference? The ratio is
  context for the gate. Output length barely follows reference length, so the
  ratio mostly shows how long each reference is.

Words are whitespace-separated tokens, counted the same way for generated and
reference text. Truncated generations are left out, because they stop at the
output cap and their length says nothing about the prompt.

Usage:

    uv run python scripts/analysis/length.py <run-folder>
"""

import argparse
import json
import logging
import statistics
from pathlib import Path

import pandas as pd

from bhc_experiment import reporting
from bhc_experiment.config import EXPERIMENT_COHORT_FILE
from bhc_experiment.results import RUN_RESULTS_FILENAME
from bhc_experiment.results_reader import cohort_attributes, load_reference_bhc, model_dirs
from mimic_data_prep.mimic import BHC_JSONL

logger = logging.getLogger(__name__)

REPORT_NAME = "length"
# The word range the format-matched and unformatted prompts ask for.
TARGET_WORDS = (250, 600)
# Every run's cohort (experiment, prompt test, smoke) is a subset of the experiment cohort.
STRATA_COHORT_FILE = EXPERIMENT_COHORT_FILE


def reference_word_quartiles(
    bhc_path: Path, excluded_hadm_ids: set[int]
) -> tuple[int, list[float]]:
    """Count and quartiles of reference word counts, successful extractions outside the cohort."""
    word_counts = []
    with bhc_path.open() as bhc_file:
        for line in bhc_file:
            extraction = json.loads(line)
            if not extraction["extraction_success"] or extraction["hadm_id"] in excluded_hadm_ids:
                continue
            word_counts.append(len(extraction["bhc_text"].split()))
    return len(word_counts), statistics.quantiles(word_counts, n=4)


def length_summary(lengths: pd.DataFrame) -> pd.DataFrame:
    """Generated words against the target and the gen/ref ratio, per model and stratum.

    Each model also gets an All row.
    """
    low, high = TARGET_WORDS
    rows = []
    for model, model_lengths in lengths.groupby("model"):
        strata = reporting.sort_strata(model_lengths["token_stratum"].unique())
        for stratum in [*strata, "All"]:
            if stratum == "All":
                stratum_lengths = model_lengths
            else:
                stratum_lengths = model_lengths[model_lengths["token_stratum"] == stratum]
            words = stratum_lengths["generated_words"]
            ratios = stratum_lengths["ratio"]
            rows.append(
                {
                    "model": model,
                    "token_stratum": stratum,
                    "n": len(stratum_lengths),
                    "median_words": words.median(),
                    "under_target": int((words < low).sum()),
                    "over_target": int((words > high).sum()),
                    "median_ratio": ratios.median(),
                    "under_half": int((ratios < 0.5).sum()),
                }
            )
    return pd.DataFrame(rows)


def _read_generations(model_dir: Path) -> dict[int, tuple[bool, str]]:
    """{hadm_id: (truncated, text)} read from the raw results file.

    Reads the JSON directly so archived runs, whose records predate the current
    result schema, can serve as a baseline.
    """
    generations = {}
    with (model_dir / RUN_RESULTS_FILENAME).open() as results_file:
        for line in results_file:
            result_record = json.loads(line)
            llm_result = result_record.get("llm_result")
            if llm_result is None:
                continue
            generations[result_record["hadm_id"]] = (
                llm_result.get("truncated", False),
                llm_result["text"] or "",
            )
    return generations


def per_admission_lengths(run_dir: Path, strata: dict[int, str]) -> tuple[pd.DataFrame, int]:
    """Per-admission generated words and gen/ref ratio, plus the truncated generation count."""
    generations_by_model = {}
    for model_dir in model_dirs(run_dir):
        generations_by_model[model_dir.name] = _read_generations(model_dir)
    all_hadm_ids = set()
    for generations in generations_by_model.values():
        all_hadm_ids.update(generations)
    references = load_reference_bhc(BHC_JSONL, all_hadm_ids)

    rows = []
    truncated_count = 0
    for model, generations in generations_by_model.items():
        for hadm_id, (truncated, text) in generations.items():
            if truncated:
                truncated_count += 1
                continue
            generated_words = len(text.split())
            rows.append(
                {
                    "model": model,
                    "token_stratum": strata[hadm_id],
                    "generated_words": generated_words,
                    "ratio": generated_words / len(references[hadm_id].split()),
                }
            )
    return pd.DataFrame(rows), truncated_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="path to the experiment run folder")
    args = parser.parse_args()

    attributes = cohort_attributes(STRATA_COHORT_FILE)
    strata = {hadm_id: attrs["token_stratum"] for hadm_id, attrs in attributes.items()}
    n_references, (p25, median, p75) = reference_word_quartiles(BHC_JSONL, set(attributes))
    lengths, truncated_count = per_admission_lengths(args.run_dir, strata)
    table = reporting.markdown_table(
        length_summary(lengths),
        float_format="{:.2f}",
        right_align=(
            "n",
            "median_words",
            "under_target",
            "over_target",
            "median_ratio",
            "under_half",
        ),
    )

    body = (
        "# Output length\n\n"
        f"_Run: `{args.run_dir.name}`_\n\n"
        f"Reference word counts over {n_references:,} successful extractions outside the "
        f"experiment cohort: 25th percentile {p25:.0f}, median {median:.0f}, "
        f"75th percentile {p75:.0f}.\n\n"
        f"The length target is {TARGET_WORDS[0]} to {TARGET_WORDS[1]} words. "
        "The ratio is generated words over reference words for the same admission. "
        f"{truncated_count} truncated generations are left out.\n\n"
        f"{table}\n"
    )
    report_path = reporting.write_report(args.run_dir, REPORT_NAME, body)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
