"""Build the filtered pool and persist it to disk.

Why: the filtered pool is the durable starting point for any admission-level
analysis (token-length distribution, stratification, sampling). Computing it
is expensive (several joins + per-admission tokenization), so it is built
once and reused from disk.

Questions answered:
- Which admissions pass the pool-level filters recorded in POOL_DEFINITION?
- For each admission, how many usable notes, labs, and prescriptions are
  attached, and what is the per-category note breakdown?
- How many tokens does each admission contribute, decomposed by source
  component (notes, labs_all, labs_abnormal, prescriptions,
  demographics) so analyses can recompose totals on demand without
  re-tokenizing?
- How many tokens does the BHC ground truth contain, so generation
  budgets and reference lengths are commensurable with the input?

Output: cohorts/filtered_pool.json

The output has raw record-level data only.

Set up the environment before running this script. See docs/setup.md
for the install steps.
"""

import argparse
import logging
from pathlib import Path

import dotenv
import duckdb

from bhc_experiment.config import FILTERED_POOL_FILE
from bhc_experiment.provenance import file_sha256, git_head
from mimic_data_prep.cohort import load_filtered_pool
from mimic_data_prep.mimic import BHC_JSONL, COHORTS_DIR, PROJECTS_DIR
from mimic_data_prep.sampling import write_filtered_pool_file
from mimic_data_prep.tokenizers import load_all_tokenizers, tokenizer_metadata

POOL_PATH = COHORTS_DIR / FILTERED_POOL_FILE

# Per-admission token-budget cap. 87.5% of 128K (Llama 3.2 3B and Gemma 4
# E4B nominal context), leaving ~16K for prompt template, structured
# admission summary, and BHC output. Applied against the maximum
# per-admission count across the three pivot tokenizers, so an admission
# in the pool fits the model with the smallest context window.
POOL_TOKEN_BUDGET = 112_000

# Thresholds pinned here so the output file documents exactly which filters
# produced it.
POOL_DEFINITION = {
    "metavision_only": True,
    "filters": {
        "los_days_min": 3,
        "charting_days_min": 3,
        "note_count_min": 2,
        "lab_count_min": 10,
        "bhc_char_count_max": 15000,
        "max_total_tokens_max": POOL_TOKEN_BUDGET,
    },
    "max_total_tokens_rule": ("max across Llama 3.2 3B, Qwen3.5-4B, and Gemma 4 E4B tokenizers"),
}

RECORDS_INCLUDED = {
    "notes": "all except discharge summaries and error-marked",
    "labs": "all",
    "prescriptions": "all",
    "pre_admission_notes": "included",
    "demographics": "included",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    dotenv.load_dotenv()  # HF_TOKEN for the gated tokenizers
    conn = duckdb.connect()

    logger.info("Step 1/3: Building filtered pool with per-model token counts")
    tokenizers = load_all_tokenizers()
    df = load_filtered_pool(
        conn,
        tokenizers=tokenizers,
    )
    logger.info("  Filtered pool before token budget: %d admissions", len(df))

    logger.info("Step 2/3: Applying %d-token max-of-three pool filter", POOL_TOKEN_BUDGET)
    before = len(df)
    df = df[df["max_total_tokens"] <= POOL_TOKEN_BUDGET].reset_index(drop=True)
    dropped = before - len(df)
    pool_filter_stats = {
        "before_token_budget": before,
        "dropped_above_token_budget": dropped,
    }
    logger.info(
        "  Dropped %d of %d admissions (%.1f%%) above the %d-token budget. Pool size: %d",
        dropped,
        before,
        100 * dropped / before if before else 0.0,
        POOL_TOKEN_BUDGET,
        len(df),
    )

    logger.info("Step 3/3: Writing pool file to %s", POOL_PATH)
    # This script lives at scripts/pipeline/, so the repo root is three levels up.
    script_repo = Path(__file__).resolve().parent.parent.parent
    extraction_repo = PROJECTS_DIR / "mimic-bhc-extraction"
    write_filtered_pool_file(
        df,
        path=POOL_PATH,
        bhc_jsonl_path=str(BHC_JSONL),
        bhc_jsonl_sha256=file_sha256(BHC_JSONL),
        bhc_extraction_commit=git_head(extraction_repo),
        build_script_commit=git_head(script_repo),
        pool_definition=POOL_DEFINITION,
        tokenizer={
            **tokenizer_metadata(tokenizers),
            "records_included": RECORDS_INCLUDED,
        },
        pool_filter_stats=pool_filter_stats,
    )
    logger.info("Done")


if __name__ == "__main__":
    main()
