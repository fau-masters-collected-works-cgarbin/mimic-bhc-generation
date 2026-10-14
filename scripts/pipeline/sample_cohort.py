"""Sample the experiment cohort from the filtered pool.

Why: experiments need a fixed, stratified sample to run against. The
design stratifies on admission size in tokens. BHC format, LOS quartile, and
note-category availability are reported as covariates. The sampler balances none
of them.

Questions answered:
- Which admissions form the experiment cohort under the seven-stratum design?
- How many admissions does each stratum have, and which strata underflow?
- What covariate distribution does each stratum reach the audit?

Input: cohorts/filtered_pool.json (built by build_filtered_pool.py with the
112K max-of-three pool filter applied).
Output: cohorts/experiment_cohort.json

See docs/specs/experiment-cohort-sampling-design.md for the full spec.
The rationale behind the stratification is documented in the dissertation.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

from bhc_experiment.config import EXPERIMENT_COHORT_FILE, FILTERED_POOL_FILE
from bhc_experiment.provenance import file_sha256, git_head
from mimic_data_prep.cohort import derive_note_category_availability, load_filtered_pool_file
from mimic_data_prep.mimic import COHORTS_DIR
from mimic_data_prep.sampling import (
    TOKEN_STRATA,
    TOKEN_STRATUM_EDGES,
    assign_los_quartiles,
    assign_token_stratum,
    sample_stratified,
    write_cohort_file,
)

# Chosen by experimenting until the cohort resembled the filtered pool
# as closely as possible while respecting the per-stratum targets.
SEED = 4

# Asymmetric per-stratum allocation. Smaller per-stratum N at long context
# reflects the quadratic cost of prefill at 96K-112K. The allocation rationale,
# the per-stratum compute budget, and the total n=132 are documented in the
# dissertation. The 4K stratum is structurally underflowing in the pool.
SAMPLES_PER_STRATUM: dict[str, int] = {
    "4K": 25,
    "8K": 25,
    "16K": 25,
    "32K": 20,
    "64K": 15,
    "96K": 12,
    "112K": 10,
}
POOL_PATH = COHORTS_DIR / FILTERED_POOL_FILE
COHORT_PATH = COHORTS_DIR / EXPERIMENT_COHORT_FILE
STRATIFICATION_AXES = ["token_stratum"]

# Written into the cohort manifest for reference.
TOKEN_STRATA_META = {
    "labels": list(TOKEN_STRATA),
    "edges": list(TOKEN_STRATUM_EDGES),
    "binning_rule": "max across Llama 3.2 3B, Qwen3.5-4B, and Gemma 4 E4B",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    logger.info("Building the cohort")

    # Step 1: Read the filtered pool. The pool builder already applied the
    # 112K max-of-three token-budget filter, so every admission here fits
    # the seven token strata. The pool is the only source of per-admission data
    # here. Querying MIMIC at sample time would let the cohort have a field the
    # pool never recorded, so the pool file would stop describing what was sampled.
    logger.info("Step 1: Read filtered pool from %s", POOL_PATH)
    df, pool_metadata = load_filtered_pool_file(POOL_PATH)
    cohort_size = len(df)
    logger.info("  Filtered pool: %d admissions", cohort_size)

    # Step 2: Derive note_category_availability as a covariate.
    df["note_category_availability"] = df["note_categories"].map(derive_note_category_availability)
    availability_counts = df["note_category_availability"].value_counts().to_dict()
    logger.info("Step 2: Note category availability: %s", availability_counts)

    # Step 3: LOS quartile covariate, reported per admission and never sampled on,
    # from the pool's
    # los_days distribution.
    df, los_edges = assign_los_quartiles(df)
    logger.info("Step 3: LOS quartile edges (days): %s", los_edges)

    # Step 4: Token stratum assignment, the one variable the sampler balances on.
    df = assign_token_stratum(df)
    stratum_counts = df["token_stratum"].value_counts().to_dict()
    logger.info("Step 4: Pool admissions per stratum: %s", stratum_counts)

    # Step 5: Stratified sampling across the seven strata.
    sample = sample_stratified(
        df,
        grouping_columns=STRATIFICATION_AXES,
        samples_per_group=SAMPLES_PER_STRATUM,
        seed=SEED,
    )
    logger.info(
        "Step 5: Sampled %d admissions across %d strata (planned total %d)",
        len(sample),
        len(stratum_counts),
        sum(SAMPLES_PER_STRATUM.values()),
    )

    # Step 6: Per-stratum target/actual counts for the manifest, so underflow
    # is visible without re-running the sampler.
    actuals = Counter(sample["token_stratum"].astype(str).tolist())
    per_stratum_counts = {
        label: {
            "target": SAMPLES_PER_STRATUM[label],
            "actual": int(actuals.get(label, 0)),
        }
        for label in TOKEN_STRATA
    }
    for label, counts in per_stratum_counts.items():
        logger.info(
            "  %s: %d / %d (%s)",
            label,
            counts["actual"],
            counts["target"],
            "ok" if counts["actual"] >= counts["target"] else "underflow",
        )

    # Step 7: Write the cohort file, forwarding pool metadata so the cohort
    # captures the pool-build inputs that produced it.
    logger.info("Step 7: Write cohort file to %s", COHORT_PATH)
    # This script lives at scripts/pipeline/, so the repo root is three levels up.
    script_repo = Path(__file__).resolve().parent.parent.parent
    bhc_jsonl = pool_metadata["data_sources"]["bhc_jsonl"]
    bhc_extraction_repo = pool_metadata["data_sources"]["bhc_extraction_repo"]

    write_cohort_file(
        sample,
        path=COHORT_PATH,
        seed=SEED,
        stratification_axes=STRATIFICATION_AXES,
        los_quartile_edges=los_edges,
        cohort_size=cohort_size,
        token_strata=TOKEN_STRATA_META,
        per_stratum_counts=per_stratum_counts,
        bhc_jsonl_path=bhc_jsonl["path"],
        bhc_jsonl_sha256=bhc_jsonl["sha256"],
        bhc_extraction_commit=(bhc_extraction_repo["commit"], bhc_extraction_repo["subject"]),
        sampling_script_commit=git_head(script_repo),
        tokenizer=pool_metadata["tokenizer"],
        pool_file={"path": str(POOL_PATH), "sha256": file_sha256(POOL_PATH)},
    )
    logger.info("Done")


if __name__ == "__main__":
    main()
