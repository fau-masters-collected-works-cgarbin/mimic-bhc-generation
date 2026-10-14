"""Select patients for the smoke test cohort.

Why: the smoke test verifies end-to-end pipeline operation (data loading,
prompt building, LLM call, evaluation, result storage). It needs a small,
deterministic subset of the experiment cohort. The selection does not need
stratification, since its only job is to verify the pipeline runs.

Why the 4K stratum: on-device models run slowly on long contexts, and the
smoke test does not need long context to exercise the pipeline. Restricting
to the smallest stratum keeps the smoke run fast while still covering the
full code path.

Questions answered:
- Which admissions form the smoke test cohort?

Output: cohorts/smoke_cohort.json
"""

import argparse
import json
import logging
from dataclasses import asdict

from bhc_experiment.config import EXPERIMENT_COHORT_FILE, SMOKE_COHORT_FILE
from mimic_data_prep.cohort import load_cohort
from mimic_data_prep.mimic import COHORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COHORT_PATH = COHORTS_DIR / EXPERIMENT_COHORT_FILE
OUTPUT_PATH = COHORTS_DIR / SMOKE_COHORT_FILE
SAMPLE_SIZE = 3
SMOKE_STRATUM = "4K"


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    cohort = load_cohort(COHORT_PATH)
    logger.info("Loaded %d admissions from %s", len(cohort.admissions), COHORT_PATH)

    # Sort by hadm_id so the selection is deterministic without a seed.
    in_stratum = [a for a in cohort.admissions if a.token_stratum == SMOKE_STRATUM]
    selected = sorted(in_stratum, key=lambda a: a.hadm_id)[:SAMPLE_SIZE]

    for a in selected:
        logger.info(
            "  HADM_ID %d (%s, %s, %d days)",
            a.hadm_id,
            a.bhc_category,
            a.los_quartile,
            a.los_days,
        )

    output = {
        "metadata": {
            "description": "Smoke test cohort (pipeline verification)",
            "source_cohort_file": COHORT_PATH.name,
            "sample_size": len(selected),
            "selection": f"first N admissions by hadm_id within the {SMOKE_STRATUM} token stratum",
            "script": "scripts/pipeline/select_smoke_cohort.py",
        },
        "admissions": [asdict(a) for a in selected],
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    logger.info("Wrote %d admissions to %s", len(selected), OUTPUT_PATH)


if __name__ == "__main__":
    main()
