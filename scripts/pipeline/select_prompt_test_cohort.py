"""Select admissions for the prompt test cohort.

Why: the revised prompt wording (a 250 to 600 word length target, among
other changes) must keep output length and format adherence on every input
size before the full 116-admission run spends days of generation on it.
Wording that meets the length target on a short record can overrun or
undershoot it on a long one. The format instruction differs for narrative
and domain-organized references. The prompt test takes one admission of each
format from every token stratum, so a wording problem shows up on all three
models before the full run.

The prompt test is drawn from the scored experiment cohort, because only those
admissions have earlier generations to compare against. It gates on length
and format adherence only (see docs/specs/prompt-builder-design.md).

Questions answered:
- Which admissions form the prompt test cohort?

Output: cohorts/prompt_test_cohort.json
"""

import argparse
import json
import logging
import random
from collections import Counter
from dataclasses import asdict

from bhc_experiment.config import EXPERIMENT_COHORT_FILE, PROMPT_TEST_COHORT_FILE
from mimic_data_prep.cohort import Admission, load_cohort
from mimic_data_prep.mimic import COHORTS_DIR
from mimic_data_prep.sampling import TOKEN_STRATA

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COHORT_PATH = COHORTS_DIR / EXPERIMENT_COHORT_FILE
OUTPUT_PATH = COHORTS_DIR / PROMPT_TEST_COHORT_FILE
SEED = 42
NARRATIVE_CATEGORY = "narrative"


def select_prompt_test(admissions: list[Admission], rng: random.Random) -> list[Admission]:
    """Pick one narrative and one domain-organized admission per token stratum.

    Every stratum of the experiment cohort has both formats.
    """
    selected: list[Admission] = []
    for stratum in TOKEN_STRATA:
        in_stratum = [a for a in admissions if a.token_stratum == stratum]
        # Sorting first makes the draw depend only on the seed.
        in_stratum.sort(key=lambda a: a.hadm_id)
        narrative = [a for a in in_stratum if a.bhc_category == NARRATIVE_CATEGORY]
        domain = [a for a in in_stratum if a.bhc_category != NARRATIVE_CATEGORY]
        selected.append(rng.choice(narrative))
        selected.append(rng.choice(domain))
    return selected


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    cohort = load_cohort(COHORT_PATH)
    logger.info("Loaded %d admissions from %s", len(cohort.admissions), COHORT_PATH)

    selected = select_prompt_test(cohort.admissions, random.Random(SEED))

    # Aggregate counts only. Admission IDs stay out of the console (DUA).
    counts = Counter((a.token_stratum, a.bhc_category) for a in selected)
    for stratum in TOKEN_STRATA:
        for category in (NARRATIVE_CATEGORY, "domain_organized"):
            logger.info("  %-5s %-16s %d", stratum, category, counts[(stratum, category)])

    output = {
        "metadata": {
            "description": "Prompt test cohort (prompt length and format adherence check)",
            "source_cohort_file": COHORT_PATH.name,
            "sample_size": len(selected),
            "seed": SEED,
            "selection": (
                "one narrative and one domain-organized BHC per token stratum, "
                "drawn with the seed after sorting by hadm_id"
            ),
            "script": "scripts/pipeline/select_prompt_test_cohort.py",
        },
        "admissions": [asdict(a) for a in selected],
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    logger.info("Wrote %d admissions to %s", len(selected), OUTPUT_PATH)


if __name__ == "__main__":
    main()
