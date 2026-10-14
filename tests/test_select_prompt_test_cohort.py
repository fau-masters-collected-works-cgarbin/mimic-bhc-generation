"""Tests for scripts/pipeline/select_prompt_test_cohort.py."""

import random
import sys
from pathlib import Path

from mimic_data_prep.cohort import Admission
from mimic_data_prep.sampling import TOKEN_STRATA

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "pipeline"
sys.path.insert(0, str(SCRIPTS_DIR))

import select_prompt_test_cohort  # noqa: E402


def _cohort(admission_dict) -> list[Admission]:
    """Every token stratum with two admissions of each format, as in the real cohort."""
    admissions = []
    hadm_id = 100
    for stratum in TOKEN_STRATA:
        for category in ("narrative", "narrative", "domain_organized", "domain_organized"):
            record = admission_dict(hadm_id, token_stratum=stratum, bhc_category=category)
            admissions.append(Admission.from_dict(record))
            hadm_id += 1
    return admissions


def test_picks_one_of_each_format_per_stratum(admission_dict) -> None:
    selected = select_prompt_test_cohort.select_prompt_test(
        _cohort(admission_dict), random.Random(0)
    )
    expected = []
    for stratum in TOKEN_STRATA:
        expected.extend([(stratum, "domain_organized"), (stratum, "narrative")])
    assert sorted((a.token_stratum, a.bhc_category) for a in selected) == sorted(expected)


def test_selection_does_not_depend_on_input_order(admission_dict) -> None:
    cohort = _cohort(admission_dict)
    shuffled = list(cohort)
    random.Random(1).shuffle(shuffled)
    first = select_prompt_test_cohort.select_prompt_test(cohort, random.Random(7))
    second = select_prompt_test_cohort.select_prompt_test(shuffled, random.Random(7))
    assert [a.hadm_id for a in first] == [a.hadm_id for a in second]
