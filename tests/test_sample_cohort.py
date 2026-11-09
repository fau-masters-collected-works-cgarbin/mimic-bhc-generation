"""Tests for scripts/pipeline/sample_cohort.py."""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "pipeline"
sys.path.insert(0, str(SCRIPTS_DIR))

import sample_cohort  # noqa: E402

from mimic_data_prep.sampling import TOKEN_STRATA  # noqa: E402


def test_samples_per_stratum_matches_allocation_contract() -> None:
    """SAMPLES_PER_STRATUM matches the allocation contract.

    Three properties together form the contract: the dict covers every
    assigned stratum, every value is a positive int, and the sum
    matches the documented cohort size of 132. Drift in any of them
    silently changes the cohort design relative to the documented
    allocation.
    """
    targets = sample_cohort.SAMPLES_PER_STRATUM
    assert set(targets.keys()) == set(TOKEN_STRATA)
    assert all(isinstance(n, int) and n > 0 for n in targets.values())
    assert sum(targets.values()) == 132
