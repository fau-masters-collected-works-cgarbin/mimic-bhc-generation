"""Tests for bhc_experiment.metrics."""

import pytest

from bhc_experiment import config, metrics

CONSTANT_NAMES = (
    "ROUGE1_F1",
    "ROUGE2_F1",
    "ROUGEL_F1",
    "DISTILBERT_F1",
    "DISTILBERT_RESCALED_F1",
    "DEBERTA_F1",
    "SAPBERT_F1",
    "SCISPACY_CLINICAL_PRECISION",
    "SCISPACY_CLINICAL_RECALL",
    "SCISPACY_CLINICAL_F1",
    "PRIMARY_METRIC",
)


def _declared_metric_names() -> set[str]:
    """Every metric `name` the scored metric set and the concept spaces declare."""
    names = {body["name"] for entry in config.METRICS for body in entry.values() if "name" in body}
    names |= {space["name"] for space in config.CONCEPT_SPACES.values()}
    return names


@pytest.mark.parametrize("const_name", CONSTANT_NAMES)
def test_registry_constant_names_a_declared_metric(const_name: str) -> None:
    stem, submetric = getattr(metrics, const_name).split("/")
    assert stem in _declared_metric_names()
    assert submetric in {"precision", "recall", "f1"}
