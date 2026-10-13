"""Smoke test for the UMLS concept-overlap metric using the real scispaCy linker.

Sanity check after the UMLS install (see docs/setup.md).
The first invocation downloads the ~1 GB UMLS linker bundle (cached under
~/.scispacy/).

Three pairs cover the metric's behavior space:
  - Identical pair: F1 = 1.0 (full CUI overlap)
  - Partial-overlap pair: 0 < F1 < 1, precision != recall (candidate is a
    subset of the reference's concepts, so recall drops and precision stays high)
  - Disjoint pair: F1 = 0.0 (no shared CUIs)

Values outside those ranges usually mean the threshold is too high or the TUI
filter is too narrow. ``SCISPACY_CLINICAL`` in src/bhc_experiment/config.py has
the configuration.
"""

import argparse

from bhc_experiment.config import MAIN_CONCEPT_SPACE
from bhc_experiment.evaluator import build_scorers

PAIRS = [
    ("Patient admitted with pneumonia.", "Patient admitted with pneumonia."),
    (
        "Patient admitted with pneumonia.",
        "Patient admitted with pneumonia and treated with ceftriaxone.",
    ),
    ("The weather is nice.", "Patient admitted with pneumonia and treated with ceftriaxone."),
]


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print("Running UMLS concept smoke test (takes a few seconds)\n")
    (concept_metric,) = build_scorers([{"umls_concept": {"space": MAIN_CONCEPT_SPACE}}]).values()
    cands = [generated for generated, _ in PAIRS]
    refs = [reference for _, reference in PAIRS]
    for (generated, reference), scores in zip(PAIRS, concept_metric(cands, refs), strict=True):
        print("---")
        print(generated, "||", reference, "->", scores)


if __name__ == "__main__":
    main()
