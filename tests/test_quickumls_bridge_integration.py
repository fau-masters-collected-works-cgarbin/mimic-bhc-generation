"""QuickUmlsExtractor against the real index and the real bridge subprocess.

The unit tests fake the bridge, which cannot check the one thing that decides
whether our numbers are MEDCON's: that parsing QuickUMLS's nested span groups
our way reaches the same CUI set Yim's ``get_matches`` reaches. That needs a
licensed index, so these skip wherever one is absent.
"""

import json
import subprocess

import pytest

from bhc_experiment.concept_spaces import concept_extractor, concept_params
from bhc_experiment.concepts import config_stem
from bhc_experiment.config import CONCEPT_SPACES
from bhc_experiment.quickumls_extractor import (
    BRIDGE_PYTHON,
    BRIDGE_REPO,
    INSTALL_METADATA_FILENAME,
    UMLS_INDEX,
    params_from_settings,
)

pytestmark = [pytest.mark.quickumls_index, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _require_install() -> None:
    """Skip when the QuickUMLS index or the bridge environment is absent."""
    if not (UMLS_INDEX / INSTALL_METADATA_FILENAME).exists():
        pytest.skip(f"no QuickUMLS index at {UMLS_INDEX}")
    if not BRIDGE_PYTHON.exists():
        pytest.skip(f"no bridge environment at {BRIDGE_PYTHON}")


@pytest.fixture(scope="module")
def extractor():
    """Built the way the pipeline builds it, from the named concept space.

    Module-scoped for the per-text cache, so the shared TEXTS below cost one
    bridge call across every test that uses them.
    """
    return concept_extractor("quickumls")


# Chosen to exercise the kinds of text the cohort actually contains: prose with
# multi-word concepts, a bare lab LABEL, a bare drug DRUG, an abbreviation
# short enough for min_match_length to drop, and text with no concept at all.
TEXTS = [
    "Patient with myocardial infarction and diabetes mellitus.",
    (
        "The patient was started on aspirin 81 mg daily for secondary prevention of "
        "coronary artery disease."
    ),
    "Hematocrit",
    (
        "Serum creatinine elevated at 2.1, consistent with acute kidney injury on "
        "chronic kidney disease stage III."
    ),
    "metoprolol tartrate",
    "PT",
    "",
]


def _yim_cuis(texts: list[str]) -> list[list[str]]:
    """Run Yim's ``get_matches`` unmodified in the bridge environment.

    ``UMLS_evaluation.py`` builds its own matcher at import time from
    ``PROJECT_DIR/resources/des``, which setup symlinks to the real index, so
    the index reaches it through that link. No argument passes it in.
    """
    script = (
        "import json,sys\n"
        "from UMLS_evaluation import get_matches\n"
        "for line in sys.stdin:\n"
        "    text = json.loads(line)['text']\n"
        "    print(json.dumps(sorted(set(get_matches(text)[1]))), flush=True)\n"
    )
    bootstrap = (
        f"import sys; sys.path.insert(0, {str(BRIDGE_REPO / 'aci_bench' / 'evaluation')!r})\n"
    )
    stdin = "".join(json.dumps({"text": t}) + "\n" for t in texts)
    result = subprocess.run(  # noqa: S603
        [str(BRIDGE_PYTHON), "-c", bootstrap + script],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def test_bridge_extract_agrees_with_yim_get_matches(extractor) -> None:
    # If our extraction and Yim's disagree, MEDCON and the coverage ceiling
    # stop describing the same concept space.
    ours = extractor.extract_cuis(TEXTS)

    assert [sorted(ours[t]) for t in TEXTS] == _yim_cuis(TEXTS)


def test_fingerprint_records_yims_settings_and_the_umls_release() -> None:
    # Every QuickUMLS setting comes from Yim's code.
    assert set(CONCEPT_SPACES["quickumls"]) == {"name", "extractor"}

    params = params_from_settings()["extractor_params"]

    assert params["umls_release"] == "2022AA"
    assert params["threshold"] == 1
    assert params["window"] == 5
    assert len(params["accepted_semtypes"]) == 66


def test_names_the_artifact_preparation_wrote() -> None:
    # Another stem points every reader at a concept artifact that was never built.
    assert config_stem(concept_params("quickumls")) == "quickumls-6be08e826feb"
