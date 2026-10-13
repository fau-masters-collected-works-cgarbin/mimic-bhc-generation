"""Build a named concept space's extractor and the parameters that fingerprint it.

``config.CONCEPT_SPACES`` names the spaces. The fingerprint names each space's
concept artifact. ``concept_params`` derives it from the settings without loading
the extractor, so it must equal what the built extractor reports.
"""

from bhc_experiment.concepts import ConceptExtractor
from bhc_experiment.config import CONCEPT_SPACES
from bhc_experiment.quickumls_extractor import QuickUmlsExtractor
from bhc_experiment.quickumls_extractor import params_from_settings as quickumls_params
from bhc_experiment.umls_extractor import MEDCON_DEFAULT_TUIS, UmlsExtractor, params_from_settings


def _scispacy_settings(space: str) -> dict:
    """The space's settings as ``UmlsExtractor`` keyword arguments."""
    return {
        key: value
        for key, value in CONCEPT_SPACES[space].items()
        if key not in {"name", "extractor"}
    }


def concept_extractor(space: str, *, n_process: int = 1) -> ConceptExtractor:
    """Build the extractor of a named concept space.

    ``n_process`` sets scispaCy's worker count, which does not change extraction.
    """
    if CONCEPT_SPACES[space]["extractor"] == "quickumls":
        return QuickUmlsExtractor()
    return UmlsExtractor(**_scispacy_settings(space), n_process=n_process)


def concept_params(space: str) -> dict:
    """The fingerprinted parameters of a named concept space, without building its extractor."""
    if CONCEPT_SPACES[space]["extractor"] == "quickumls":
        return quickumls_params()
    settings = _scispacy_settings(space)
    return params_from_settings(
        settings["spacy_model"],
        settings["linker_name"],
        settings["threshold"],
        frozenset(MEDCON_DEFAULT_TUIS) | frozenset(settings.get("extra_tuis", ())),
    )
