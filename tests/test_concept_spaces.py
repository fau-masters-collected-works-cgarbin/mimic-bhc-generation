"""Tests for bhc_experiment.concept_spaces."""

import typing

import pytest

from bhc_experiment.concept_spaces import concept_extractor, concept_params
from bhc_experiment.concepts import config_stem
from bhc_experiment.config import CONCEPT_SPACES
from bhc_experiment.quickumls_extractor import QuickUmlsExtractor
from bhc_experiment.umls_extractor import MEDCON_DEFAULT_TUIS, UmlsExtractor, params_from_extractor

from .conftest import install_fake_scispacy

SCISPACY_SPACES = [
    space for space, entry in CONCEPT_SPACES.items() if entry["extractor"] == "scispacy"
]


class TestConceptParams:
    _VERSIONS: typing.ClassVar[dict[str, str]] = {
        "en_core_sci_md": "0.5.4",
        "scispacy": "0.6.2",
        "spacy": "3.7.5",
    }

    @pytest.mark.parametrize("space", SCISPACY_SPACES)
    def test_match_what_the_built_extractor_reports(self, monkeypatch, space) -> None:
        # Both sides read the installed package versions themselves.
        install_fake_scispacy(monkeypatch, {})

        assert concept_params(space) == params_from_extractor(concept_extractor(space))

    @pytest.mark.parametrize(
        ("space", "stem"),
        [
            ("scispacy", "scispacy-b42bfe475bbb"),
            ("scispacy-plus-procedures", "scispacy-6a574984e5c8"),
        ],
    )
    def test_name_the_artifacts_preparation_wrote(self, space, stem) -> None:
        # Another stem points every reader at a concept artifact that was never built.
        assert config_stem(concept_params(space)) == stem

    def test_builds_no_extractor(self, monkeypatch) -> None:
        def _boom(*_args, **_kwargs):
            msg = "concept_params must not construct UmlsExtractor"
            raise AssertionError(msg)

        monkeypatch.setattr(UmlsExtractor, "__init__", _boom)
        monkeypatch.setattr("importlib.metadata.version", lambda pkg: self._VERSIONS[pkg])

        assert concept_params("scispacy")["extractor_params"]["tuis"] == sorted(MEDCON_DEFAULT_TUIS)


class TestConceptExtractor:
    def test_quickumls_space_builds_the_quickumls_extractor(self) -> None:
        assert isinstance(concept_extractor("quickumls"), QuickUmlsExtractor)


@pytest.mark.slow
def test_params_match_a_real_built_extractor() -> None:
    # A real UmlsExtractor construction takes about 44s.
    assert concept_params("scispacy") == params_from_extractor(concept_extractor("scispacy"))
