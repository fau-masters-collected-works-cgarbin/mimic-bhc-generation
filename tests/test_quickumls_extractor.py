"""QuickUmlsExtractor tests, run against a faked bridge.

The bridge is a subprocess with a line-oriented JSONL contract, so a fake that
speaks that contract exercises everything on this side of it. The one thing a
fake cannot check, that our span parsing agrees with Yim's ``get_matches``, is
an integration test that needs the real index.
"""

import json

import pytest

from bhc_experiment import quickumls_extractor
from bhc_experiment.quickumls_extractor import QuickUmlsExtractor, params_from_settings

# What matcher.get_info() reports, at Yim's settings.
SETTINGS = {"threshold": 1, "window": 5, "similarity_name": "jaccard", "min_match_length": 3}


def fake_bridge(spans_by_text: dict[str, list], settings: dict | None = None):
    """Replace run_extract with a canned reply, recording what it was asked for.

    Patching at the subprocess boundary means everything above it, including
    the span parsing, runs for real.
    """
    calls: list[list[str]] = []

    def _run(texts: list[str]) -> tuple[dict, dict[str, list]]:
        calls.append(list(texts))
        reported = SETTINGS if settings is None else settings
        return reported, {t: spans_by_text.get(t, []) for t in texts}

    _run.calls = calls
    return _run


def span(cui: str, term: str, semtypes: list[str], start: int, end: int, similarity: float = 1.0):
    return {
        "cui": cui,
        "term": term,
        "semtypes": semtypes,
        "similarity": similarity,
        "start": start,
        "end": end,
    }


@pytest.fixture
def extractor(monkeypatch):
    """An extractor over one text with two spans, the second with two candidate CUIs."""
    bridge = fake_bridge(
        {
            "chest pain and MI": [
                [span("C0008031", "chest pain", ["T184"], 0, 10)],
                [
                    span("C0027051", "myocardial infarction", ["T047"], 15, 17),
                    span("C1550655", "MI", ["T031"], 15, 17, similarity=0.9),
                ],
            ],
            "nothing here": [],
        }
    )
    monkeypatch.setattr(quickumls_extractor, "run_extract", bridge)
    return QuickUmlsExtractor(), bridge


class TestExtraction:
    def test_extract_cuis_keeps_every_candidate_in_a_span_group(self, extractor) -> None:
        # Yim's get_matches appends every m['cui'] in every group, so a
        # top-candidate-only rule here would score a different concept set.
        ext, _ = extractor
        assert ext.extract_cuis(["chest pain and MI"]) == {
            "chest pain and MI": frozenset({"C0008031", "C0027051", "C1550655"})
        }

    def test_extract_concepts_returns_one_concept_per_candidate(self, extractor) -> None:
        ext, _ = extractor
        concepts = ext.extract_concepts(["chest pain and MI"])["chest pain and MI"]
        assert [(c.cui, c.term, c.semtypes, c.start, c.end) for c in concepts] == [
            ("C0008031", "chest pain", ("T184",), 0, 10),
            ("C0027051", "myocardial infarction", ("T047",), 15, 17),
            ("C1550655", "MI", ("T031",), 15, 17),
        ]

    def test_empty_text_gets_an_entry(self, extractor) -> None:
        # Every input text appears in the result, even one with no concepts.
        ext, _ = extractor
        assert ext.extract_cuis(["nothing here"]) == {"nothing here": frozenset()}

    def test_extract_entries_counts_span_groups_and_reports_no_unlinked(self, extractor) -> None:
        # A dictionary matcher proposes a span only when it already matched, so
        # "proposals that failed to link" is zero by construction.
        ext, _ = extractor
        entry = ext.extract_entries(["chest pain and MI"])["chest pain and MI"]
        assert entry.n_entities == 2
        assert entry.n_unlinked == 0
        assert entry.rejected == []

    def test_repeated_texts_reach_the_bridge_once(self, extractor) -> None:
        ext, bridge = extractor
        ext.extract_cuis(["chest pain and MI", "chest pain and MI", "nothing here"])
        ext.extract_cuis(["chest pain and MI"])

        # One subprocess, with each distinct text once.
        assert bridge.calls == [["chest pain and MI", "nothing here"]]

    def test_applies_no_blocklist(self, monkeypatch) -> None:
        # LINKING_ARTIFACT_CUIS is a scispaCy repair. Suppressing a CUI the
        # reference implementation emits would be a deviation from MEDCON.
        from bhc_experiment.umls_extractor import LINKING_ARTIFACT_CUIS

        blocked = min(LINKING_ARTIFACT_CUIS)
        monkeypatch.setattr(
            quickumls_extractor,
            "run_extract",
            fake_bridge({"t": [[span(blocked, "blocked", ["T031"], 0, 1)]]}),
        )

        assert QuickUmlsExtractor().extract_cuis(["t"]) == {"t": frozenset({blocked})}


class TestVersionHandshake:
    def test_mismatch_raises(self) -> None:
        # The pins decide what the matcher computes, so an upgrade in the
        # sibling repo would move scores under an unchanged fingerprint.
        drifted = dict(quickumls_extractor.EXPECTED_BRIDGE_VERSIONS, quickumls_version="1.4.3")

        with pytest.raises(RuntimeError, match="quickumls_version"):
            quickumls_extractor._verify_bridge_versions(drifted)  # noqa: SLF001


INSTALL = {"umls_release": "2022AA", "mrconso_sha256": "abc", "mrsty_sha256": "def"}


class TestFingerprint:
    @pytest.fixture
    def index_dir(self, tmp_path, monkeypatch):
        (tmp_path / "install.json").write_text(json.dumps(INSTALL))
        monkeypatch.setattr(quickumls_extractor, "run_extract", fake_bridge({}))
        return tmp_path

    def test_records_the_matchers_settings_and_the_umls_release(self, index_dir) -> None:
        # The settings come off the live matcher, so the fingerprint records what
        # actually matched. A config could declare something else.
        params = params_from_settings(index=index_dir)

        assert params["extractor"] == "quickumls"
        assert params["extractor_params"]["threshold"] == 1
        assert params["extractor_params"]["window"] == 5
        assert params["extractor_params"]["umls_release"] == "2022AA"

    def test_index_path_is_not_fingerprinted(self, index_dir) -> None:
        # The index location is machine-specific. Hashing it would invalidate
        # every artifact on a second machine that extracted identical concepts.
        params = params_from_settings(index=index_dir)

        assert "index" not in json.dumps(params)

    def test_incomplete_install_metadata_raises(self, tmp_path, monkeypatch) -> None:
        # A missing digest would otherwise hash as null, making two different
        # UMLS versions share one fingerprint.
        (tmp_path / "install.json").write_text(json.dumps({"umls_release": "2022AA"}))
        monkeypatch.setattr(quickumls_extractor, "run_extract", fake_bridge({}))

        with pytest.raises(KeyError, match="mrconso_sha256"):
            params_from_settings(index=tmp_path)
