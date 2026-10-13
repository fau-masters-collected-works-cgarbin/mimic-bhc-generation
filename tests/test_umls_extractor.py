"""Tests for UmlsExtractor.

These tests stub out spacy and scispacy so they run without the
``umls_concept`` extra installed, via ``install_fake_scispacy`` in
``conftest.py`` (shared with ``test_scorer.py`` and ``test_evaluator.py``,
which also need to construct a stubbed extractor). The fake KB and entity
classes only need to expose the attributes ``UmlsExtractor`` reads
(``types``, ``canonical_name``, ``kb_ents``, ``start_char``, ``end_char``).
The fake NLP also records every call into ``pipe`` so the cache tests can
assert how many unique texts the extractor actually sent through the
pipeline.

Most tests describe a mention as a plain ``(cui, tuis)`` pair, which gets a
deterministic placeholder span (index-based) and score (0.99). The tests
that assert exact span or score values pass a full ``Mention`` instead.
"""

import typing

import pytest

from .conftest import Mention, install_fake_scispacy


class TestExtractConcepts:
    """extract_concepts returns per-text Concept lists with full per-mention data."""

    def test_returns_concept_tuples_with_expected_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_scispacy(
            monkeypatch,
            {"text-a": [("C1", ["T047"]), ("C2", ["T017"])]},
        )
        from bhc_experiment.concepts import Concept
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047", "T017"])
        result = ex.extract_concepts(["text-a"])

        assert list(result.keys()) == ["text-a"]
        concepts = result["text-a"]
        assert len(concepts) == 2
        assert all(isinstance(c, Concept) for c in concepts)
        assert concepts[0] == Concept(
            cui="C1", term="name_for_C1", semtypes=("T047",), score=0.99, start=0, end=1
        )
        assert concepts[1] == Concept(
            cui="C2", term="name_for_C2", semtypes=("T017",), score=0.99, start=10, end=11
        )

    def test_text_with_no_in_scope_concepts_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The single mention's TUI is outside the filter, so it should
        # be dropped. The text is still a key in the result, with [].
        install_fake_scispacy(monkeypatch, {"text-a": [("C9", ["T999"])]})
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        result = ex.extract_concepts(["text-a"])
        assert result == {"text-a": []}

    def test_semtypes_stores_every_kb_type_not_just_the_matching_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # C1's KB entity has T061 (outside the filter) and T047 (inside
        # it). The inclusion filter only needs one matching type, but the
        # stored semtypes is the CUI's full KB type set, sorted for
        # determinism, so both types appear.
        install_fake_scispacy(monkeypatch, {"text-a": [("C1", ["T061", "T047"])]})
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        concept = ex.extract_concepts(["text-a"])["text-a"][0]
        assert concept.semtypes == ("T047", "T061")


class TestConceptSpanAndScore:
    """start/end delimit the exact matched span. score is the top candidate's confidence."""

    def test_start_end_delimit_the_matched_span_in_the_source_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = "history of diabetes and sepsis"
        mentions = [
            Mention("C1", ["T047"], start=11, end=19),  # "diabetes"
            Mention("C2", ["T047"], start=24, end=30),  # "sepsis"
        ]
        install_fake_scispacy(monkeypatch, {text: mentions})
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        concepts = ex.extract_concepts([text])[text]
        assert text[concepts[0].start : concepts[0].end] == "diabetes"
        assert text[concepts[1].start : concepts[1].end] == "sepsis"

    def test_score_is_the_top_candidate_linker_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The score on a Concept is the linker score of the candidate the
        # extractor kept, passed through unchanged.
        install_fake_scispacy(monkeypatch, {"text-a": [Mention("C1", ["T047"], score=0.734)]})
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        concept = ex.extract_concepts(["text-a"])["text-a"][0]
        assert concept.score == pytest.approx(0.734)


class TestExtractCuis:
    """extract_cuis projects extract_concepts to per-text CUI sets."""

    def test_projection_is_a_strict_subset_of_extract_concepts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The same text is asked once via extract_concepts and once via
        # extract_cuis. The CUI set returned by extract_cuis must equal
        # the set of cui fields on the Concept list, with no ordering
        # or per-mention dup info preserved.
        install_fake_scispacy(
            monkeypatch,
            {"text-a": [("C1", ["T047"]), ("C2", ["T047"]), ("C1", ["T047"])]},
        )
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        concepts = ex.extract_concepts(["text-a"])["text-a"]
        cuis = ex.extract_cuis(["text-a"])["text-a"]
        # Three mentions in the concept list (C1 twice, C2 once). The
        # CUI projection drops the duplicate and is a frozenset.
        assert len(concepts) == 3
        assert cuis == frozenset({"C1", "C2"})


class TestCaching:
    """Per-text results survive across calls and across the two extract methods."""

    def test_repeated_text_in_one_call_is_piped_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The fake pipe records every text it receives. The deduped
        # path means the repeated text appears in piped_texts exactly
        # once even though the input list has it twice.
        ent_map = {"text-a": [("C1", ["T047"])], "text-b": [("C2", ["T047"])]}
        fake_nlp_cls = install_fake_scispacy(monkeypatch, ent_map)
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        ex.extract_concepts(["text-a", "text-b", "text-a"])
        assert sorted(fake_nlp_cls.piped_texts) == ["text-a", "text-b"]

    def test_extract_cuis_after_extract_concepts_does_not_re_pipe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # extract_concepts populates the cache on the first call. The
        # follow-up extract_cuis call should hit the cache and add no
        # new entries to piped_texts.
        fake_nlp_cls = install_fake_scispacy(monkeypatch, {"text-a": [("C1", ["T047"])]})
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        ex.extract_concepts(["text-a"])
        before = list(fake_nlp_cls.piped_texts)
        ex.extract_cuis(["text-a"])
        assert fake_nlp_cls.piped_texts == before


class TestExtractEntries:
    """extract_entries returns the full per-text ExtractionResult: matches, rejected, and counts."""

    def test_n_entities_counts_every_ner_proposal_including_dropped_ones(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Four proposals: one in-filter match, one blocklisted, one
        # out-of-filter, one with no linking candidate at all. n_entities
        # counts all four, including the three that never reach matches.
        install_fake_scispacy(
            monkeypatch,
            {
                "text-a": [
                    ("C1", ["T047"]),
                    ("C0032207", ["T196"]),
                    ("C9", ["T999"]),
                    None,
                ]
            },
        )
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        entries = ex.extract_entries(["text-a"])
        assert entries["text-a"].n_entities == 4

    def test_entity_with_no_linking_candidate_increments_n_unlinked_with_no_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_scispacy(monkeypatch, {"text-a": [None]})
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        result = ex.extract_entries(["text-a"])["text-a"]
        assert result.n_unlinked == 1
        assert result.rejected == []
        assert result.matches == []

    def test_blocklisted_cui_appears_in_rejected_with_reason_and_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # C0032207 is the default blocklist's "Pt"/"PT" -> platinum mislink
        # (see LINKING_ARTIFACT_CUIS). Its tuis are irrelevant here: the
        # blocklist check runs before the type filter.
        install_fake_scispacy(
            monkeypatch,
            {"text-a": [Mention("C0032207", ["T196"], start=4, end=6, score=0.83)]},
        )
        from bhc_experiment.concepts import Rejection
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        result = ex.extract_entries(["text-a"])["text-a"]
        assert result.matches == []
        assert result.n_unlinked == 0
        assert result.rejected == [
            Rejection(cui="C0032207", score=0.83, reason="blocklisted", start=4, end=6)
        ]

    def test_out_of_filter_cui_appears_in_rejected_with_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_scispacy(
            monkeypatch,
            {"text-a": [Mention("C9", ["T999"], start=7, end=12, score=0.6)]},
        )
        from bhc_experiment.concepts import Rejection
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        result = ex.extract_entries(["text-a"])["text-a"]
        assert result.matches == []
        assert result.n_unlinked == 0
        assert result.rejected == [
            Rejection(cui="C9", score=0.6, reason="out_of_filter", start=7, end=12)
        ]

    def test_mixed_text_produces_consistent_matches_rejected_and_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # One match, one blocklisted, one out-of-filter, one unlinked, in
        # that entity order.
        install_fake_scispacy(
            monkeypatch,
            {
                "text-a": [
                    Mention("C1", ["T047"], start=0, end=4, score=0.95),
                    Mention("C0032207", ["T196"], start=5, end=7, score=0.83),
                    Mention("C9", ["T999"], start=8, end=13, score=0.6),
                    None,
                ]
            },
        )
        from bhc_experiment.concepts import Concept, Rejection
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        result = ex.extract_entries(["text-a"])["text-a"]

        assert result.n_entities == 4
        assert result.n_unlinked == 1
        assert result.matches == [
            Concept(cui="C1", term="name_for_C1", semtypes=("T047",), score=0.95, start=0, end=4)
        ]
        assert result.rejected == [
            Rejection(cui="C0032207", score=0.83, reason="blocklisted", start=5, end=7),
            Rejection(cui="C9", score=0.6, reason="out_of_filter", start=8, end=13),
        ]


class TestSemanticTypes:
    """semantic_types exposes a CUI's full type set for callers with a bare CUI."""

    def test_returns_every_type_the_kb_carries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A knowledge-base query that answers for a bare CUI with
        # no Concept in hand. Concept.semtypes already has the same full set
        # for a CUI reached through extraction.
        install_fake_scispacy(monkeypatch, {"text-a": [("C1", ["T047", "T061"])]})
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T047"])
        assert ex.semantic_types("C1") == frozenset({"T047", "T061"})


class TestBlockedCuis:
    """Stop-listed CUIs are dropped from extraction. Everything else survives."""

    def test_default_drops_the_platinum_linking_artifact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # C0032207 "platinum" is the element Pt, whose UMLS aliases include the
        # bare symbol "Pt". scispaCy links the two-character token "Pt" (patient)
        # and "PT" (prothrombin time) to it by string similarity, so every match
        # in this corpus is a mislink onto the element. The default
        # stop-list drops it while a real concept (C0010294 creatinine) is kept.
        install_fake_scispacy(
            monkeypatch,
            {"text-a": [("C0032207", ["T196"]), ("C0010294", ["T109"])]},
        )
        from bhc_experiment.umls_extractor import UmlsExtractor

        ex = UmlsExtractor(tuis=["T196", "T109"])
        assert ex.extract_cuis(["text-a"])["text-a"] == frozenset({"C0010294"})


class TestParamsFromExtractor:
    """params_from_extractor: the hand-written allow-list read off a constructed extractor.

    The fingerprint tests inject ``versions`` so they never hit
    ``importlib.metadata`` under the fake scispacy stub, where looking up a
    stubbed package raises ``PackageNotFoundError``.
    """

    _VERSIONS: typing.ClassVar[dict[str, str]] = {
        "en_core_sci_md": "0.5.4",
        "scispacy": "0.6.2",
        "spacy": "3.7.5",
    }

    def test_returns_exactly_the_documented_allow_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_scispacy(monkeypatch, {})
        from bhc_experiment.umls_extractor import (
            LINKING_ARTIFACT_CUIS,
            SCISPACY_EXTRACTOR_VERSION,
            UmlsExtractor,
            params_from_extractor,
        )

        ex = UmlsExtractor(tuis=["T047", "T017"], threshold=0.5, linker_name="umls")
        params = params_from_extractor(ex, versions=self._VERSIONS)

        assert params == {
            "extractor": "scispacy",
            "extractor_version": SCISPACY_EXTRACTOR_VERSION,
            "extractor_params": {
                "spacy_model": "en_core_sci_md",
                "spacy_model_version": "0.5.4",
                "scispacy_version": "0.6.2",
                "spacy_version": "3.7.5",
                "linker_name": "umls",
                "threshold": 0.5,
                "resolve_abbreviations": True,
                "tuis": ["T017", "T047"],
                "linking_artifact_cuis": sorted(LINKING_ARTIFACT_CUIS),
            },
        }

    def test_performance_only_params_are_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # n_process and _PIPE_BATCH_SIZE are performance-only: they never
        # move extraction output, so including them would invalidate the
        # artifact per machine for nothing.
        install_fake_scispacy(monkeypatch, {})
        from bhc_experiment.umls_extractor import UmlsExtractor, params_from_extractor

        ex = UmlsExtractor(tuis=["T047"], n_process=4)
        params = params_from_extractor(ex, versions=self._VERSIONS)

        assert "n_process" not in params["extractor_params"]
        assert "pipe_batch_size" not in params["extractor_params"]
        assert "_PIPE_BATCH_SIZE" not in params["extractor_params"]

    def test_n_process_does_not_change_the_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_scispacy(monkeypatch, {})
        from bhc_experiment.concepts import extractor_config_sha256
        from bhc_experiment.umls_extractor import UmlsExtractor, params_from_extractor

        ex_a = UmlsExtractor(tuis=["T047"], n_process=1)
        ex_b = UmlsExtractor(tuis=["T047"], n_process=8)

        sha_a = extractor_config_sha256(params_from_extractor(ex_a, versions=self._VERSIONS))
        sha_b = extractor_config_sha256(params_from_extractor(ex_b, versions=self._VERSIONS))
        assert sha_a == sha_b

    def test_pipe_batch_size_does_not_change_the_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _PIPE_BATCH_SIZE is a module constant, so varying it takes a
        # monkeypatch on the module between two builds.
        install_fake_scispacy(monkeypatch, {})
        from bhc_experiment import umls_extractor
        from bhc_experiment.concepts import extractor_config_sha256
        from bhc_experiment.umls_extractor import UmlsExtractor, params_from_extractor

        monkeypatch.setattr(umls_extractor, "_PIPE_BATCH_SIZE", 16)
        ex_a = UmlsExtractor(tuis=["T047"])
        sha_a = extractor_config_sha256(params_from_extractor(ex_a, versions=self._VERSIONS))

        monkeypatch.setattr(umls_extractor, "_PIPE_BATCH_SIZE", 999)
        ex_b = UmlsExtractor(tuis=["T047"])
        sha_b = extractor_config_sha256(params_from_extractor(ex_b, versions=self._VERSIONS))

        assert sha_a == sha_b

    def test_reordering_the_tui_tuple_does_not_change_the_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _tui_set = frozenset(tuis), so construction order has no
        # meaning. The allow-list re-sorts it, so the stem must match.
        install_fake_scispacy(monkeypatch, {})
        from bhc_experiment.concepts import config_stem
        from bhc_experiment.umls_extractor import UmlsExtractor, params_from_extractor

        ex_a = UmlsExtractor(tuis=["T047", "T017"])
        ex_b = UmlsExtractor(tuis=["T017", "T047"])

        stem_a = config_stem(params_from_extractor(ex_a, versions=self._VERSIONS))
        stem_b = config_stem(params_from_extractor(ex_b, versions=self._VERSIONS))
        assert stem_a == stem_b

    def test_two_instances_from_the_same_config_produce_the_same_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_scispacy(monkeypatch, {})
        from bhc_experiment.concepts import extractor_config_sha256
        from bhc_experiment.umls_extractor import UmlsExtractor, params_from_extractor

        def build() -> UmlsExtractor:
            return UmlsExtractor(
                spacy_model="en_core_sci_md", linker_name="umls", tuis=["T047"], threshold=0.7
            )

        sha_a = extractor_config_sha256(params_from_extractor(build(), versions=self._VERSIONS))
        sha_b = extractor_config_sha256(params_from_extractor(build(), versions=self._VERSIONS))
        assert sha_a == sha_b

    def test_versions_none_default_looks_up_via_importlib_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When versions=None (the production default), package versions come from
        # importlib.metadata.version.
        install_fake_scispacy(monkeypatch, {})
        from bhc_experiment.umls_extractor import UmlsExtractor, params_from_extractor

        def fake_version(pkg: str) -> str:
            return {
                "en_core_sci_md": "0.5.4",
                "scispacy": "0.6.2",
                "spacy": "3.7.5",
            }[pkg]

        monkeypatch.setattr("importlib.metadata.version", fake_version)
        ex = UmlsExtractor(tuis=["T047"])
        params = params_from_extractor(ex)

        assert params["extractor_params"]["spacy_model_version"] == "0.5.4"
        assert params["extractor_params"]["scispacy_version"] == "0.6.2"
        assert params["extractor_params"]["spacy_version"] == "3.7.5"


class TestParamsFromSettings:
    """params_from_settings: the load-free core params_from_extractor delegates to."""

    _VERSIONS: typing.ClassVar[dict[str, str]] = {
        "en_core_sci_md": "0.5.4",
        "scispacy": "0.6.2",
        "spacy": "3.7.5",
    }

    def test_matches_params_from_extractor_for_the_same_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_scispacy(monkeypatch, {})
        from bhc_experiment.umls_extractor import (
            UmlsExtractor,
            params_from_extractor,
            params_from_settings,
        )

        ex = UmlsExtractor(
            spacy_model="en_core_sci_md", linker_name="umls", tuis=["T047", "T017"], threshold=0.5
        )
        from_extractor = params_from_extractor(ex, versions=self._VERSIONS)
        from_settings = params_from_settings(
            "en_core_sci_md", "umls", 0.5, ["T047", "T017"], versions=self._VERSIONS
        )

        assert from_settings == from_extractor
