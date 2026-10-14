"""Tests for bhc_experiment.concept_artifact.

The tests that need an extractor inject a hand-rolled one that implements
``extract_entries`` and counts its own calls, alongside a ``_fake_concept_text``
stand-in. Bundles and records use
obviously-synthetic identifiers (HADM_ID=100, SUBJECT_ID=10, small ROW_IDs),
never values that could collide with real MIMIC-III data.
"""

import json

import pytest

from bhc_experiment.concept_artifact import (
    SCHEMA_VERSION,
    _match_dict,
    _rejection_dict,
    _stored_hashes,
    _verify_complete,
    build_artifact,
    collect_texts,
    serialize_entry,
    write_meta,
)
from bhc_experiment.concepts import Concept, ExtractionResult, Rejection, text_sha256

from .conftest import make_bundle, make_lab, make_note, make_prescription


def _fake_concept_text(record: object) -> str:
    """Test double for the not-yet-built concept_text: a distinct string per record."""
    return f"{type(record).__name__}:{record.ROW_ID}"


def _result_for(_text: str, *, cui: str = "C0000001") -> ExtractionResult:
    """A trivial one-match ExtractionResult. ``cui`` is all that varies across calls."""
    return ExtractionResult(
        matches=[Concept(cui=cui, term="Term", semtypes=("T047",), score=0.9, start=0, end=1)],
        rejected=[],
        n_entities=1,
        n_unlinked=0,
    )


class FakeExtractor:
    """Hand-rolled ConceptExtractor test double: canned results, self-counted calls."""

    def __init__(self, results_by_text: dict[str, ExtractionResult]) -> None:
        self._results_by_text = results_by_text
        self.calls: list[list[str]] = []

    def extract_entries(self, texts: list[str]) -> dict[str, ExtractionResult]:
        self.calls.append(list(texts))
        return {t: self._results_by_text[t] for t in texts}


class TestMatchDict:
    """A match dict's key set is exactly {cui, term, semtypes, score, start, end}."""

    def test_key_set_excludes_ngram_and_matched_text(self) -> None:
        concept = Concept(
            cui="C1", term="Term", semtypes=("T047", "T033"), score=0.9, start=0, end=4
        )

        d = _match_dict(concept)

        assert set(d.keys()) == {"cui", "term", "semtypes", "score", "start", "end"}
        assert "ngram" not in d
        assert "text" not in d
        assert d["semtypes"] == ["T047", "T033"]  # serialized as a JSON list


class TestRejectionDict:
    """A rejection dict's key set is exactly {cui, score, reason, start, end}."""

    def test_key_set_excludes_ngram_and_matched_text(self) -> None:
        rejection = Rejection(cui="C1", score=0.5, reason="blocklisted", start=0, end=4)

        d = _rejection_dict(rejection)

        assert set(d.keys()) == {"cui", "score", "reason", "start", "end"}
        assert "ngram" not in d
        assert "text" not in d


class TestSerializeEntry:
    """serialize_entry never has source text, only its hash and length."""

    def test_rejected_out_of_filter_and_unlinked_entity_both_recorded(self) -> None:
        text = "synthetic note text used only to compute a hash"
        result = ExtractionResult(
            matches=[],
            rejected=[Rejection(cui="C9999999", score=0.5, reason="out_of_filter", start=2, end=5)],
            n_entities=2,
            n_unlinked=1,
        )

        entry = serialize_entry(text, result)

        assert entry["rejected"] == [
            {"cui": "C9999999", "score": 0.5, "reason": "out_of_filter", "start": 2, "end": 5}
        ]
        # The unlinked entity increments n_unlinked without producing a
        # rejection: it never had a CUI, so there is nothing to record
        # beyond the count.
        assert entry["n_unlinked"] == 1
        assert entry["matches"] == []

    def test_entry_never_carries_the_source_text_only_hash_and_length(self) -> None:
        text = "sensitive-patient-identifiable-string-must-not-leak"
        result = ExtractionResult(matches=[], rejected=[], n_entities=0, n_unlinked=0)

        entry = serialize_entry(text, result)

        assert entry["text_sha256"] == text_sha256(text)
        assert entry["text_length"] == len(text)
        serialized = json.dumps(entry)
        assert text not in serialized


class TestCollectTexts:
    """collect_texts pins the no-prompt-filters rule: only discharge summaries are excluded."""

    def test_skips_discharge_summary_keeps_pre_admission_note_and_non_abnormal_lab(self) -> None:
        ds_note = make_note(1, is_discharge_summary=True)
        pre_note = make_note(2, is_pre_admission=True)
        # FLAG is None (not "abnormal"): a prompt filter would drop this,
        # collect_texts must not.
        normal_lab = make_lab(FLAG=None)
        bundle = make_bundle([ds_note, pre_note], labs=[normal_lab])

        texts = collect_texts([bundle], _fake_concept_text)

        assert _fake_concept_text(pre_note) in texts
        assert _fake_concept_text(normal_lab) in texts
        assert _fake_concept_text(ds_note) not in texts

    def test_ground_truth_bhc_added_raw_not_through_concept_text(self) -> None:
        bundle = make_bundle(reference="the reference BHC, added as-is")

        texts = collect_texts([bundle], _fake_concept_text)

        assert "the reference BHC, added as-is" in texts

    def test_prescriptions_included(self) -> None:
        rx = make_prescription()
        bundle = make_bundle(prescriptions=[rx])

        texts = collect_texts([bundle], _fake_concept_text)

        assert _fake_concept_text(rx) in texts


class TestVerifyComplete:
    """_verify_complete raises on any gap, naming only counts and the path."""

    def test_raises_when_a_text_is_missing_naming_counts_and_path_never_text(
        self, tmp_path
    ) -> None:
        jsonl_path = tmp_path / "artifact.jsonl"
        present_text = "present text"
        missing_text = "sensitive-patient-text-must-not-leak"
        jsonl_path.write_text(
            json.dumps(serialize_entry(present_text, _result_for(present_text))) + "\n"
        )

        with pytest.raises(RuntimeError) as exc_info:
            _verify_complete([present_text, missing_text], jsonl_path)

        message = str(exc_info.value)
        assert missing_text not in message
        assert "1" in message
        assert str(jsonl_path) in message


class TestBuildArtifact:
    """build_artifact: extract the misses, append, verify, resume on a second call."""

    def test_multichunk_crash_and_resume_preserves_first_chunk_on_disk(self, tmp_path) -> None:
        """Crash mid-multichunk: first chunk on disk, resume extracts only missing."""
        # Generate 600 unique texts so we exceed the 500-text chunk size.
        texts = [f"synthetic text {i:04d}" for i in range(600)]

        # First extractor: succeeds on call #1, raises on call #2.
        class CrashingExtractor:
            def __init__(self):
                self.call_count = 0

            def extract_entries(self, chunk_texts: list[str]) -> dict[str, ExtractionResult]:
                self.call_count += 1
                if self.call_count == 2:
                    msg = "simulated crash on second chunk"
                    raise RuntimeError(msg)
                return {t: _result_for(t) for t in chunk_texts}

        first_extractor = CrashingExtractor()
        jsonl_path = tmp_path / "artifact.jsonl"

        # First run: crash on second chunk.
        with pytest.raises(RuntimeError, match="simulated crash on second chunk"):
            build_artifact(texts, first_extractor, jsonl_path)

        # Verify first chunk (~500 lines) is on disk, readable, and valid JSON.
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().splitlines()
        # First chunk was 500 texts, so we expect around 500 lines on disk.
        assert len(lines) >= 490
        assert len(lines) <= 510
        # Verify each line is valid JSON with the expected schema.
        stored_hashes = set()
        for line in lines:
            entry = json.loads(line)  # must not raise
            assert "text_sha256" in entry
            assert "text_length" in entry
            stored_hashes.add(entry["text_sha256"])

        # Second run: resume with a fresh extractor that succeeds on all calls.
        second_extractor = FakeExtractor({t: _result_for(t) for t in texts})
        n_new_second = build_artifact(texts, second_extractor, jsonl_path)

        # Verify: resume extracted only the remaining ~100 texts (600 - 500).
        assert n_new_second >= 90
        assert n_new_second <= 110
        # Verify only one extract_entries call made (the second chunk).
        assert len(second_extractor.calls) == 1
        # Verify the resumed extractor was not asked for texts already on disk.
        resumed_chunk = second_extractor.calls[0]
        for text in resumed_chunk:
            assert text_sha256(text) not in stored_hashes

        # Verify the final artifact contains all 600 texts.
        final_lines = jsonl_path.read_text().splitlines()
        assert len(final_lines) == 600
        final_hashes = {json.loads(line)["text_sha256"] for line in final_lines}
        assert final_hashes == {text_sha256(t) for t in texts}

    def test_first_run_extracts_and_appends_every_text(self, tmp_path) -> None:
        texts = ["alpha text", "beta text"]
        fake = FakeExtractor({t: _result_for(t) for t in texts})
        jsonl_path = tmp_path / "artifact.jsonl"

        n_new = build_artifact(texts, fake, jsonl_path)

        assert n_new == 2
        lines = jsonl_path.read_text().splitlines()
        assert len(lines) == 2
        stored_hashes = {json.loads(line)["text_sha256"] for line in lines}
        assert stored_hashes == {text_sha256(t) for t in texts}

    def test_second_run_over_same_inputs_adds_no_lines_and_calls_extractor_zero_times(
        self, tmp_path
    ) -> None:
        texts = ["alpha text", "beta text"]
        fake = FakeExtractor({t: _result_for(t) for t in texts})
        jsonl_path = tmp_path / "artifact.jsonl"

        build_artifact(texts, fake, jsonl_path)
        calls_after_first_run = len(fake.calls)

        n_new_second = build_artifact(texts, fake, jsonl_path)

        assert n_new_second == 0
        assert len(fake.calls) == calls_after_first_run  # zero additional calls
        assert len(jsonl_path.read_text().splitlines()) == 2

    def test_second_fingerprint_writes_a_second_pair_leaving_the_first_byte_identical(
        self, tmp_path
    ) -> None:
        texts = ["gamma text"]
        fake_a = FakeExtractor({t: _result_for(t, cui="C_A") for t in texts})
        fake_b = FakeExtractor({t: _result_for(t, cui="C_B") for t in texts})
        path_a = tmp_path / "scispacy-aaaaaaaaaaaa.jsonl"
        path_b = tmp_path / "scispacy-bbbbbbbbbbbb.jsonl"

        build_artifact(texts, fake_a, path_a)
        content_before = path_a.read_bytes()

        build_artifact(texts, fake_b, path_b)

        assert path_a.read_bytes() == content_before
        assert path_b.exists()
        assert path_b.read_bytes() != content_before  # different cui, different content

    def test_interrupted_run_leaves_a_readable_file_and_resuming_completes_it(
        self, tmp_path
    ) -> None:
        texts = ["one", "two", "three"]
        fake = FakeExtractor({t: _result_for(t) for t in texts})
        jsonl_path = tmp_path / "artifact.jsonl"
        # Simulate a prior crash: only "one" made it to disk.
        jsonl_path.write_text(json.dumps(serialize_entry("one", _result_for("one"))) + "\n")

        n_new = build_artifact(texts, fake, jsonl_path)

        assert n_new == 2  # "two" and "three"
        assert fake.calls == [["two", "three"]]
        stored_hashes = _stored_hashes(jsonl_path)
        assert stored_hashes == {text_sha256(t) for t in texts}

    def test_extractor_omitting_a_requested_text_fails_the_build_not_silently_skipped(
        self, tmp_path
    ) -> None:
        # An extractor with no result for a requested text raises KeyError.
        texts = ["only text"]
        fake = FakeExtractor({})
        jsonl_path = tmp_path / "artifact.jsonl"

        with pytest.raises(KeyError):
            build_artifact(texts, fake, jsonl_path)


class TestWriteMeta:
    """.meta.json gains one build entry per run and preserves prior history."""

    def _params(self) -> dict:
        return {
            "extractor": "scispacy",
            "extractor_version": 1,
            "extractor_params": {"threshold": 0.7},
        }

    def test_writes_schema_version_params_and_the_commit(self, tmp_path) -> None:
        meta_path = tmp_path / "artifact.meta.json"
        params = self._params()

        write_meta(meta_path, params, n_texts=5, n_new=5)

        meta = json.loads(meta_path.read_text())
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["params"] == params
        assert meta["n_new"] == 5
        assert len(meta["git_commit"]) == 40
        assert "created_at" in meta

    def test_second_run_overwrites(self, tmp_path) -> None:
        meta_path = tmp_path / "artifact.meta.json"
        params = self._params()

        write_meta(meta_path, params, n_texts=5, n_new=5)
        write_meta(meta_path, params, n_texts=8, n_new=3)

        meta = json.loads(meta_path.read_text())
        assert (meta["n_texts"], meta["n_new"]) == (8, 3)
