"""Tests for the outlier-review notes persistence layer."""

from bhc_experiment.results import (
    save_review_note,
)
from bhc_experiment.results_reader import load_review_notes


class TestSaveReviewNote:
    def test_overwrites_entry_for_same_hadm_id(self, tmp_path):
        save_review_note(tmp_path, hadm_id=100, tags=["extra_content"], notes="v1")
        save_review_note(tmp_path, hadm_id=100, tags=["structural_mismatch"], notes="v2")

        loaded = load_review_notes(tmp_path)
        assert len(loaded) == 1
        assert loaded[100].tags == ["structural_mismatch"]
        assert loaded[100].notes == "v2"

    def test_preserves_other_entries(self, tmp_path):
        save_review_note(tmp_path, hadm_id=100, tags=["extra_content"], notes="a")
        save_review_note(tmp_path, hadm_id=200, tags=["missing_content"], notes="b")
        save_review_note(tmp_path, hadm_id=100, tags=["extra_content"], notes="a-updated")

        loaded = load_review_notes(tmp_path)
        assert loaded[100].notes == "a-updated"
        assert loaded[200].notes == "b"
