"""Tests for the per-admission note_redundancy logic."""

from mimic_data_prep.note_redundancy import detect_note_redundancy


def _note(
    *,
    row_id: int,
    text: str,
    chartdate: str = "2150-01-01",
    charttime: str | None = None,
    category: str = "Nursing",
    description: str = "Note",
) -> dict:
    return {
        "row_id": row_id,
        "chartdate": chartdate,
        "charttime": charttime,
        "category": category,
        "description": description,
        "text": text,
    }


class TestDetectNoteRedundancy:
    """Clusters, PFI/final pairs, and the indices each rule would discard."""

    def test_cluster_keep_drop_and_discarded(self) -> None:
        text = "patient stable on admission with no acute distress noted by the team today"
        notes = [
            _note(row_id=1, text=text, charttime="2150-01-01 08:00:00"),
            _note(row_id=2, text=text, charttime="2150-01-01 16:00:00"),
            _note(
                row_id=3,
                text="left knee radiograph shows a moderate effusion without acute fracture",
            ),
        ]
        detail = detect_note_redundancy(notes)
        assert len(detail["near_duplicate_pairs"]) == 1
        assert len(detail["clusters"]) == 1
        cluster = detail["clusters"][0]
        # Most recent note (index 1) kept. The older (index 0) dropped.
        assert cluster["keep"] == 1
        assert [d["index"] for d in cluster["drop"]] == [0]
        assert cluster["drop"][0]["jaccard"] >= 0.99
        assert detail["in_cluster_count"] == 2
        assert detail["discarded_indices"] == {0}

    def test_pfi_final_pair_detail(self) -> None:
        notes = [
            _note(
                row_id=1,
                text="PFI REPORT WET READ lungs are essentially clear no acute process provisional",
                category="Radiology",
                description="CHEST",
                charttime="2150-01-01 08:00:00",
            ),
            _note(
                row_id=2,
                text="FINAL REPORT lungs clear no acute cardiopulmonary process on the exam today",
                category="Radiology",
                description="CHEST",
                charttime="2150-01-01 14:00:00",
            ),
        ]
        detail = detect_note_redundancy(notes)
        assert len(detail["near_duplicate_pairs"]) == 0
        assert len(detail["pfi_final_pairs"]) == 1
        pair = detail["pfi_final_pairs"][0]
        assert pair["pfi_index"] == 0
        assert pair["final_index"] == 1
        assert detail["discarded_indices"] == {0}
