"""Tests for the prompt builder (bhc_experiment.prompt_builder)."""

import dataclasses
from pathlib import Path

import pytest

from bhc_experiment.loader import AdmissionBundle, BhcStructure, NoteEvent
from bhc_experiment.prompt_builder import (
    build_prompt,
    chronological_sort_key,
    filter_notes,
    format_admission_records,
    resolve_template_name,
)

from .conftest import make_bundle, make_lab, make_note, make_prescription


class TestResolveTemplateName:
    """Both format-matched templates exist, so a routing mix-up runs without error."""

    def test_format_matched_narrative(self) -> None:
        structure = BhcStructure(category="narrative", numbering_convention=None, modifiers=[])
        assert resolve_template_name("format-matched", structure) == "format-matched_narrative"

    def test_format_matched_domain(self) -> None:
        structure = BhcStructure(
            category="domain_organized", numbering_convention="#", modifiers=["acute_chronic"]
        )
        assert resolve_template_name("format-matched", structure) == "format-matched_domain"


def _make_notes() -> list[NoteEvent]:
    return [
        make_note(1, "Day 2 note."),
        make_note(2, "Error note.", CATEGORY="Physician", ISERROR="1"),
        make_note(3, "Pre-admission.", is_pre_admission=True),
        make_note(4, "BHC text.", CATEGORY="Discharge summary", is_discharge_summary=True),
    ]


class TestFilterNotes:
    """The note filter drops notes marked in error and discharge summaries."""

    def test_keeps_every_other_note_including_pre_admission(self) -> None:
        texts = [n.TEXT for n in filter_notes(_make_notes())]
        assert texts == ["Day 2 note.", "Pre-admission."]


class TestChronologicalSortKey:
    """The (effective time, ROW_ID) sort key handles all three event types."""

    def test_lab_with_charttime(self) -> None:
        lab = make_lab(7, CHARTTIME="2150-01-03 08:00:00")
        assert chronological_sort_key(lab) == ("2150-01-03 08:00:00", 7)

    def test_note_with_charttime_uses_it(self) -> None:
        note = make_note(42, CHARTDATE="2150-01-03", CHARTTIME="2150-01-03 14:30:00")
        assert chronological_sort_key(note) == ("2150-01-03 14:30:00", 42)

    def test_note_without_charttime_falls_back_to_noon(self) -> None:
        note = make_note(42, CHARTDATE="2150-01-03", CHARTTIME=None, CATEGORY="ECG")
        assert chronological_sort_key(note) == ("2150-01-03 12:00:00", 42)

    def test_prescription_uses_startdate_at_noon(self) -> None:
        rx = make_prescription(9, STARTDATE="2150-01-04")
        assert chronological_sort_key(rx) == ("2150-01-04 12:00:00", 9)


class TestSerializeRecords:
    """Tests for record ordering."""

    def test_chronological_interleaves_by_timestamp(self) -> None:
        notes = [make_note(1, "Note text.", CHARTDATE="2150-01-03", CHARTTIME=None)]
        labs = [make_lab(0, CHARTTIME="2150-01-02 08:00:00")]
        prescriptions = [make_prescription(0, STARTDATE="2150-01-04")]
        result = format_admission_records(notes, labs, prescriptions)
        lines = result.strip().split("\n")
        # Lab (Jan 2) before Note (Jan 3) before Prescription (Jan 4)
        assert "LAB" in lines[0]
        assert "NOTE" in lines[1]
        assert "PRESCRIPTION" in lines[2]

    @pytest.mark.parametrize("text", ["Error note.", "BHC text."])
    def test_rejects_a_note_preparation_drops(self, text: str) -> None:
        """Preparation drops these notes. format_admission_records only checks that it did."""
        notes = [n for n in _make_notes() if text == n.TEXT]
        with pytest.raises(ValueError, match="preparation"):
            format_admission_records(notes, [], [])


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts directory with a test template file."""
    d = tmp_path / "prompts"
    d.mkdir()

    (d / "unformatted.txt").write_text(
        "You are a clinical assistant.\n"
        "\n"
        "---\n"
        "\n"
        "Patient: {age} year old {gender}\n"
        "Admitted: {admit_time}\n"
        "Discharged: {discharge_time}\n"
        "\n"
        "{admission_data}"
    )

    return d


def _make_bundle() -> AdmissionBundle:
    """Create a minimal AdmissionBundle for assembler tests."""
    return make_bundle(
        [make_note(1, "Day 2 note.", CHARTDATE="2150-01-03", CHARTTIME=None)],
        labs=[make_lab(0, CHARTTIME="2150-01-03 08:00:00")],
    )


class TestBuildPrompt:
    """Tests for the full build_prompt pipeline."""

    def test_system_message_from_above_separator(self, prompts_dir: Path) -> None:
        """System message comes from text above --- in the template."""
        result = build_prompt(_make_bundle(), "unformatted", prompts_dir=prompts_dir)
        assert "clinical assistant" in result.system
        assert "Patient:" not in result.system

    def test_user_message_has_demographics(self, prompts_dir: Path) -> None:
        bundle = _make_bundle()
        admission = dataclasses.replace(bundle.admission, age=72, GENDER="M")

        result = build_prompt(
            dataclasses.replace(bundle, admission=admission), "unformatted", prompts_dir=prompts_dir
        )

        assert "Patient: 72 year old M" in result.user


class TestRealTemplates:
    """A placeholder typo in a real template (e.g. {admit_date}) is left in the prompt
    as literal text, and the run goes ahead without error.
    """

    @pytest.mark.parametrize(
        ("variant", "category"),
        [
            ("unformatted", "narrative"),
            ("format-matched", "narrative"),
            ("format-matched", "domain_organized"),
            ("detailed", "narrative"),
        ],
    )
    def test_every_placeholder_is_filled(self, variant: str, category: str) -> None:
        bundle = make_bundle(
            [make_note(1, "Day 2 note.", CHARTDATE="2150-01-03", CHARTTIME=None)],
            bhc_category=category,
        )
        result = build_prompt(bundle, variant)
        records = format_admission_records(bundle.noteevents, [], [])
        assert records in result.user
        assert "{" not in result.user.replace(records, "")
