"""Tests for the prompt builder (bhc_experiment.prompt_builder)."""

import dataclasses
from pathlib import Path

import pytest

from bhc_experiment.loader import AdmissionBundle, BhcStructure, NoteEvent
from bhc_experiment.prompt_builder import (
    build_prompt,
    filter_notes,
    format_admission_records,
    format_note,
    resolve_template_name,
)

from .conftest import make_bundle, make_lab, make_note, make_prescription


class TestResolveTemplateName:
    """Tests for template name resolution from variant + BhcStructure.

    One test per heading convention.
    """

    def test_unformatted_ignores_structure(self) -> None:
        """Unformatted variant always resolves to 'unformatted', ignoring structure."""
        structure = BhcStructure(
            category="domain_organized",
            numbering_convention="#",
            modifiers=["acute_chronic"],
        )
        assert resolve_template_name("unformatted", structure) == "unformatted"

    def test_brief_narrative(self) -> None:
        structure = BhcStructure(category="narrative", numbering_convention=None, modifiers=[])
        assert resolve_template_name("brief", structure) == "brief_narrative"

    def test_brief_standardized_template_routes_to_domain(self) -> None:
        """Format routing stops at narrative vs. domain-organized."""
        structure = BhcStructure(
            category="standardized_template", numbering_convention=None, modifiers=[]
        )
        assert resolve_template_name("brief", structure) == "brief_domain"

    @pytest.mark.parametrize("convention", ["#", None])
    def test_brief_domain_ignores_numbering_convention(self, convention: str | None) -> None:
        """A domain-organized BHC resolves to 'brief_domain' regardless of
        numbering convention.
        """
        structure = BhcStructure(
            category="domain_organized", numbering_convention=convention, modifiers=[]
        )
        assert resolve_template_name("brief", structure) == "brief_domain"

    @pytest.mark.parametrize(
        "modifiers", [["acute_chronic"], ["acute_chronic", "transitional_issues"]]
    )
    def test_brief_domain_ignores_modifiers(self, modifiers: list[str]) -> None:
        """Format routing stops at narrative vs. domain-organized."""
        structure = BhcStructure(
            category="domain_organized", numbering_convention="#", modifiers=modifiers
        )
        assert resolve_template_name("brief", structure) == "brief_domain"

    def test_brief_domain_transitional_only_falls_through(self) -> None:
        """transitional_issues alone does not trigger its own template."""
        structure = BhcStructure(
            category="domain_organized",
            numbering_convention="#",
            modifiers=["transitional_issues"],
        )
        assert resolve_template_name("brief", structure) == "brief_domain"

    @pytest.mark.parametrize("category", ["narrative", "domain_organized"])
    def test_thorough_skips_format_routing(self, category: str) -> None:
        structure = BhcStructure(category=category, numbering_convention=None, modifiers=[])
        assert resolve_template_name("thorough", structure) == "thorough"


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
        from bhc_experiment.prompt_builder import chronological_sort_key

        lab = make_lab(7, CHARTTIME="2150-01-03 08:00:00")
        assert chronological_sort_key(lab) == ("2150-01-03 08:00:00", 7)

    def test_note_with_charttime_uses_it(self) -> None:
        from bhc_experiment.prompt_builder import chronological_sort_key

        note = make_note(42, CHARTDATE="2150-01-03", CHARTTIME="2150-01-03 14:30:00")
        assert chronological_sort_key(note) == ("2150-01-03 14:30:00", 42)

    def test_note_without_charttime_falls_back_to_noon(self) -> None:
        from bhc_experiment.prompt_builder import chronological_sort_key

        note = make_note(42, CHARTDATE="2150-01-03", CHARTTIME=None, CATEGORY="ECG")
        assert chronological_sort_key(note) == ("2150-01-03 12:00:00", 42)

    def test_prescription_uses_startdate_at_noon(self) -> None:
        from bhc_experiment.prompt_builder import chronological_sort_key

        rx = make_prescription(9, STARTDATE="2150-01-04")
        assert chronological_sort_key(rx) == ("2150-01-04 12:00:00", 9)


class TestFormatNote:
    def test_category_trailing_space_is_stripped(self) -> None:
        """MIMIC-III stores "Physician ", "Respiratory ", and "Case Management "."""
        note = make_note(1, "Text.", CATEGORY="Physician ", CHARTTIME="2150-01-01 14:00:00")
        assert format_note(note) == "[2150-01-01 14:00:00] NOTE (Physician): Text."


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

    def test_chronological_same_day_lab_before_date_only_note(self) -> None:
        """Notes without CHARTTIME fall back to CHARTDATE at noon. A lab at
        08:00 on the same day sorts before that noon fallback.
        """
        notes = [
            make_note(1, "Date-only note.", CHARTDATE="2150-01-03", CHARTTIME=None, CATEGORY="ECG")
        ]
        labs = [make_lab(0, CHARTTIME="2150-01-03 08:00:00")]
        result = format_admission_records(notes, labs, [])
        lines = result.strip().split("\n")
        assert "LAB" in lines[0]
        assert "NOTE" in lines[1]

    @pytest.mark.parametrize("text", ["Error note.", "BHC text."])
    def test_rejects_a_note_preparation_drops(self, text: str) -> None:
        """Preparation drops these notes. format_admission_records only checks that it did."""
        notes = [n for n in _make_notes() if text == n.TEXT]
        with pytest.raises(ValueError, match="preparation"):
            format_admission_records(notes, [], [])


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts directory with test template files."""
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

    (d / "brief_narrative.txt").write_text(
        "You are a clinical assistant.\n"
        "\n"
        "Write as a continuous narrative.\n"
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
    """Smoke tests that verify real template files work with build_prompt.

    Catches placeholder typos (e.g. {admit_date} instead of {admit_time})
    that would otherwise only fail during a run.
    """

    def _bundle_with_structure(self, **structure_kwargs: object) -> AdmissionBundle:
        return dataclasses.replace(
            _make_bundle(),
            bhc_category=structure_kwargs.get("category", "narrative"),
            bhc_structure=BhcStructure(**structure_kwargs),  # type: ignore[arg-type]
        )

    def test_unformatted(self) -> None:
        result = build_prompt(_make_bundle(), "unformatted")
        assert result.template_name == "unformatted"

    def test_brief_narrative(self) -> None:
        bundle = self._bundle_with_structure(
            category="narrative", numbering_convention=None, modifiers=[]
        )
        result = build_prompt(bundle, "brief")
        assert result.template_name == "brief_narrative"

    def test_brief_domain(self) -> None:
        bundle = self._bundle_with_structure(
            category="domain_organized", numbering_convention="#", modifiers=[]
        )
        result = build_prompt(bundle, "brief")
        assert result.template_name == "brief_domain"

    def test_brief_domain_numbered_routes_to_domain(self) -> None:
        bundle = self._bundle_with_structure(
            category="domain_organized", numbering_convention="N.", modifiers=[]
        )
        result = build_prompt(bundle, "brief")
        assert result.template_name == "brief_domain"


_REAL_PROMPTS_DIR = Path(__file__).parents[1] / "prompts"
# Paragraph positions in every template. Template text is split on blank lines,
# so the user message after --- is compared too.
_CONTENT_PARAGRAPH = 2
_FORMAT_PARAGRAPH = 7


def _paragraphs(template_name: str) -> list[str]:
    return (_REAL_PROMPTS_DIR / f"{template_name}.txt").read_text().split("\n\n")


def _without(paragraphs: list[str], index: int) -> list[str]:
    return paragraphs[:index] + paragraphs[index + 1 :]


class TestPromptVariantDifferences:
    """brief and thorough each differ from unformatted in exactly one paragraph,
    so a score difference between them comes from that instruction alone.
    """

    def test_template_set(self) -> None:
        assert {p.stem for p in _REAL_PROMPTS_DIR.glob("*.txt")} == {
            "unformatted",
            "brief_narrative",
            "brief_domain",
            "thorough",
        }

    @pytest.mark.parametrize("fmt", ["narrative", "domain"])
    def test_unformatted_is_brief_without_format(self, fmt: str) -> None:
        assert _paragraphs("unformatted") == _without(
            _paragraphs(f"brief_{fmt}"), _FORMAT_PARAGRAPH
        )

    def test_thorough_is_unformatted_with_thoroughness(self) -> None:
        assert _without(_paragraphs("thorough"), _CONTENT_PARAGRAPH) == _without(
            _paragraphs("unformatted"), _CONTENT_PARAGRAPH
        )
        thoroughness = _paragraphs("thorough")[_CONTENT_PARAGRAPH]
        assert thoroughness != _paragraphs("unformatted")[_CONTENT_PARAGRAPH]
