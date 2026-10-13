"""Prompt builder for experiment pipeline.

Three-component pipeline: resolver picks the template, serializer converts
admission records to text, assembler combines them into a PromptResult.

See docs/specs/prompt-builder-design.md for the full design.
"""

from dataclasses import dataclass
from pathlib import Path

from bhc_experiment.loader import (
    AdmissionBundle,
    BhcStructure,
    LabEvent,
    NoteEvent,
    Prescription,
)


def resolve_template_name(variant: str, structure: BhcStructure) -> str:
    """Map a prompt variant + BhcStructure to a template name.

    Returns the template name without extension (e.g., "brief_domain").

    For "unformatted", format conditioning is skipped. Every other variant gets a
    format suffix from the BHC structure.
    """
    if variant == "unformatted":
        return "unformatted"

    suffix = _resolve_format_suffix(structure)
    return f"{variant}{suffix}"


def _resolve_format_suffix(structure: BhcStructure) -> str:
    """Map BhcStructure to a format suffix for template file naming.

    Routing stops at narrative vs. domain-organized. Modifiers such as
    acute_chronic and the numbering convention are ignored: templates fitted to
    finer ground-truth structure were starting to overfit it. The pool has no
    standardized_template admissions, and the domain template is their closest
    format.
    """
    if structure.category == "narrative":
        return "_narrative"
    return "_domain"


def format_admission_records(
    notes: list[NoteEvent],
    labs: list[LabEvent],
    prescriptions: list[Prescription],
) -> str:
    """Convert admission records into the {admission_data} text block.

    Raises if the notes still have records the note filter excludes, then
    serializes every record in one chronological stream.

    Frozen format. Do not change this function's output, including the helper
    formatters it calls. Every already-generated run's saved prompt was built
    here, so a changed format would make new runs incomparable to those. If the
    format must change, add a format_admission_records_v2 and leave this one
    intact.
    """
    _reject_unprepared(notes)
    return _serialize_chronological(notes, labs, prescriptions)


def _reject_unprepared(notes: list[NoteEvent]) -> None:
    """Raise if the notes have anything preparation's filter should have removed.

    Incorrectly prepared records would put filtered-out information back into the
    prompt and invalidate the experiment's results. This check runs at prompt-build
    time to catch a misconfiguration or a bug in the preparation step.

    See ``filter_notes``.
    """
    for note in notes:
        if note.is_discharge_summary:
            kind = "discharge summary"
        elif note.ISERROR is not None:
            kind = "error note"
        else:
            continue
        msg = (
            f"format_admission_records found a {kind} (ROW_ID={note.ROW_ID}). "
            "This bundle was not prepared. Run scripts/pipeline/prepare_inputs.py "
            "first: preparation is the only step that filters."
        )
        raise ValueError(msg)


def filter_notes(notes: list[NoteEvent]) -> list[NoteEvent]:
    """The notes the model may see: no notes marked in error, no discharge summary.

    The discharge note contains the ground-truth BHC, so including it would leak
    the answer into the prompt and make scores meaningless.
    """
    return [n for n in notes if n.ISERROR is None and not n.is_discharge_summary]


def format_note(note: NoteEvent) -> str:
    """Format a single note for serialization."""
    timestamp = note.CHARTTIME or note.CHARTDATE or "unknown date"
    return f"[{timestamp}] NOTE ({note.CATEGORY}): {note.TEXT}"


def format_lab(lab: LabEvent) -> str:
    """Format a single lab result for serialization."""
    timestamp = lab.CHARTTIME or "unknown time"
    value = lab.VALUE if lab.VALUE is not None else "no value"
    text = f"[{timestamp}] LAB {lab.LABEL}: {value}"
    if lab.VALUEUOM:
        text += f" {lab.VALUEUOM}"
    if lab.FLAG:
        text += f" ({lab.FLAG})"
    return text


def format_prescription(rx: Prescription) -> str:
    """Format a single prescription for serialization.

    Follows the same [timestamp] TYPE: content pattern as notes and labs.
    The end date (if present) goes in the body since it's a property of
    the medication order.
    """
    timestamp = rx.STARTDATE or "unknown date"
    text = f"[{timestamp}] PRESCRIPTION: {rx.DRUG}"
    if rx.DOSE_VAL_RX and rx.DOSE_UNIT_RX:
        text += f" {rx.DOSE_VAL_RX}{rx.DOSE_UNIT_RX}"
    if rx.ROUTE:
        text += f" {rx.ROUTE}"
    if rx.ENDDATE:
        text += f" (until {rx.ENDDATE})"
    return text


# Impossible date used as a sort sentinel: a record with no timestamp sorts last
# under lexical string comparison.
_NO_TIMESTAMP_SENTINEL_DATE = "9999-99-99"


def chronological_sort_key(record: NoteEvent | LabEvent | Prescription) -> tuple[str, int]:
    """Sort key for the chronological stream.

    Returns (effective time, row_id). The effective time is CHARTTIME (or
    STARTDATE for prescriptions) when present, and CHARTDATE at noon
    otherwise. Noon keeps a date-only event from always sorting before
    same-day timed events, which is what midnight would do. row_id is the
    record's MIMIC ROW_ID, which makes the order deterministic within each
    type (a lab panel drawn at one CHARTTIME, or same-day prescriptions,
    would otherwise tie and keep arbitrary input order). ROW_ID is
    table-scoped, so a lab and a note at the same second could still tie,
    but same-second cross-type ties are vanishingly rare in the BHC use case.
    """
    if isinstance(record, LabEvent):
        return (record.CHARTTIME or f"{_NO_TIMESTAMP_SENTINEL_DATE} 12:00:00", record.ROW_ID)
    if isinstance(record, NoteEvent):
        if record.CHARTTIME:
            return (record.CHARTTIME, record.ROW_ID)
        chartdate = record.CHARTDATE or _NO_TIMESTAMP_SENTINEL_DATE
        return (f"{chartdate} 12:00:00", record.ROW_ID)
    # Prescription
    startdate = record.STARTDATE or _NO_TIMESTAMP_SENTINEL_DATE
    return (f"{startdate} 12:00:00", record.ROW_ID)


def _serialize_chronological(
    notes: list[NoteEvent],
    labs: list[LabEvent],
    prescriptions: list[Prescription],
) -> str:
    """Merge all records into one stream, sorted by the chronological sort key."""
    entries: list[tuple[tuple[str, int], str]] = []
    entries.extend((chronological_sort_key(note), format_note(note)) for note in notes)
    entries.extend((chronological_sort_key(lab), format_lab(lab)) for lab in labs)
    entries.extend((chronological_sort_key(rx), format_prescription(rx)) for rx in prescriptions)

    entries.sort(key=lambda x: x[0])
    return "\n".join(line for _, line in entries)


# Assumes the package is run from the repo via uv run.
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@dataclass(frozen=True)
class PromptResult:
    """Prompt split into system and user messages, ready for the LLM interface.

    system and user are separate fields because LLM APIs accept them as
    distinct messages with different roles.
    """

    system: str
    user: str
    template_name: str


def build_prompt(
    bundle: AdmissionBundle,
    variant: str,
    *,
    prompts_dir: Path | None = None,
) -> PromptResult:
    """Build the prompt for one admission under a prompt variant.

    Frozen format, for the same reason as format_admission_records (see that
    function's docstring). Add a new version if the format must change.

    prompts_dir overrides the default prompts/ directory, for testing.
    """
    if prompts_dir is None:
        prompts_dir = _PROMPTS_DIR

    # 1. Resolve template
    template_name = resolve_template_name(variant, bundle.bhc_structure)
    template_path = prompts_dir / f"{template_name}.txt"
    if not template_path.exists():
        msg = f"Template file not found: {template_path}"
        raise FileNotFoundError(msg)
    # One file is the entire prompt, so the file shows exactly what was sent to
    # the model.
    template_text = template_path.read_text()

    # 2. Split on --- before substitution, so a --- inside the admission data cannot
    # move the system/user boundary.
    parts = template_text.split("---", maxsplit=1)
    system_part = parts[0].strip()
    user_template = parts[1].strip() if len(parts) > 1 else ""

    # 3. Serialize records
    admission_data = format_admission_records(
        bundle.noteevents, bundle.labevents, bundle.prescriptions
    )

    # 4. Substitute placeholders into user portion.
    # Uses str.replace() instead of str.format() because admission_data may
    # contain curly braces from clinical notes (e.g. "{pt reports...}"),
    # which would cause KeyError or silent substitution with .format().
    user_part = user_template
    user_part = user_part.replace("{age}", str(bundle.admission.age))
    user_part = user_part.replace("{gender}", bundle.admission.GENDER)
    user_part = user_part.replace("{service}", str(bundle.admission.SERVICE or ""))
    user_part = user_part.replace("{admit_time}", bundle.admission.ADMITTIME)
    user_part = user_part.replace("{discharge_time}", bundle.admission.DISCHTIME)
    user_part = user_part.replace("{los_days}", str(bundle.admission.los_days))
    # admission_data last: it contains arbitrary clinical text that may have
    # characters matching other placeholder names.
    user_part = user_part.replace("{admission_data}", admission_data)

    return PromptResult(system=system_part, user=user_part, template_name=template_name)
