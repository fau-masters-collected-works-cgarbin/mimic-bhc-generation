"""Outlier Review: per-admission qualitative review of scored experiments.

The worklist is the worst N admissions in the run, ranked by composite
z-score and computed live (no pre-generation step).

Composite z-score: for each metric, how far an admission scores below the
cohort mean, in standard deviations. An admission's composite is its worst
metric's gap, so the ranking flags whoever underperforms the cohort most on
at least one metric.

Entry: uv run streamlit run app/outlier_review.py
"""

import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import streamlit as st
from _presentation import DETAIL_COHORT_ATTRS, NOTE_STYLE, html_escape
from mimic_bhc import StructureResult, classify_bhc_structure

from bhc_experiment.results import ReviewNote, ScoredResult, save_review_note
from bhc_experiment.results_reader import (
    PRIMARY_METRICS,
    cohort_attributes,
    discover_runs,
    extract_run_label,
    load_reference_bhc,
    load_review_notes,
    load_run,
    load_run_df,
    run_cli_parameters,
    worst_by_composite_z,
)
from mimic_data_prep.mimic import BHC_JSONL, RESULTS_DIR

# Controlled tag vocabulary. Meanings live in the multiselect help text.
REVIEW_TAGS = [
    "extra_content",
    "missing_content",
    "temporal_confusion",
    "structural_mismatch",
    "fabrication",
    "ok",
]

# Metrics used for the between-model stdev pre-read signal (does the failure
# vary across the models, or is it admission-inherent?).
_BETWEEN_MODEL_SIGNAL_METRICS = [
    "rougeL/f1",
    "sapbert/f1",
    "scispacy-clinical/f1",
    "te3small/similarity",
]


st.set_page_config(page_title="Outlier Review", layout="wide")
st.title("Outlier Review")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data
def _discover_scored_runs() -> list[str]:
    """Every run directory under the results tree, minus smoke runs.

    A smoke run is a single-model wiring check with no between-model signal, so
    it is dropped here. Every run the app loads then has the full model set.
    """
    return [
        str(run_dir)
        for run_dir in discover_runs(RESULTS_DIR)
        if not run_dir.parent.name.startswith("smoke")
    ]


@st.cache_data
def _load_run_df_cached(run_dir: str) -> pd.DataFrame:
    return load_run_df(Path(run_dir))


@st.cache_data
def _load_hadm_lookup(run_dir: str) -> dict[int, list]:
    """Build {hadm_id: [(meta, scored_result), ...]} over the run's models."""
    lookup: dict[int, list] = defaultdict(list)
    for meta, scored_list in load_run(Path(run_dir)):
        for sr in scored_list:
            lookup[sr.hadm_id].append((meta, sr))
    return dict(lookup)


@st.cache_data
def _load_cohort_map(run_dir: str) -> dict[int, dict]:
    return cohort_attributes(run_cli_parameters(Path(run_dir)).cohort_file)


@st.cache_data
def _load_reference_map(bhc_path_str: str, pick_ids_key: str) -> dict[int, str]:
    """Load reference BHC text for every pick in one pass over bhc.jsonl.

    Cache key is the full set of pick IDs so the file is scanned once per run
    per session. A Streamlit rerun reuses it.
    """
    if not pick_ids_key:
        return {}
    hids = {int(x) for x in pick_ids_key.split(",")}
    return load_reference_bhc(Path(bhc_path_str), hids)


# ---------------------------------------------------------------------------
# Sidebar: run selection
# ---------------------------------------------------------------------------


runs = _discover_scored_runs()
if not runs:
    st.warning("No scored runs found under the results tree.")
    st.stop()

selected_run = st.sidebar.selectbox(
    "Run",
    options=runs,
    format_func=lambda run: str(Path(run).relative_to(RESULTS_DIR)),
    key="selected_run",
)
run_dir = Path(selected_run)

worklist_size = st.sidebar.slider(
    "Worklist size (worst N by composite z)",
    min_value=5,
    max_value=50,
    value=24,
    key="worklist_size",
)

with st.sidebar.expander("Reviewer protocol", expanded=False):
    st.markdown(
        """
**What to tag**

Properties of the text itself: truncation, repetition, register
drift (summary vs copy-pasted notes), organization (problem-based,
chronological, or disorganized), internal contradiction, missing sections
(admission reason, course, disposition), obvious demographic errors.

**Pre-read signals**

Four signals appear in the admission and model panels. Read them before the
text to set expectations.

| Signal | What it is | How to read it |
| --- | --- | --- |
| Length ratio | Generated words / reference words | Well below 1: truncation or missing content. Well above 1: padding or raw-note quotation. |
| MEDCON P-R delta | SciSpaCy precision minus recall | Positive: missing content. Negative: extra content, a reference gap or fabrication. |
| Between-model spread | How much the metric varies across the models for this admission | High: some models did better, so open the weak one. Low: the admission is hard for all of them (sparse input, extreme LOS). |
| Structure | BHC structure (format) classifier for the generated text | Disagreement with the reference BHC format may explain semantic scores. |

**Separate observation from interpretation**

Tag what the text does. Causes go in the notes field or the
end-of-session memo.

**Stop condition**

Review to saturation. When several picks in a row produce no
new failure modes, stop and move to synthesis.

**End of session**

Write down what surprised you, which tags cluster, and what is worth
measuring across the whole cohort. The dominant tags become the next thing
to quantify.
        """
    )


# ---------------------------------------------------------------------------
# Load run data
# ---------------------------------------------------------------------------


hadm_lookup = _load_hadm_lookup(selected_run)
if not hadm_lookup:
    st.error("No results in this run.")
    st.stop()

cohort_map = _load_cohort_map(selected_run)
# Notes are read on every rerun (no cache). A save should be reflected
# immediately in the sidebar progress and the form's prefill state.
# Caching would need explicit invalidation on save.
notes_map = load_review_notes(run_dir)

# The run's worklist: worst N admissions, worst first.
run_df = _load_run_df_cached(selected_run)
ranked_hadm_ids: list[int] = []
reason_by_hadm: dict[int, tuple[float, str]] = {}
for hadm_id, composite_z, driver in worst_by_composite_z(run_df, worklist_size):
    ranked_hadm_ids.append(hadm_id)
    reason_by_hadm[hadm_id] = (composite_z, driver)
reference_map = _load_reference_map(str(BHC_JSONL), ",".join(str(h) for h in ranked_hadm_ids))


# ---------------------------------------------------------------------------
# Sidebar: progress + pick list
# ---------------------------------------------------------------------------

reviewed_count = sum(1 for hid in ranked_hadm_ids if hid in notes_map)
st.sidebar.divider()
st.sidebar.subheader("Picks")
st.sidebar.caption(f"Reviewed: **{reviewed_count} / {len(ranked_hadm_ids)}**")
hide_reviewed = st.sidebar.checkbox("Hide reviewed", value=False, key="hide_reviewed")

# Rank order, worst first, because this is a triage worklist.
pick_items: list[tuple[int, str]] = []
for hid in ranked_hadm_ids:
    if hide_reviewed and hid in notes_map:
        continue
    marker = "\u2705" if hid in notes_map else "\u2b1c"
    cat = cohort_map.get(hid, {}).get("bhc_category", "?")
    composite_z, driver = reason_by_hadm[hid]
    pick_items.append((hid, f"{marker} {hid} | {cat} | {driver} z={composite_z:.2f}"))

if not pick_items:
    st.info("All picks reviewed. Uncheck 'Hide reviewed' to revisit.")
    st.stop()

# Reset the selected admission when the run changes so a leftover
# hadm_id from another run doesn't point at nothing.
if st.session_state.get("_last_run") != selected_run:
    st.session_state["_last_run"] = selected_run
    st.session_state["selected_hadm_id"] = pick_items[0][0]

pick_ids = [p[0] for p in pick_items]
pick_labels = [p[1] for p in pick_items]

# Sync the radio state to the current selection BEFORE the widget
# renders, so programmatic changes (run switch, "hide reviewed"
# filter removing the current pick) take effect on the next paint.
current_hid = st.session_state.get("selected_hadm_id")
if current_hid in pick_ids:
    target_label = pick_labels[pick_ids.index(current_hid)]
    if st.session_state.get("pick_radio") != target_label:
        st.session_state["pick_radio"] = target_label
else:
    st.session_state["selected_hadm_id"] = pick_ids[0]
    st.session_state["pick_radio"] = pick_labels[0]


def _on_pick_change() -> None:
    label = st.session_state["pick_radio"]
    st.session_state["selected_hadm_id"] = pick_ids[pick_labels.index(label)]


st.sidebar.radio(
    "Select admission",
    pick_labels,
    key="pick_radio",
    on_change=_on_pick_change,
    label_visibility="collapsed",
)

selected_hid = st.session_state["selected_hadm_id"]


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _composite_z_note(composite_z: float, driver_metric: str) -> str:
    """One-line explanation of why this admission is on the worklist."""
    return (
        f"Worst {composite_z:.2f}\u03c3 below cohort mean, driven by {driver_metric}. "
        f"Read the outputs for the failure pattern the metric is flagging."
    )


def _format_structure(
    category: str | None, modifiers: Iterable[str] | None, numbering: str | None
) -> str:
    """Render a (category, modifiers, numbering) tuple as a compact label."""
    if not category:
        return "(unknown)"
    mods = list(modifiers) if modifiers else []
    parts = [category]
    if mods:
        parts.append(f"mods={','.join(mods)}")
    if numbering:
        parts.append(f"numbering={numbering}")
    return " | ".join(parts)


def _classify_generated(text: str) -> StructureResult | None:
    """Classify generated text, returning None on empty input."""
    if not text or not text.strip():
        return None
    return classify_bhc_structure(text)


def _structure_mismatch(ref_attrs: dict, gen_structure: StructureResult | None) -> bool:
    """Return True if generated structure differs from reference metadata.

    Numbering convention (hash or numbered headers) is excluded, since matching
    it in the prompt did not change the scores.
    """
    if gen_structure is None:
        return False
    ref_cat = ref_attrs.get("bhc_category")
    ref_mods = set(ref_attrs.get("modifiers") or [])
    gen_mods = set(gen_structure.modifiers or ())
    return gen_structure.category != ref_cat or gen_mods != ref_mods


def _between_model_stdev(
    model_rows: list[tuple[dict, ScoredResult]], metrics: list[str]
) -> dict[str, float]:
    """Population stdev of each metric across the admission's models.

    Stdev (not variance) is used so the number is on the same scale as
    the metric itself. A metric absent from the run's scores is skipped.
    """
    out: dict[str, float] = {}
    for m in metrics:
        vals = [sr.scores[m] for _, sr in model_rows if sr.scores and m in sr.scores]
        if vals:
            out[m] = statistics.pstdev(vals)
    return out


def _render_admission_signals(
    attrs: dict,
    model_rows: list[tuple[dict, ScoredResult]],
    ref_word_count: int | None,
) -> None:
    """Render the admission-level pre-read signal row."""
    ref_structure_label = _format_structure(
        attrs.get("bhc_category"), attrs.get("modifiers"), attrs.get("numbering_convention")
    )
    signal_lines: list[str] = [f"Reference: {ref_structure_label}"]
    if ref_word_count:
        signal_lines[-1] += f" | words={ref_word_count}"
    stdevs = _between_model_stdev(model_rows, _BETWEEN_MODEL_SIGNAL_METRICS)
    if stdevs:
        stdev_str = ", ".join(f"{k}={v:.3f}" for k, v in stdevs.items())
        signal_lines.append(f"Between-model stdev: {stdev_str}")
    st.caption("  \n".join(signal_lines))


def _model_signal_parts(
    gen_text: str, scores: dict, ref_word_count: int | None, ref_attrs: dict
) -> list[str]:
    """Assemble the pre-read signal row for one model as a list of labels."""
    parts: list[str] = []
    gen_words = len(gen_text.split()) if gen_text else 0
    if ref_word_count:
        parts.append(f"words={gen_words} (ratio {gen_words / ref_word_count:.2f})")
    else:
        parts.append(f"words={gen_words}")

    gen_structure = _classify_generated(gen_text)
    if gen_structure is not None:
        gen_label = _format_structure(
            gen_structure.category,
            gen_structure.modifiers,
            gen_structure.numbering_convention,
        )
        mark = "!=" if _structure_mismatch(ref_attrs, gen_structure) else "=="
        parts.append(f"structure {mark} {gen_label}")

    if "scispacy-clinical/precision" in scores and "scispacy-clinical/recall" in scores:
        p = scores["scispacy-clinical/precision"]
        r = scores["scispacy-clinical/recall"]
        parts.append(f"MEDCON P-R={p - r:+.3f}")
    return parts


def _render_model_panel(
    meta: dict,
    scored: ScoredResult,
    *,
    highlighted_metrics: list[str],
    ref_word_count: int | None,
    ref_attrs: dict,
) -> None:
    """Render one model's panel: model label, scores, generated text."""
    _, model_key = extract_run_label(meta)
    st.markdown(f"**{model_key}**")

    llm = scored.generation.llm_result
    gen_text = llm.text if llm else ""
    error_text = None if llm else f"(Error: {scored.generation.error})"

    # Pre-read signals are skipped when a model errored to avoid a misleading
    # "words=0 (ratio 0.00)" row when there is no output to characterize.
    if llm is not None:
        signal_parts = _model_signal_parts(gen_text, scored.scores, ref_word_count, ref_attrs)
        if signal_parts:
            st.caption(" | ".join(signal_parts))

    if scored.scores:
        # Primary metrics first, so the pick's driver is scannable without
        # re-reading the full row.
        present = [m for m in highlighted_metrics if m in scored.scores]
        if present:
            parts = [f"**`{m}`={scored.scores[m]:.4f}**" for m in present]
            st.markdown(" | ".join(parts))
        score_row = " | ".join(f"`{k}`={v:.4f}" for k, v in sorted(scored.scores.items()))
        st.caption(score_row)

    display_text = gen_text or error_text or ""
    with st.container(height=360):
        st.markdown(
            f'<div style="{NOTE_STYLE}">{html_escape(display_text)}</div>',
            unsafe_allow_html=True,
        )


def _render_admission(  # noqa: PLR0913
    hadm_id: int,
    *,
    composite_z: float,
    driver_metric: str,
    attrs: dict,
    ref_text: str,
    model_rows: list[tuple[dict, ScoredResult]],
    note: ReviewNote | None,
    write_dir: Path,
) -> None:
    """Render the per-admission view: header, reference, one panel per model, notes form."""
    st.subheader(f"HADM_ID: {hadm_id}")
    attr_parts = [f"{k}: {attrs[k]}" for k in DETAIL_COHORT_ATTRS if attrs.get(k) is not None]
    if attr_parts:
        st.caption(" | ".join(attr_parts))
    st.info(_composite_z_note(composite_z, driver_metric))

    ref_word_count = attrs.get("bhc_word_count")
    _render_admission_signals(attrs, model_rows, ref_word_count)

    with st.expander("Reference BHC", expanded=True):
        st.markdown(
            f'<div style="{NOTE_STYLE}">{html_escape(ref_text)}</div>',
            unsafe_allow_html=True,
        )

    highlighted = list(PRIMARY_METRICS)
    for meta, sr in sorted(model_rows, key=lambda pair: extract_run_label(pair[0])[1]):
        with st.container(border=True):
            _render_model_panel(
                meta,
                sr,
                highlighted_metrics=highlighted,
                ref_word_count=ref_word_count,
                ref_attrs=attrs,
            )

    # Notes form.
    st.divider()
    st.subheader("Review notes")
    with st.form(f"note_form_{hadm_id}", clear_on_submit=False):
        tags = st.multiselect(
            "Tags",
            options=REVIEW_TAGS,
            default=note.tags if note else [],
            key=f"note_tags_{hadm_id}",
            help=(
                "extra_content = content in the output that is not in the "
                "reference BHC. May be valid EHR-grounded content the "
                "reference omitted, or fabrication. Reference alone cannot "
                "distinguish. Use fabrication only after a source-notes "
                "check has been run. "
                "missing_content = content in the reference BHC that is not "
                "in the output. "
                "temporal_confusion = events in the wrong order or with the "
                "wrong date. "
                "structural_mismatch = narrative shape does not match the "
                "reference (e.g., narrative response to a domain-organized "
                "reference). "
                "fabrication = content in the output that is not supported "
                "by the source notes. Automation territory. Faithfulness "
                "against the EHR is measured by metrics and automated "
                "grounding checks, not by opening notes to adjudicate by "
                "hand. Flag only when the reference directly contradicts "
                "the output or when a faithfulness check has already "
                "flagged the case. "
                "ok = reviewed, no issue worth tagging."
            ),
        )
        note_text = st.text_area(
            "Notes",
            value=note.notes if note else "",
            height=150,
            key=f"note_text_{hadm_id}",
            placeholder="Free-text observations. What failure modes are visible? "
            "Does the pattern differ across variants or models?",
        )
        submitted = st.form_submit_button("Save")

    if submitted:
        saved = save_review_note(write_dir, hadm_id=hadm_id, tags=tags, notes=note_text)
        st.success(f"Saved at {saved.updated_at}")
    elif note is not None:
        st.caption(f"Last saved: {note.updated_at}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


selected_z, selected_driver = reason_by_hadm[selected_hid]
_render_admission(
    selected_hid,
    composite_z=selected_z,
    driver_metric=selected_driver,
    attrs=cohort_map.get(selected_hid, {}),
    ref_text=reference_map.get(selected_hid, "(Reference BHC not found in bhc.jsonl)"),
    model_rows=hadm_lookup.get(selected_hid, []),
    note=notes_map.get(selected_hid),
    write_dir=run_dir,
)
