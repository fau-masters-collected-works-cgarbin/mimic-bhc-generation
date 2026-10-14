"""Run Explorer: summary table and per-admission detail view."""

from pathlib import Path

import pandas as pd
import streamlit as st
from _presentation import DETAIL_COHORT_ATTRS, NOTE_STYLE, html_escape, scores_to_table

from bhc_experiment.config import CliParameters
from bhc_experiment.results_reader import (
    cohort_attributes,
    get_result_record,
    load_reference_bhc,
    load_results_df,
    load_run_record,
)
from mimic_data_prep.mimic import BHC_JSONL

st.title("Run Explorer")

# ---------------------------------------------------------------------------
# Load run data
# ---------------------------------------------------------------------------
model_dir = st.session_state.get("selected_model_dir")
if model_dir is None:
    st.info("Select a model from the sidebar.")
    st.stop()


@st.cache_data
def _load_record(model_dir_str: str):
    return load_run_record(Path(model_dir_str))


@st.cache_data
def _load_df(model_dir_str: str):
    return load_results_df(Path(model_dir_str))


run_record = _load_record(str(model_dir))
df = _load_df(str(model_dir))

if df.empty:
    st.warning("No results in this run.")
    st.stop()

# ---------------------------------------------------------------------------
# State: selected_hadm_id drives everything.
# None = summary view, int = detail view for that admission.
# ---------------------------------------------------------------------------
hadm_ids = df["hadm_id"].tolist()

if "selected_hadm_id" not in st.session_state:
    st.session_state["selected_hadm_id"] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data
def _load_bhc_refs(bhc_path_str: str, hadm_ids_key: str) -> dict[int, str]:
    """Load reference BHC text for the given hadm_ids."""
    hids = {int(x) for x in hadm_ids_key.split(",")}
    return load_reference_bhc(Path(bhc_path_str), hids)


@st.cache_data
def _get_record(model_dir_str: str, hid: int):
    """Retrieve the full JSONL record for a single hadm_id."""
    return get_result_record(Path(model_dir_str), hid)


@st.cache_data
def _load_cohort_lookup(cohort_file: str) -> dict[int, dict]:
    return cohort_attributes(cohort_file)


cohort_lookup = _load_cohort_lookup(CliParameters(**run_record["cli_parameters"]).cohort_file)


# ---------------------------------------------------------------------------
# Summary view
# ---------------------------------------------------------------------------


def _show_summary():
    """Render the run record header and summary table."""
    cli_parameters = CliParameters(**run_record["cli_parameters"])
    run_name = cli_parameters.run_name
    model = cli_parameters.model
    status = run_record.get("status", "?")
    started = run_record.get("started_at", "?")
    git_commit = run_record.get("git_commit", "?")
    total = run_record.get("total_admissions", "?")
    completed = run_record.get("completed_count", "?")
    failed = run_record.get("failed_count", "?")

    st.caption(
        f"{run_name} | {model} | {status} | started {started} | "
        f"git {git_commit} | {completed}/{total} completed, {failed} failed"
    )

    # Score columns are the metric keys ("family/submetric"). Fix them to four
    # decimals so Streamlit's dropped trailing zeros do not leave the column ragged.
    score_cols = [c for c in df.columns if "/" in c]
    event = st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="summary_table",
        column_config={c: st.column_config.NumberColumn(format="%.4f") for c in score_cols},
    )

    # st.dataframe has no on_change callback, so an explicit st.rerun() after
    # updating session state is the only way to drive the dispatch.
    selected_rows = event.selection.rows
    if selected_rows:
        row_idx = selected_rows[0]
        new_hid = hadm_ids[row_idx]
        if st.session_state["selected_hadm_id"] != new_hid:
            st.session_state["selected_hadm_id"] = new_hid
            st.rerun()


# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------


def _render_source_material(attrs: dict, record) -> None:
    """How much source material this admission has, as a difficulty read.

    Summarization gets harder as the input grows and fragments, so this shows the
    stay length, the source-record counts by type, and the token load the model
    compressed against the reference length.
    """
    if not attrs:
        st.caption("No cohort profile for this admission.")
        return

    st.markdown("**Source material**")

    diversity = attrs.get("source_diversity", {})
    completeness = attrs.get("timestamp_completeness", {})
    lines = [
        f"- **LOS**: {attrs['los_days']} days",
        f"- **Charting span**: {attrs['charting_days']} days",
        f"- **Notes**: {attrs['note_count']}",
        f"- **Labs**: {attrs['lab_count']} ({attrs['lab_abnormal_count']} abnormal)",
        f"- **Prescriptions**: {attrs['prescription_count']}",
        (
            f"- **Distinct labs / drugs**: "
            f"{diversity.get('distinct_lab_items', 0)} / {diversity.get('distinct_drugs', 0)}"
        ),
        f"- **Notes missing timestamp**: {completeness.get('null_charttime_note_count', 0)}",
    ]
    llm = record.llm_result
    if llm and llm.input_tokens:
        lines.append(f"- **Input tokens**: {llm.input_tokens:,}")
    else:
        lines.append(f"- **Source tokens**: {attrs['max_total_tokens']:,} (max across models)")
    lines.append(f"- **Reference BHC**: {attrs['bhc_word_count']} words")
    st.markdown("\n".join(lines))


def _render_notes_by_category(attrs: dict) -> None:
    """Per-type note counts, the record-count breakdown behind the note total."""
    categories = attrs.get("note_categories", {})
    if not categories:
        return
    st.markdown("**Notes by category**")
    ordered = sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
    table = pd.DataFrame([{"category": cat.strip(), "n": n} for cat, n in ordered])
    st.dataframe(table, hide_index=True, width="content")


def _render_detail_header(hadm_id: int, scored, record) -> None:
    """Render the admission header: hadm_id, cohort attrs, scores, source profile."""
    st.subheader(f"HADM_ID: {hadm_id}")
    attrs = cohort_lookup.get(hadm_id, {})
    parts = [f"{k}: {attrs[k]}" for k in DETAIL_COHORT_ATTRS if attrs.get(k) is not None]
    if parts:
        st.caption(" | ".join(parts))

    col_scores, col_material, col_notes = st.columns([3, 2, 2])
    with col_scores:
        if scored.scores:
            table = scores_to_table(scored.scores)
            st.dataframe(
                table,
                hide_index=True,
                width="content",
                column_config={
                    col: st.column_config.NumberColumn(format="%.4f")
                    for col in table.columns
                    if col != "metric"
                },
            )
        elif record.error:
            st.error(f"Error: {record.error}")
    with col_material:
        _render_source_material(attrs, record)
    with col_notes:
        _render_notes_by_category(attrs)


def _show_detail(hadm_id: int):
    if st.button("\u2190 Back to summary"):
        st.session_state["selected_hadm_id"] = None
        # Clear stale table selection so summary doesn't immediately bounce back.
        st.session_state.pop("summary_table", None)
        # Buttons have no on_change callback, so an explicit rerun is needed
        # to re-dispatch to summary view within the same interaction.
        st.rerun()

    scored = _get_record(str(model_dir), hadm_id)
    if scored is None:
        st.error(f"Record not found for HADM_ID {hadm_id}")
        return
    record = scored.generation

    hadm_ids_key = ",".join(str(h) for h in hadm_ids)
    bhc_refs = _load_bhc_refs(str(BHC_JSONL), hadm_ids_key)
    ref_text = bhc_refs.get(hadm_id, "(Reference BHC not found in bhc.jsonl)")

    _render_detail_header(hadm_id, scored, record)

    col_gen, col_ref = st.columns(2)

    with col_gen:
        gen_text = ""
        llm = record.llm_result
        if llm:
            gen_text = llm.text
            latency_str = f"{llm.latency_ms:.0f}ms"
            parts = [f"{llm.input_tokens} input"]
            if llm.cached_tokens:
                parts.append(f"{llm.cached_tokens} cached")
            parts.append(f"{llm.output_tokens} output")
            parts.append(latency_str)
            # Stats ride the heading line so this column stays one row tall and
            # its text box lines up with the reference box beside it.
            st.markdown(f"**Generated BHC** · {' | '.join(parts)}")
        else:
            st.markdown("**Generated BHC** · (LLM call failed)")
        with st.container(height=600):
            st.markdown(
                f'<div style="{NOTE_STYLE}">{html_escape(gen_text)}</div>',
                unsafe_allow_html=True,
            )

    with col_ref:
        st.markdown("**Reference BHC**")
        with st.container(height=600):
            st.markdown(
                f'<div style="{NOTE_STYLE}">{html_escape(ref_text)}</div>',
                unsafe_allow_html=True,
            )

    with st.expander("Full prompt"):
        st.markdown("**System message**")
        with st.container(height=300):
            st.markdown(
                f'<div style="{NOTE_STYLE}">{html_escape(record.prompt["system"])}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("**User message**")
        with st.container(height=600):
            st.markdown(
                f'<div style="{NOTE_STYLE}">{html_escape(record.prompt["user"])}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Dispatch based on selected_hadm_id
# ---------------------------------------------------------------------------
selected = st.session_state["selected_hadm_id"]
if selected is not None and selected in hadm_ids:
    _show_detail(selected)
else:
    _show_summary()
