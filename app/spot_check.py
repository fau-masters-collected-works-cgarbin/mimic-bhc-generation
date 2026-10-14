"""Spot-check review app for experiment cohort quality gate.

Run: uv run streamlit run app/spot_check.py
"""

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from _presentation import NOTE_STYLE, html_escape

from bhc_experiment.config import EXPERIMENT_COHORT_FILE
from mimic_data_prep.cohort import load_cohort, total_tokens
from mimic_data_prep.mimic import BHC_JSONL, COHORTS_DIR, MIMIC_DIR, RESULTS_DIR
from mimic_data_prep.spot_check_queries import (
    fetch_bhc_data_batch,
    fetch_demographics_batch,
    fetch_lab_count_batch,
    fetch_notes_batch,
    load_reviews,
    longest_no_note_gap,
    save_review,
    select_review_cases,
)

st.set_page_config(page_title="Spot-Check Review", layout="wide")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COHORT_PATH = COHORTS_DIR / EXPERIMENT_COHORT_FILE
REVIEW_PATH = RESULTS_DIR / "spot_check_reviews.json"
SELECTION_SEED = 99

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BHC_HIGHLIGHT = (
    "background-color: rgba(41, 128, 185, 0.25); border-left: 4px solid #2980b9; padding-left: 8px;"
)
_ERROR_STYLE = (
    "background-color: rgba(231, 76, 60, 0.15); border-left: 4px solid #e74c3c; padding-left: 8px;"
)
_PRE_ADMISSION_STYLE = (
    "background-color: rgba(243, 156, 18, 0.15); border-left: 4px solid #f39c12; padding-left: 8px;"
)

_NOTE_CATEGORY_AVAILABILITY_LABELS = {
    "clinical_narrative": "clin-narr",
    "imaging_only": "img-only",
}

_BHC_CATEGORY_SHORT = {
    "narrative": "narr",
    "domain_organized": "dom",
}

_CATEGORY_ABBREVIATIONS = {"Discharge summary": "DS"}

# Short display labels for the per-model token-count keys defined in
# ``mimic_data_prep.models.MODELS``. A new model needs an entry here.
_MODEL_DISPLAY = {
    "llama_3_2_3b": "llama",
    "qwen_3_5_4b": "qwen",
    "gemma_4_e4b": "gemma",
}


def _max_model_key(record_token_counts: dict[str, dict[str, int]]) -> str:
    """Return the model key whose record total drives ``max_total_tokens``."""
    return max(record_token_counts, key=lambda k: total_tokens(record_token_counts[k]))


def _format_tokens_k(n_tokens: int) -> str:
    """Render an integer token count as a compact ``XX.XK`` label."""
    return f"{n_tokens / 1000:.1f}K"


# ---------------------------------------------------------------------------
# Load cohort and select cases (cached, runs once)
# ---------------------------------------------------------------------------


@st.cache_data
def _select_cases(path_str: str, seed: int) -> list[dict]:
    """Select review cases from the cohort file."""
    cohort = load_cohort(Path(path_str))
    return select_review_cases(
        [asdict(a) for a in cohort.admissions],
        seed=seed,
    )


# Scans each MIMIC CSV once with HADM_ID IN (...) instead of once per
# admission, so load time doesn't scale with cohort size.
@st.cache_data(show_spinner="Loading all admissions (one-time scan)...")
def _fetch_cohort_data(hadm_ids: tuple[int, ...]) -> dict[int, dict]:
    """Fetch per-admission data for every HADM_ID in one pass per CSV."""
    conn = duckdb.connect()
    demos = fetch_demographics_batch(
        conn,
        MIMIC_DIR / "PATIENTS.csv.gz",
        MIMIC_DIR / "ADMISSIONS.csv.gz",
        hadm_ids=hadm_ids,
    )
    bhcs = fetch_bhc_data_batch(conn, BHC_JSONL, MIMIC_DIR / "NOTEEVENTS.csv.gz", hadm_ids=hadm_ids)
    notes = fetch_notes_batch(
        conn,
        MIMIC_DIR / "NOTEEVENTS.csv.gz",
        MIMIC_DIR / "ADMISSIONS.csv.gz",
        hadm_ids=hadm_ids,
    )
    lab_counts = fetch_lab_count_batch(conn, MIMIC_DIR / "LABEVENTS.csv.gz", hadm_ids=hadm_ids)
    return {
        hid: {
            "demo": demos.get(hid),
            "bhc": bhcs.get(hid),
            "notes": notes.get(hid, pd.DataFrame()),
            "lab_count": lab_counts[hid]["total"],
            "lab_abnormal_count": lab_counts[hid]["abnormal"],
        }
        for hid in hadm_ids
    }


cases = _select_cases(str(COHORT_PATH), SELECTION_SEED)
all_case_data = _fetch_cohort_data(tuple(admission["hadm_id"] for admission in cases))

# ---------------------------------------------------------------------------
# Sidebar: case list with review status
# ---------------------------------------------------------------------------
reviews = load_reviews(REVIEW_PATH)
reviewed_ids = set(reviews.get("reviews", {}).keys())
n_reviewed = sum(1 for admission in cases if str(admission["hadm_id"]) in reviewed_ids)

st.sidebar.header(f"Progress: {n_reviewed}/{len(cases)} reviewed")

# Token stratum leads: it is the cohort's sampling axis.
case_labels = []
for admission in cases:
    hadm_id = admission["hadm_id"]
    icon = "\u2705" if str(hadm_id) in reviewed_ids else "\u2b1c"
    bhc_category_short = _BHC_CATEGORY_SHORT[admission["bhc_category"]]
    note_category_availability_label = _NOTE_CATEGORY_AVAILABILITY_LABELS[
        admission["note_category_availability"]
    ]
    case_labels.append(
        f"{icon} {hadm_id} | {admission['token_stratum']} | {bhc_category_short} | "
        f"{note_category_availability_label}"
    )

selected_label = st.sidebar.radio(
    "Select case",
    case_labels,
    index=st.session_state.get("case_idx", 0),
    key="case_radio",
    label_visibility="collapsed",
)
st.session_state["case_idx"] = case_labels.index(selected_label)

# ---------------------------------------------------------------------------
# Current case: load cached data and render header
# ---------------------------------------------------------------------------
case = cases[st.session_state["case_idx"]]
hadm_id = case["hadm_id"]

case_data = all_case_data[hadm_id]
demo = case_data["demo"]
bhc_data = case_data["bhc"]
df_notes = case_data["notes"]
lab_count = case_data["lab_count"]
lab_abnormal_count = case_data["lab_abnormal_count"]

if demo is None:
    st.error(f"Demographics not found for HADM_ID {hadm_id}.")
    st.stop()

if bhc_data is None:
    st.error(f"BHC data not found for HADM_ID {hadm_id}.")
    st.stop()

saved = reviews.get("reviews", {}).get(str(hadm_id))

# ---------------------------------------------------------------------------
# Two-column layout: discharge note (left) | review controls (right)
# ---------------------------------------------------------------------------
col_note, col_review = st.columns([1, 1])

# --- Left column: full discharge note with BHC highlighted ---
with col_note:
    bhc_category = case["bhc_category"]
    note_category_availability_label = _NOTE_CATEGORY_AVAILABILITY_LABELS[
        case["note_category_availability"]
    ]
    # Token count on its own line so a mismatch with the visible text (say, a
    # 112K admission that reads much smaller) is easy to catch before the
    # cohort feeds thousands of calls. Covariates go on a second line so
    # neither wraps.
    max_model = _max_model_key(case["record_token_counts"])
    max_tokens_label = _format_tokens_k(case["max_total_tokens"])
    max_model_label = _MODEL_DISPLAY[max_model]
    st.caption(
        f"HADM {hadm_id} | stratum {case['token_stratum']} "
        f"({max_tokens_label} max via {max_model_label})"
    )
    st.caption(
        f"{bhc_category} | {case['los_quartile']} | {note_category_availability_label} | "
        f"LOS {demo['los_days']}d | {demo['admittime'][:10]} \u2192 {demo['dischtime'][:10]} | "
        f"{demo['gender']} {demo['age']}y"
    )
    st.caption(f"BHC: {bhc_data['bhc_char_count']:,} chars, {bhc_data['bhc_word_count']:,} words")

    note_text = bhc_data["note_text"] or ""
    bhc_start = bhc_data["bhc_start"]
    bhc_end = bhc_data["bhc_end"]

    with st.container(height=800):
        if bhc_start is not None and bhc_end is not None:
            pre = html_escape(note_text[:bhc_start])
            bhc = html_escape(note_text[bhc_start:bhc_end])
            post = html_escape(note_text[bhc_end:])
            st.markdown(
                f'<div style="{NOTE_STYLE}">{pre}'
                f'<div style="{_BHC_HIGHLIGHT}">{bhc}</div>'
                f"{post}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("(BHC offsets not available)")
            st.markdown(
                f'<div style="{NOTE_STYLE}">{html_escape(note_text)}</div>',
                unsafe_allow_html=True,
            )

# --- Right column: checklists, coverage, save ---
with col_review:
    st.markdown("**BHC Coherence**")
    saved_coherence = saved["coherence"] if saved else {}
    coherence_key_prefix = f"coherence_{hadm_id}"
    text_coherent_checked = st.checkbox(
        "Text is coherent (no garbled characters, duplicated paragraphs, or empty template fields)",
        value=saved_coherence.get("text_coherent", False),
        key=f"{coherence_key_prefix}_text_coherent",
    )
    not_truncated_checked = st.checkbox(
        "Not truncated (BHC content is complete, doesn't end mid-sentence)",
        value=saved_coherence.get("not_truncated", False),
        key=f"{coherence_key_prefix}_not_truncated",
    )
    boundaries_correct_checked = st.checkbox(
        "Boundaries correct (highlighted region starts/ends at the right "
        "headers, not content completeness)",
        value=saved_coherence.get("boundaries_correct", False),
        key=f"{coherence_key_prefix}_boundaries_correct",
    )

    if df_notes.empty:
        st.markdown("**Coverage:** no notes found")
    else:
        admit_date = pd.Timestamp(demo["admittime"]).normalize()
        df_notes = df_notes.copy()
        df_notes["hospital_day"] = (pd.to_datetime(df_notes["CHARTDATE"]) - admit_date).dt.days + 1

        # DATE_DIFF counts day boundaries crossed, one fewer than the calendar days present.
        # A patient admitted Jan 1 and discharged Jan 10 has los_days=9 but was
        # present on 10 calendar days. Add 1 so the coverage percentage and the
        # grid reindex use the same stay length.
        stay_days = demo["los_days"] + 1

        coverage_notes = df_notes[
            (df_notes["CATEGORY"] != "Discharge summary")
            & (~df_notes["is_error"])
            & (~df_notes["is_pre_admission"])
        ]
        if not coverage_notes.empty:
            days_with_notes = coverage_notes["hospital_day"].nunique()
            n_categories = coverage_notes["CATEGORY"].nunique()
            pct = round(100 * days_with_notes / stay_days) if stay_days > 0 else 0
            st.markdown(
                f"**Coverage:** {days_with_notes} of {stay_days} days "
                f"({pct}%) | {n_categories} non-DS cat | "
                f"{lab_count:,} labs ({lab_abnormal_count:,} abnormal)"
            )
        else:
            st.markdown(
                f"**Coverage:** no non-DS post-admit notes | "
                f"{lab_count:,} labs ({lab_abnormal_count:,} abnormal)"
            )

        n_error = int(df_notes["is_error"].sum())
        n_pre_adm = int(df_notes["is_pre_admission"].sum())
        if n_error > 0 or n_pre_adm > 0:
            flags = []
            if n_error > 0:
                flags.append(f"{n_error} error")
            if n_pre_adm > 0:
                flags.append(f"{n_pre_adm} pre-admit")
            st.caption(f"\u26a0\ufe0f {', '.join(flags)}")
            flagged = df_notes[df_notes["is_error"] | df_notes["is_pre_admission"]]
            for _, row in flagged.iterrows():
                flag = "ISERROR" if row["is_error"] else "pre-admission"
                st.caption(
                    f"  \u2022 Day {row['hospital_day']} | "
                    f"{row['CATEGORY']} | ROW_ID {row['ROW_ID']} | {flag}"
                )

        if not coverage_notes.empty:
            cat_counts = coverage_notes["CATEGORY"].value_counts().to_dict()
            cat_summary = " \u00b7 ".join(f"{c} {n}" for c, n in cat_counts.items())
            st.caption(cat_summary)

            # Pre-admission notes reach the prompt but fall outside the hospital
            # days, so they get their own count.
            base_chars = int(coverage_notes["note_length"].sum())
            pre_adm_chars = int(
                df_notes[
                    (df_notes["CATEGORY"] != "Discharge summary")
                    & (~df_notes["is_error"])
                    & df_notes["is_pre_admission"]
                ]["note_length"].sum()
            )
            input_str = f"Input: {base_chars:,} chars"
            if pre_adm_chars:
                input_str += f" (+{pre_adm_chars:,} pre-adm)"
            days_set = set(coverage_notes["hospital_day"].dropna().astype(int))
            longest_gap = longest_no_note_gap(days_set, stay_days)
            st.caption(
                f"{input_str} | "
                f"charting {case['charting_days']}/{stay_days} days | "
                f"longest non-DS gap {longest_gap}d | {case['prescription_count']:,} Rx"
            )

            # Pinned to gemma_4_e4b so counts are comparable across cases.
            gemma_record_counts = case["record_token_counts"]["gemma_4_e4b"]
            token_breakdown = " | ".join(f"{n:,} {k}" for k, n in gemma_record_counts.items())
            st.caption(f"Tokens (gemma_4_e4b): {token_breakdown}")

        # Abbreviate long category names so grid columns stay narrow.
        df_grid = df_notes.copy()
        df_grid["CATEGORY"] = df_grid["CATEGORY"].replace(_CATEGORY_ABBREVIATIONS)
        grid = df_grid.pivot_table(
            index="hospital_day",
            columns="CATEGORY",
            values="ROW_ID",
            aggfunc="count",
            fill_value=0,
        )
        # Reindex to include all days in the stay so gaps are visible.
        all_days = range(1, stay_days + 1)
        grid = grid.reindex(all_days, fill_value=0)
        # Convert to string so empty cells are uniform (avoids mixed int/str
        # which breaks Arrow serialization in Streamlit).
        grid = grid.astype(int).astype(str).replace("0", "")
        grid.index.name = "Day"
        st.dataframe(grid, width="stretch")

        with st.expander("Note viewer"):
            available_days = sorted(df_notes["hospital_day"].unique())
            available_cats = sorted(df_notes["CATEGORY"].unique())

            vcols = st.columns(2)
            selected_day = vcols[0].selectbox("Day", available_days, key=f"viewer_day_{hadm_id}")
            selected_cat = vcols[1].selectbox(
                "Category", available_cats, key=f"viewer_cat_{hadm_id}"
            )

            matching = df_notes[
                (df_notes["hospital_day"] == selected_day) & (df_notes["CATEGORY"] == selected_cat)
            ]
            if matching.empty:
                st.caption("No notes for this day and category.")
            else:
                for _, row in matching.iterrows():
                    if row["is_error"]:
                        st.warning("ISERROR note")
                        note_style = _ERROR_STYLE
                    elif row["is_pre_admission"]:
                        st.info("Pre-admission note")
                        note_style = _PRE_ADMISSION_STYLE
                    else:
                        note_style = ""

                    st.caption(f"ROW_ID: {row['ROW_ID']} | {row['note_length']:,} chars")
                    with st.container(height=300):
                        style = f"{NOTE_STYLE} {note_style}" if note_style else NOTE_STYLE
                        st.markdown(
                            f'<div style="{style}">{html_escape(row["TEXT"])}</div>',
                            unsafe_allow_html=True,
                        )

    # The clinical-narrative question is disabled for imaging_only admissions
    # because the relevant notes don't exist there. The imaging question stays
    # enabled because imaging/ECG notes appear in both admission types.
    saved_coverage = saved["coverage"] if saved else {}
    coverage_key_prefix = f"coverage_{hadm_id}"
    note_category_availability = case["note_category_availability"]
    clinical_narrative_applicable = note_category_availability == "clinical_narrative"

    clinical_narrative_spread_checked = st.checkbox(
        "Nursing/Physician notes appear on more than one or two days (not clumped)",
        value=saved_coverage.get("clinical_narrative_spread", False),
        key=f"{coverage_key_prefix}_clinical_narrative_spread",
        disabled=not clinical_narrative_applicable,
        help=None if clinical_narrative_applicable else "Not applicable for this admission",
    )
    imaging_spread_checked = st.checkbox(
        "Imaging/ECG notes appear on more than one or two days (not clumped)",
        value=saved_coverage.get("imaging_spread", False),
        key=f"{coverage_key_prefix}_imaging_spread",
    )

    saved_notes = saved["notes"] if saved else ""
    review_notes = st.text_area(
        "Review notes",
        value=saved_notes,
        key=f"notes_{hadm_id}",
        placeholder="Optional: flag anything notable.",
    )

    if st.button("Save Review", type="primary", key=f"save_{hadm_id}"):
        save_review(
            REVIEW_PATH,
            hadm_id=hadm_id,
            coherence={
                "text_coherent": text_coherent_checked,
                "not_truncated": not_truncated_checked,
                "boundaries_correct": boundaries_correct_checked,
            },
            coverage=(
                {
                    "clinical_narrative_spread": clinical_narrative_spread_checked,
                    "imaging_spread": imaging_spread_checked,
                }
                if clinical_narrative_applicable
                else {"imaging_spread": imaging_spread_checked}
            ),
            notes=review_notes,
            metadata={
                "cohort_file_path": str(COHORT_PATH),
                "selection_seed": SELECTION_SEED,
                "app_script": "app/spot_check.py",
                "created_at": datetime.now(tz=UTC).isoformat(),
            },
        )
        st.success(f"Review saved for HADM_ID {hadm_id}.")
        st.rerun()
