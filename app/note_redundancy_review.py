"""Sentence-dedup review: the prepare-once pipeline's step-3 gate.

Run: uv run streamlit run app/note_redundancy_review.py

Confirms every sentence dedup removed had a surviving identical copy. Reads only the
frozen prepared artifact: no duckdb, no re-segmentation, no
re-classification.
"""

import streamlit as st
from _presentation import NOTE_STYLE, html_escape

from bhc_experiment.config import cohort_file
from bhc_experiment.note_review import NoteRemovals, RemovedSentence, load_note_removals
from bhc_experiment.prepared_store import PreparedStore
from mimic_data_prep.cohort import load_cohort
from mimic_data_prep.mimic import COHORTS_DIR, PREPARED_DIR

st.set_page_config(page_title="Note Redundancy Review", layout="wide")

# Sentence dedup is the only method in the recipe: removals exist only under it.
_DEDUPLICATION = "sentence"

# Within-note and between-note get distinct colors so the two removal reasons
# are separable at a glance.
_WITHIN_STYLE = (
    "background-color: rgba(243,156,18,0.25); border-left: 4px solid #f39c12; padding: 1px 6px;"
)
_BETWEEN_STYLE = (
    "background-color: rgba(41,128,185,0.22); border-left: 4px solid #2980b9; padding: 1px 6px;"
)
_TAG_BADGE = (
    "display:inline-block; color:#fff; font-size:12px; font-weight:700; "
    "padding:1px 8px; border-radius:8px; margin-left:8px; vertical-align:middle;"
)
_SENT_STYLE = {"within": _WITHIN_STYLE, "between": _BETWEEN_STYLE}
_BADGE_COLOR = {"within": "#f39c12", "between": "#2980b9"}


@st.cache_resource(show_spinner="Opening prepared-input store...")
def _prepared_store(cohort: str):
    """Open the sentence-dedup prepared store for one cohort."""
    return PreparedStore(PREPARED_DIR, cohort, _DEDUPLICATION)


@st.cache_data
def _hadm_ids(cohort_file: str) -> list[int]:
    """Return every admission in the cohort (the same set the prepared store has)."""
    return [a.hadm_id for a in load_cohort(COHORTS_DIR / cohort_file).admissions]


def _note_caption(entry: NoteRemovals) -> str:
    """Build the one-line metadata caption shown above a note's text."""
    note = entry.note
    when = note.CHARTTIME or note.CHARTDATE or "no-time"
    kept = len(note.TEXT or "")
    original = kept + sum(len(s.text) for s in entry.removed)
    return (
        f"ROW {note.ROW_ID} | {note.CATEGORY} | {when} | "
        f"{kept:,} of {original:,} chars kept | {len(entry.removed)} sentences removed"
    )


def _note_box(entry: NoteRemovals, height: int = 260) -> None:
    st.caption(_note_caption(entry))
    with st.container(height=height):
        st.markdown(
            f'<div style="{NOTE_STYLE}">{html_escape(entry.note.TEXT or "")}</div>',
            unsafe_allow_html=True,
        )


def _removed_div(sentence: RemovedSentence) -> str:
    """Render one removed sentence as HTML, highlighted and badged with its within/between kind."""
    text = html_escape(" ".join(sentence.text.split()))
    style = _SENT_STYLE[sentence.kind]
    color = _BADGE_COLOR[sentence.kind]
    badge = f'<span style="{_TAG_BADGE} background:{color};">{sentence.kind}</span>'
    return f'<div style="{style}">{text}{badge}</div>'


# ---------------------------------------------------------------------------
# Sidebar: cohort, admission (heaviest sentence removal first)
# ---------------------------------------------------------------------------
st.sidebar.header("Note Redundancy Review")
cohort = st.sidebar.radio("Cohort", ["experiment", "smoke"], horizontal=True)
selected_cohort_file = cohort_file(smoke=cohort == "smoke")

prepared_store = _prepared_store(selected_cohort_file)
hadm_ids = _hadm_ids(selected_cohort_file)


def _removed_count(hadm_id: int) -> int:
    return sum(len(v) for v in prepared_store.removed_sentences(hadm_id).values())


counts = {hadm_id: _removed_count(hadm_id) for hadm_id in hadm_ids}
ordered = sorted(hadm_ids, key=lambda hadm_id: counts[hadm_id], reverse=True)
labels = {
    f"{'●' if counts[hadm_id] else '○'} {hadm_id} | {counts[hadm_id]} sentences removed": hadm_id
    for hadm_id in ordered
}
selected_label = st.sidebar.selectbox("Select admission", list(labels.keys()))
hadm_id = labels[selected_label]
hide_clean_notes = st.sidebar.checkbox("Hide notes with no removals", value=False)

# ---------------------------------------------------------------------------
# Body: each surviving note with its removed sentences
# ---------------------------------------------------------------------------
removals = load_note_removals(prepared_store, hadm_id)

st.title(f"Sentence dedup: admission {hadm_id}")
st.caption(f"{selected_cohort_file} | {len(removals)} surviving notes")
st.caption(
    "Highlighted sentences were cut as duplicates: orange within-note, blue between-note. "
    "The note text shown is what survives after dedup."
)

clean: list[str] = []
for entry in removals:
    if hide_clean_notes and not entry.removed:
        clean.append(f"ROW {entry.note.ROW_ID} ({entry.note.CATEGORY})")
        continue
    _note_box(entry)
    if entry.removed:
        body = "".join(_removed_div(s) for s in entry.removed)
        with st.container(height=200):
            st.markdown(f'<div style="{NOTE_STYLE}">{body}</div>', unsafe_allow_html=True)

if clean:
    st.caption("Notes with no removals: " + ", ".join(clean))
