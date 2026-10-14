"""Review concept extraction: the prepare-once pipeline's step-3 gate.

Run: uv run streamlit run app/concept_extraction_review.py

Lets a reviewer eyeball extraction quality, for example a drug abbreviation
linked to the wrong UMLS concept. Reads only the frozen prepared-input and
concept artifacts: no scispaCy, no fresh extraction.
"""

import streamlit as st
from _presentation import NOTE_STYLE, html_escape

from bhc_experiment.concept_review import load_concept_review
from bhc_experiment.concept_spaces import concept_params
from bhc_experiment.concept_store import ConceptStore
from bhc_experiment.concepts import Concept
from bhc_experiment.config import DEDUPLICATIONS, MAIN_CONCEPT_SPACE, cohort_file
from bhc_experiment.prepared_store import PreparedStore
from mimic_data_prep.cohort import load_cohort
from mimic_data_prep.mimic import COHORTS_DIR, CONCEPTS_DIR, PREPARED_DIR

st.set_page_config(page_title="Concept Extraction Review", layout="wide")


@st.cache_resource(show_spinner="Opening prepared-input store...")
def _prepared_store(cohort: str, deduplication: str):
    """The prepared store for one (cohort, dedup)."""
    return PreparedStore(PREPARED_DIR, cohort, deduplication)


@st.cache_resource(show_spinner="Opening concept store...")
def _concept_store() -> ConceptStore:
    """The concept store for the main concept space.

    Fixed so a reviewer judges links against one stable extraction config.
    """
    return ConceptStore(CONCEPTS_DIR, concept_params(MAIN_CONCEPT_SPACE))


@st.cache_data
def _hadm_ids(cohort_file: str) -> list[int]:
    """Every admission in the cohort, the same set the prepared store has."""
    return [a.hadm_id for a in load_cohort(COHORTS_DIR / cohort_file).admissions]


def _concept_line(concept: Concept) -> str:
    """One concept as a compact review line: term, CUI, and the matched span."""
    return f"{concept.term} ({concept.cui}) [{concept.start}:{concept.end}]"


def _render_concepts(concepts: list[Concept]) -> None:
    """A concept list as one caption line each, or a placeholder when empty."""
    if not concepts:
        st.caption("(no concepts)")
        return
    for concept in concepts:
        st.caption(_concept_line(concept))


def _render_text_panel(text: str, height: int = 260) -> None:
    """A scrollable monospace panel for one source text."""
    with st.container(height=height):
        st.markdown(f'<div style="{NOTE_STYLE}">{html_escape(text)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: cohort, dedup, admission
# ---------------------------------------------------------------------------
st.sidebar.header("Concept Review")
cohort = st.sidebar.radio("Cohort", ["experiment", "smoke"], horizontal=True)
deduplication = st.sidebar.selectbox(
    "Dedup", DEDUPLICATIONS, index=DEDUPLICATIONS.index("sentence")
)
selected_cohort_file = cohort_file(smoke=cohort == "smoke")

hadm_ids = _hadm_ids(selected_cohort_file)
hadm_id = st.sidebar.selectbox("Admission (HADM_ID)", hadm_ids)

# ---------------------------------------------------------------------------
# Load the stores and the selected admission's concept review
# ---------------------------------------------------------------------------
prepared_store = _prepared_store(selected_cohort_file, deduplication)
concept_store = _concept_store()

review = load_concept_review(prepared_store, concept_store, hadm_id)

st.title(f"Concept extraction review: admission {hadm_id}")
st.caption(f"{selected_cohort_file} | dedup={deduplication} | {len(review.records)} records")

for i, record in enumerate(review.records, 1):
    st.subheader(f"Record {i}/{len(review.records)}")
    text_col, concept_col = st.columns([2, 1])
    with text_col:
        _render_text_panel(record.formatted_text)
    with concept_col:
        _render_concepts(record.concepts)

st.subheader("Reference BHC")
ref_text_col, ref_concept_col = st.columns([2, 1])
with ref_text_col:
    _render_text_panel(prepared_store.bundle(hadm_id).ground_truth_bhc, height=400)
with ref_concept_col:
    _render_concepts(review.reference_concepts)
