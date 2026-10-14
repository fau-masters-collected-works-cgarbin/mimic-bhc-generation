"""Results viewer: entry point with run selector and page navigation.

Run: uv run streamlit run app/results_app.py
"""

from pathlib import Path

import streamlit as st

from bhc_experiment.results import RUN_RECORD_FILENAME
from bhc_experiment.results_reader import discover_model_dirs
from mimic_data_prep.mimic import RESULTS_DIR

st.set_page_config(page_title="Results Viewer", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: run selection
# ---------------------------------------------------------------------------
RESULTS_ROOT = RESULTS_DIR

st.sidebar.header("Run Selection")

prefix_filter = st.sidebar.text_input(
    "Filter by model",
    value="",
    placeholder="e.g. gemma_4_e4b",
    key="run_prefix_filter",
)

if "force_rediscover" not in st.session_state:
    st.session_state["force_rediscover"] = 0


@st.cache_data
def _discover(results_root_str: str, prefix: str, _version: int):
    return discover_model_dirs(Path(results_root_str), prefix=prefix)


found_model_dirs = _discover(str(RESULTS_ROOT), prefix_filter, st.session_state["force_rediscover"])


available_selections = sorted({d.parent.parent.name for d in found_model_dirs})
selection_filter = st.sidebar.selectbox(
    "Filter by selection",
    options=["(all)", *available_selections],
    key="selection_filter",
)

if st.sidebar.button("Refresh run list"):
    st.session_state["force_rediscover"] += 1
    st.rerun()

model_choices: dict[str, Path] = {}
for model_dir in found_model_dirs:
    if selection_filter not in ("(all)", model_dir.parent.parent.name):
        continue
    model_choices[f"{model_dir.parent.relative_to(RESULTS_ROOT)} | {model_dir.name}"] = model_dir

selected_model_dir: Path | None = None

if model_choices:
    selected_label = st.sidebar.selectbox(
        "Select model",
        options=list(model_choices.keys()),
        key="model_selector",
    )
    if selected_label:
        selected_model_dir = model_choices[selected_label]
else:
    st.sidebar.info(f"No models found in {RESULTS_ROOT}")

# Direct path override (for debugging or accessing models outside the results root).
st.sidebar.divider()
direct_path = st.sidebar.text_input(
    "Or enter a model directory path",
    value="",
    placeholder="/path/to/<selection>/<prompt>-prompt/<model>",
    key="direct_model_path",
)
if direct_path.strip():
    candidate = Path(direct_path.strip())
    if (candidate / RUN_RECORD_FILENAME).exists():
        selected_model_dir = candidate
    else:
        st.sidebar.error(f"Directory does not contain {RUN_RECORD_FILENAME}")

st.session_state["selected_model_dir"] = selected_model_dir

# ---------------------------------------------------------------------------
# Page navigation
# ---------------------------------------------------------------------------
pages = st.navigation(
    [
        st.Page("results_explorer.py", title="Run Explorer", icon="🔬"),
    ]
)
pages.run()
