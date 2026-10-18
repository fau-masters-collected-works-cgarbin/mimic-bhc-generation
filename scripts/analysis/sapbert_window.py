"""How much of each text falls outside SapBERT's window.

Questions answered:

- What share of references and generated outputs is longer than the window
  SapBERT scores, per reference category?

The evaluator truncates SapBERT input at 512 wordpieces (see evaluator.py),
two of them special tokens, so text past 510 wordpieces is never scored. A
prompt or model that writes longer outputs loses more of its text to the cut,
which biases SapBERT comparisons between runs of different length. Truncated
generations have no scores and are left out.

Usage:

    uv run python scripts/analysis/sapbert_window.py <run-folder>
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from bhc_experiment import reporting
from bhc_experiment.config import EXPERIMENT_COHORT_FILE, METRICS
from bhc_experiment.results import RUN_RESULTS_FILENAME
from bhc_experiment.results_reader import cohort_attributes, load_reference_bhc, model_dirs
from mimic_data_prep.mimic import BHC_JSONL

logger = logging.getLogger(__name__)

REPORT_NAME = "sapbert_window"
# 512 positions less the [CLS] and [SEP] tokens bert_score adds.
CONTENT_LIMIT = 510
SAPBERT_MODEL = next(
    metric["bertscore"]["model_type"]
    for metric in METRICS
    if metric.get("bertscore", {}).get("name") == "sapbert"
)


def window_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Per source and reference category: n, median wordpieces, and the share past the limit."""
    rows = []
    for (source, category), group in counts.groupby(["source", "bhc_category"]):
        over = int((group["wordpieces"] > CONTENT_LIMIT).sum())
        rows.append(
            {
                "source": source,
                "bhc_category": category,
                "n": len(group),
                "median_wordpieces": group["wordpieces"].median(),
                "over_window": over,
                "share_over_window": over / len(group),
            }
        )
    return pd.DataFrame(rows)


def _scored_texts(model_dir: Path) -> dict[int, str]:
    """Each untruncated output's text, by admission."""
    texts = {}
    with (model_dir / RUN_RESULTS_FILENAME).open() as results_file:
        for line in results_file:
            result_record = json.loads(line)
            llm_result = result_record.get("llm_result")
            if llm_result and not llm_result.get("truncated"):
                texts[result_record["hadm_id"]] = llm_result.get("text") or ""
    return texts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="path to the experiment run folder")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL)
    attributes = cohort_attributes(EXPERIMENT_COHORT_FILE)
    texts_by_model = {d.name: _scored_texts(d) for d in model_dirs(args.run_dir)}
    hadm_ids = set().union(*texts_by_model.values())
    references = load_reference_bhc(BHC_JSONL, hadm_ids)

    rows = []
    for hadm_id in sorted(hadm_ids):
        category = attributes[hadm_id]["bhc_category"]
        rows.append(
            {
                "source": "reference",
                "bhc_category": category,
                "wordpieces": len(tokenizer.tokenize(references[hadm_id])),
            }
        )
    for model, texts in texts_by_model.items():
        for hadm_id, text in texts.items():
            category = attributes[hadm_id]["bhc_category"]
            rows.append(
                {
                    "source": model,
                    "bhc_category": category,
                    "wordpieces": len(tokenizer.tokenize(text)),
                }
            )

    table = reporting.markdown_table(
        window_summary(pd.DataFrame(rows)),
        float_format="{:.2f}",
        right_align=("n", "median_wordpieces", "over_window", "share_over_window"),
    )
    body = (
        "# SapBERT window\n\n"
        f"_Run: `{args.run_dir.name}`_\n\n"
        f"SapBERT scores the first {CONTENT_LIMIT} wordpieces of each text. The share over "
        "the window is the share of texts with some content SapBERT never sees. Truncated "
        "generations are left out.\n\n"
        f"{table}\n"
    )
    report_path = reporting.write_report(args.run_dir, REPORT_NAME, body)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
