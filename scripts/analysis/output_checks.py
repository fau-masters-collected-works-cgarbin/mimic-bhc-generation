"""Output checks the prompt test gates read.

Questions answered:

- Per model, how many outputs restate the section title on their first line?
- How many standalone lines say information is missing from the records?
- How many outputs use header lines?

The prompt test compares these counts with the previous prompt test on the same
admissions (see "Testing a prompt change" in docs/specs/prompt-builder-design.md).
Every generated output is checked, truncated ones included, because a truncated
output can still echo the title or write a missing-information line.

Usage:

    uv run python scripts/analysis/output_checks.py <run-folder>
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

from bhc_experiment import reporting
from bhc_experiment.results import RUN_RESULTS_FILENAME
from bhc_experiment.results_reader import model_dirs

logger = logging.getLogger(__name__)

REPORT_NAME = "output_checks"
_TITLE = re.compile(r"brief hospital course", re.IGNORECASE)
_MISSING = re.compile(
    r"not (documented|specified|mentioned|available)|no information", re.IGNORECASE
)
# A missing-information phrase inside a sentence this long is inline phrasing,
# which the disposition rule does not target.
_STANDALONE_MAX_WORDS = 12


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def title_echo(text: str) -> bool:
    """True when the first nonblank line names the section."""
    lines = _lines(text)
    return bool(lines) and bool(_TITLE.search(lines[0]))


def missing_information_lines(text: str) -> int:
    """Standalone lines saying information is missing from the records."""
    count = 0
    for line in _lines(text):
        if _MISSING.search(line) and len(line.split()) < _STANDALONE_MAX_WORDS:
            count += 1
    return count


def header_lines(text: str) -> int:
    """Lines that start with `#` or are bold text alone."""
    count = 0
    for line in _lines(text):
        if line.startswith("#") or (line.startswith("**") and line.endswith("**")):
            count += 1
    return count


def model_checks(model_dir: Path) -> dict[str, int]:
    """The check counts over one model's generated outputs."""
    checks = {"outputs": 0, "truncated": 0, "title_echoes": 0, "missing_info_lines": 0}
    checks["outputs_with_headers"] = 0
    with (model_dir / RUN_RESULTS_FILENAME).open() as results_file:
        for line in results_file:
            llm_result = json.loads(line).get("llm_result")
            if llm_result is None:
                continue
            text = llm_result.get("text") or ""
            checks["outputs"] += 1
            checks["truncated"] += bool(llm_result.get("truncated"))
            checks["title_echoes"] += title_echo(text)
            checks["missing_info_lines"] += missing_information_lines(text)
            checks["outputs_with_headers"] += header_lines(text) > 0
    return checks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="path to the experiment run folder")
    args = parser.parse_args()

    rows = [{"model": d.name, **model_checks(d)} for d in model_dirs(args.run_dir)]
    table = reporting.markdown_table(
        pd.DataFrame(rows),
        right_align=(
            "outputs",
            "truncated",
            "title_echoes",
            "missing_info_lines",
            "outputs_with_headers",
        ),
    )
    body = (
        "# Output checks\n\n"
        f"_Run: `{args.run_dir.name}`_\n\n"
        "Counts over every generated output, truncated ones included. A title echo is a "
        "first line naming the section. A missing-information line is a line of fewer than "
        f"{_STANDALONE_MAX_WORDS} words saying information is not documented, specified, "
        "mentioned or available.\n\n"
        f"{table}\n"
    )
    report_path = reporting.write_report(args.run_dir, REPORT_NAME, body)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
