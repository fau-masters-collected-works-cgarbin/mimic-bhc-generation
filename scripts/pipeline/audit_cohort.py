"""Per-stratum covariate balance audit for the experiment cohort.

Why: the cohort stratifies on a single variable, admission size in tokens.
Other fields that affect generation quality are recorded per admission
but not balanced at sample time: BHC format, LOS quartile, note-category
availability, mortality, and service mix. These are the covariates. This
audit reports their per-stratum distribution and flags strata where any
major covariate level falls below a 30% representation floor. The 30%
floor is a working cohort-QA threshold for review, chosen by judgment.
Inferential analysis on the covariates is future work.

The pool section shows the same stats for the full filtered pool so a
skewed cohort stratum can be traced back to either an unlucky draw or a
structural skew in the pool itself.

The verdict (PASS or RESAMPLE) is mechanical, but choosing how to
respond to a PASS with structural breaches (accept as-is, note as a
limitation, or narrow per-stratum interpretation) is still a human
judgment.

Questions answered:
- How do the pool and cohort compare stratum by stratum across all
  covariates?
- Does any major covariate level drop below the 30% floor in one or more
  strata?
- How many strata are undersized relative to the per-stratum target?

Inputs:
  cohorts/filtered_pool.json (built by build_filtered_pool.py)
  cohorts/experiment_cohort.json (built by sample_cohort.py)
Output: a per-stratum readout to stdout.
"""

import argparse
import json
import logging
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bhc_experiment.config import EXPERIMENT_COHORT_FILE, FILTERED_POOL_FILE
from mimic_data_prep.cohort import derive_note_category_availability, load_filtered_pool_file
from mimic_data_prep.mimic import COHORTS_DIR
from mimic_data_prep.sampling import TOKEN_STRATA, assign_token_stratum

# 30% representation floor for major covariate levels, a QA threshold
# not a derived minimum.
COVARIATE_FLOOR = 0.30

# Margin on the pool threshold for the structural-vs-sampling split. A
# pool share within this margin of the floor is too close for a
# different seed to robustly clear it: at per-stratum n of 15-25
# admissions, a pool at floor + 5pp yields draws that clear the floor
# only ~50-70% of the time.
RESAMPLE_MARGIN = 0.05

# Covariates checked against the floor, listing only the levels common enough
# for the floor to mean anything.
COVARIATE_LEVELS: dict[str, tuple[str, ...]] = {
    "bhc_category": ("narrative", "domain_organized"),
    "note_category_availability": ("clinical_narrative", "imaging_only"),
}

# Stratum labels are short alphanumerics ("4K", "112K") that don't sort
# by token size as plain strings. Use TOKEN_STRATA's order so tables and
# violation lists read smallest to largest.
_STRATUM_ORDER = {label: i for i, label in enumerate(TOKEN_STRATA)}


def _stratum_sort_key(label: str) -> tuple[int, str]:
    return (_STRATUM_ORDER.get(label, len(_STRATUM_ORDER)), label)


# ---------------------------------------------------------------------------
# Stat computation, shared for pool and cohort admissions
# ---------------------------------------------------------------------------


def _max_narrative_tokens(adm: dict) -> int:
    """Collapse the per-model narrative-token dict to a single int.

    Picks the max across the tokenizers, so the count holds up against the
    model with the smallest context window.
    """
    counts = adm["narrative_record_token_counts"]
    return int(max(counts.values()))


def _quantiles(values: Sequence[int | float]) -> dict[str, float]:
    """Median, Q1, and Q3.

    Uses method='inclusive' to match the cohort's LOS quartile assignment.
    """
    if not values:
        return {"median": 0.0, "q1": 0.0, "q3": 0.0}
    if len(values) == 1:
        v = float(values[0])
        return {"median": v, "q1": v, "q3": v}
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": float(cuts[1]), "q1": float(cuts[0]), "q3": float(cuts[2])}


def summarize_stratum(label: str, admissions: list[dict]) -> dict[str, Any]:
    """Per-covariate distributions for one stratum.

    Works on both pool and cohort admission dicts. Pool rows must have
    note_category_availability and token_stratum derived before calling.
    """
    bhc_mix = Counter(adm["bhc_category"] for adm in admissions)
    nca_mix = Counter(adm["note_category_availability"] for adm in admissions)
    los_days = [adm["los_days"] for adm in admissions]
    note_counts = [adm["note_count"] for adm in admissions]
    narrative_tokens = [_max_narrative_tokens(adm) for adm in admissions]
    deaths = sum(int(adm["hospital_expire_flag"]) for adm in admissions)
    services = Counter(adm.get("service") for adm in admissions)

    total = len(admissions)
    return {
        "stratum": label,
        "n": total,
        "bhc_format_mix": dict(bhc_mix),
        "note_category_availability_mix": dict(nca_mix),
        "los_days": _quantiles(los_days),
        "note_count": _quantiles(note_counts),
        "narrative_token_total": _quantiles(narrative_tokens),
        "mortality": {
            "deaths": deaths,
            "total": total,
            "rate": deaths / total if total else 0.0,
        },
        "service_mix": dict(services),
    }


def compute_all_strata(admissions: list[dict]) -> dict[str, dict]:
    """Summarize every stratum in TOKEN_STRATA, including the empty ones."""
    by_stratum: dict[str, list[dict]] = {label: [] for label in TOKEN_STRATA}
    for adm in admissions:
        # An unknown label means the input was never stratified, so index
        # directly and let it raise.
        by_stratum[adm["token_stratum"]].append(adm)
    return {label: summarize_stratum(label, rows) for label, rows in by_stratum.items()}


# ---------------------------------------------------------------------------
# Pool preparation
# ---------------------------------------------------------------------------


def _prep_pool_admissions(pool_path: Path) -> list[dict]:
    """Load the pool and add the two fields summarize_stratum needs.

    Pool rows have note_categories as a raw dict. Cohort rows store the
    derived note_category_availability label. Deriving it here, along with
    token_stratum, lets summarize_stratum handle both sources the same way.
    """
    df, _ = load_filtered_pool_file(pool_path)
    df["note_category_availability"] = df["note_categories"].map(derive_note_category_availability)
    df = assign_token_stratum(df)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Floor violations and status
# ---------------------------------------------------------------------------


def _mix_key(field: str) -> str:
    return "bhc_format_mix" if field == "bhc_category" else "note_category_availability_mix"


def _pool_share(
    pool_per_stratum: dict[str, dict[str, Any]],
    stratum: str,
    field: str,
    level: str,
) -> float:
    """Pool share at one (stratum, covariate, level) cell."""
    summary = pool_per_stratum[stratum]
    return summary[_mix_key(field)].get(level, 0) / summary["n"]


def _floor_violations(
    per_stratum: dict[str, dict[str, Any]],
    pool_per_stratum: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strata x covariate levels below the floor, classified by kind.

    A violation is ``structural`` when the pool's share at that cell is
    below ``COVARIATE_FLOOR + RESAMPLE_MARGIN``, so no resample of this
    pool can reliably clear the floor. It is ``sampling`` when the pool
    clears that threshold comfortably and only the specific draw
    missed, so a different seed could plausibly fix it.
    """
    sampling_threshold = COVARIATE_FLOOR + RESAMPLE_MARGIN
    violations: list[dict[str, Any]] = []
    for stratum_label, summary in per_stratum.items():
        n = summary["n"]
        if n == 0:
            continue
        for field, levels in COVARIATE_LEVELS.items():
            mix = summary[_mix_key(field)]
            for level in levels:
                share = mix.get(level, 0) / n
                if share < COVARIATE_FLOOR:
                    pool_share = _pool_share(pool_per_stratum, stratum_label, field, level)
                    kind = "structural" if pool_share < sampling_threshold else "sampling"
                    violations.append(
                        {
                            "stratum": stratum_label,
                            "covariate": field,
                            "level": level,
                            "share": share,
                            "n": mix.get(level, 0),
                            "pool_share": pool_share,
                            "kind": kind,
                        }
                    )
    return violations


def _cell_label(violation: dict[str, Any]) -> str:
    """Render a violation cell as ``stratum(level)``.

    Levels are unique strings across the two checked covariates so the
    field name does not need to appear.
    """
    return f"{violation['stratum']}({violation['level']})"


def _violation_sort_key(v: dict[str, Any]) -> tuple[tuple[int, str], str, str]:
    return (_stratum_sort_key(v["stratum"]), v["covariate"], v["level"])


def _status(
    violations: list[dict[str, Any]],
    per_stratum_targets: dict[str, dict[str, int]] | None,
) -> dict[str, Any]:
    """Summarize floor violations and undersized strata, with a pass/resample verdict.

    Verdict logic is mechanical: any sampling-kind violation triggers
    RESAMPLE, since a different seed could plausibly clear the floor.
    Anything without a sampling-kind violation is PASS, because no
    resample of the current pool can fix a pool-level shortfall.
    """
    strata_with_violations = sorted({v["stratum"] for v in violations}, key=_stratum_sort_key)
    structural = sorted(
        (v for v in violations if v["kind"] == "structural"), key=_violation_sort_key
    )
    sampling = sorted((v for v in violations if v["kind"] == "sampling"), key=_violation_sort_key)
    structural_count = len(structural)
    sampling_count = len(sampling)
    undersized: list[tuple[str, int, int]] = []
    if per_stratum_targets:
        for label, counts in per_stratum_targets.items():
            target = int(counts.get("target", 0))
            actual = int(counts.get("actual", 0))
            if target and actual < target:
                undersized.append((label, actual, target))
    undersized.sort(key=lambda u: _stratum_sort_key(u[0]))

    verdict = "RESAMPLE" if sampling_count > 0 else "PASS"
    violation_count = len(violations)

    parts = [f"Verdict: {verdict}."]
    if violation_count == 0 and not undersized:
        parts.append("All covariates >= 30% floor, all strata at target n.")
    else:
        if sampling_count:
            label = "Sampling breach" if sampling_count == 1 else "Sampling breaches"
            cells = ", ".join(_cell_label(v) for v in sampling)
            parts.append(f"{label}: {cells}.")
        if structural_count:
            label = "Structural breach" if structural_count == 1 else "Structural breaches"
            cells = ", ".join(_cell_label(v) for v in structural)
            parts.append(f"{label} (no resample fix): {cells}.")
        if undersized:
            label = "Undersized stratum" if len(undersized) == 1 else "Undersized strata"
            cells = ", ".join(f"{s}({a}/{t})" for s, a, t in undersized)
            parts.append(f"{label} (no resample fix): {cells}.")
    summary_line = " ".join(parts)

    return {
        "verdict": verdict,
        "violation_count": violation_count,
        "structural_count": structural_count,
        "sampling_count": sampling_count,
        "strata_with_violations": strata_with_violations,
        "undersized_strata": [u[0] for u in undersized],
        "summary": summary_line,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _format_stratum_table(
    per_stratum: dict[str, dict],
    header_lines: list[str],
    *,
    targets: dict[str, int] | None = None,
) -> str:
    """Render a per-stratum stats table.

    ``targets`` is None when there is no per-stratum target, and the target
    column then shows a dash.
    """
    col_header = (
        f"{'Stratum':<8}{'N':>5}{'target':>8}{'narr%':>8}{'image%':>8}"
        f"{'narr_fmt%':>11}{'domain%':>10}{'LOS p50':>9}{'mort%':>8}"
    )
    lines = [*header_lines, "", col_header]
    for label, st in per_stratum.items():
        n = max(st["n"], 1)
        narr = st["note_category_availability_mix"].get("clinical_narrative", 0) / n
        img = st["note_category_availability_mix"].get("imaging_only", 0) / n
        narr_fmt = st["bhc_format_mix"].get("narrative", 0) / n
        dom = st["bhc_format_mix"].get("domain_organized", 0) / n
        target_cell = f"{targets[label]:>8}" if targets is not None else f"{'-':>8}"
        lines.append(
            f"{label:<8}{st['n']:>5}{target_cell}{narr * 100:>7.1f}%{img * 100:>7.1f}%"
            f"{narr_fmt * 100:>10.1f}%{dom * 100:>9.1f}%"
            f"{st['los_days']['median']:>9.1f}{st['mortality']['rate'] * 100:>7.1f}%"
        )
    return "\n".join(lines)


def _cohort_targets(summary: dict[str, Any]) -> dict[str, int] | None:
    """Per-stratum targets pulled from the cohort manifest's per_stratum_counts.

    Returns None when the manifest predates per_stratum_counts.
    """
    counts = summary.get("per_stratum_counts")
    if not counts:
        return None
    return {label: int(entry.get("target", 0)) for label, entry in counts.items()}


def _format_text_report(summary: dict[str, Any]) -> str:
    """Render the pool table, the cohort table, and the verdict line."""
    pool_section = _format_stratum_table(
        summary["pool_per_stratum"],
        [f"Pool: {summary['pool_path']}", f"Pool size: {summary['pool_size']}"],
        targets=None,
    )
    cohort_section = _format_stratum_table(
        summary["per_stratum"],
        [f"Cohort: {summary['cohort_path']}", f"Sample size: {summary['sample_size']}"],
        targets=_cohort_targets(summary),
    )
    return "\n".join([pool_section, "", cohort_section, "", summary["status"]["summary"]])


# ---------------------------------------------------------------------------
# Top-level audit
# ---------------------------------------------------------------------------


def audit_cohort(
    cohort_path: Path,
    *,
    pool_path: Path | None = None,
) -> dict[str, Any]:
    """Audit the cohort manifest against the filtered pool.

    Derives pool_path from cohort_path.parent when not provided.
    """
    if pool_path is None:
        pool_path = cohort_path.parent / FILTERED_POOL_FILE

    pool_admissions = _prep_pool_admissions(pool_path)
    pool_per_stratum = compute_all_strata(pool_admissions)

    data = json.loads(cohort_path.read_text())
    admissions = data["admissions"]
    per_stratum = compute_all_strata(admissions)

    violations = _floor_violations(per_stratum, pool_per_stratum)
    status = _status(violations, data["metadata"].get("per_stratum_counts"))

    return {
        "cohort_path": str(cohort_path),
        "pool_path": str(pool_path),
        "sample_size": len(admissions),
        "pool_size": len(pool_admissions),
        "stratification_axes": data["metadata"].get("stratification_axes"),
        "covariate_floor": COVARIATE_FLOOR,
        "pool_per_stratum": pool_per_stratum,
        "per_stratum": per_stratum,
        "per_stratum_counts": data["metadata"].get("per_stratum_counts"),
        "status": status,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=COHORTS_DIR / EXPERIMENT_COHORT_FILE,
        help="Path to the cohort manifest JSON.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    summary = audit_cohort(args.cohort)
    print(_format_text_report(summary))


if __name__ == "__main__":
    main()
