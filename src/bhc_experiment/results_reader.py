"""Read-path for experiment results.

Navigation (finding runs), data loading (parsing JSONL and YAML), and
computation (aggregation, per-admission extraction). Presentation-free:
returns typed dataclasses and DataFrames with no opinion about display.

The write-path (creating runs, appending records) stays in results.py.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from bhc_experiment.analysis import compose_series_labels
from bhc_experiment.config import CliParameters
from bhc_experiment.llm import timing_breakdown
from bhc_experiment.metrics import ROUGEL_F1, SAPBERT_F1, SCISPACY_CLINICAL_F1
from bhc_experiment.prepared_store import PreparedStore
from bhc_experiment.results import (
    OUTLIER_REVIEW_NOTES_FILENAME,
    RUN_RECORD_FILENAME,
    RUN_RESULTS_FILENAME,
    ResultRecord,
    ReviewNote,
    ScoredResult,
    iter_eval_records,
    iter_result_records,
)
from mimic_data_prep.cohort import Cohort, load_cohort
from mimic_data_prep.mimic import COHORTS_DIR, PREPARED_DIR

# Metrics whose max-z composite ranks the outlier worklist. The three primary
# metrics: lexical (ROUGE-L), embedding (SapBERT), and clinical
# concept (SciSpaCy) F1.
PRIMARY_METRICS: tuple[str, ...] = (ROUGEL_F1, SAPBERT_F1, SCISPACY_CLINICAL_F1)


def load_review_notes(run_dir: Path) -> dict[int, ReviewNote]:
    """Load the run-level review notes keyed by hadm_id.

    Returns an empty dict when no notes file exists yet.
    """
    path = run_dir / OUTLIER_REVIEW_NOTES_FILENAME
    if not path.is_file():
        return {}
    records = json.loads(path.read_text())
    return {
        int(r["hadm_id"]): ReviewNote(
            hadm_id=int(r["hadm_id"]),
            tags=list(r.get("tags", [])),
            notes=r.get("notes", ""),
            updated_at=r.get("updated_at", ""),
        )
        for r in records
    }


def load_eval_scores(model_dir: Path) -> dict[int, dict[str, float]]:
    """Load evaluation scores from eval_results.jsonl, one line per admission."""
    return {record.hadm_id: record.scores for record in iter_eval_records(model_dir)}


def discover_model_dirs(results_root: Path, *, prefix: str = "") -> list[Path]:
    """Find model directories (each containing a run record), sorted by name, descending."""
    if not results_root.is_dir():
        return []
    found = [p.parent for p in results_root.rglob(RUN_RECORD_FILENAME) if p.parent != results_root]
    if prefix:
        found = [d for d in found if d.name.startswith(prefix)]
    found.sort(key=lambda d: d.name, reverse=True)
    return found


def model_dirs(run_dir: Path) -> list[Path]:
    """Model directories directly under one run (those with a run record), sorted by name.

    Ascending, unlike ``discover_model_dirs``.
    """
    return [c for c in sorted(Path(run_dir).iterdir()) if (c / RUN_RECORD_FILENAME).is_file()]


def discover_runs(results_root: Path) -> list[Path]:
    """The run directories under results_root, sorted.

    A run is the directory that groups the model directories, the layout's
    ``results_root/<selection>/<prompt>-prompt/<model>``. It is the natural
    unit of analysis for EDA: you compare models within a run.
    """
    if not results_root.is_dir():
        return []
    return sorted({model_dir.parent for model_dir in discover_model_dirs(results_root)})


def load_run(run_dir: Path) -> list[tuple[dict, list[ScoredResult]]]:
    """Load every model under a run as (meta, scored) tuples.

    Models without a ``run_record.yaml`` are skipped silently, so a directory
    created but never executed does not break aggregation.
    """
    if not run_dir.is_dir():
        return []
    return [(load_run_record(d), load_result_records(d)) for d in model_dirs(run_dir)]


def load_run_record(model_dir: Path) -> dict:
    """Read the YAML run record from a model directory.

    Raises FileNotFoundError if the run record file is missing.
    """
    meta_path = model_dir / RUN_RECORD_FILENAME
    return yaml.safe_load(meta_path.read_text())


def _first_model_run_record(run_dir: Path) -> dict:
    """The run record of the first model under a run.

    A run's models share every CLI parameter except ``model``. Raises
    FileNotFoundError when no model has a record.
    """
    dirs = model_dirs(run_dir)
    if not dirs:
        msg = f"no run record found under any model of {run_dir}"
        raise FileNotFoundError(msg)
    return load_run_record(dirs[0])


def run_cli_parameters(run_dir: Path) -> CliParameters:
    """The CLI parameters of the run's first model."""
    return CliParameters(**_first_model_run_record(run_dir)["cli_parameters"])


def cohort_attributes(cohort_file: str) -> dict[int, dict]:
    """Each admission's cohort attributes, keyed by hadm_id."""
    cohort = load_cohort(COHORTS_DIR / cohort_file)
    return {admission.hadm_id: asdict(admission) for admission in cohort.admissions}


def run_prepared_store(run_dir: Path) -> PreparedStore:
    """Every prepared record of the run's cohort and deduplication, before any oracle selection."""
    cli_parameters = run_cli_parameters(run_dir)
    return PreparedStore(PREPARED_DIR, cli_parameters.cohort_file, cli_parameters.deduplication)


def load_result_records(model_dir: Path) -> list[ScoredResult]:
    """Parse the JSONL results file and merge eval scores per admission.

    Returns an empty list if the file is missing or empty. Scores are
    merged from the sibling ``eval_results.jsonl`` when present.

    ``run_results.jsonl`` is append-only, so a resumed run or a manual
    re-run can leave multiple lines for the same admission. Only the
    last occurrence per ``hadm_id`` is returned, matching the keep-last
    semantics of the eval-scores merge.
    """
    records = list(iter_result_records(model_dir))
    if not records:
        return []
    by_hadm: dict[int, ResultRecord] = {}
    for rec in records:
        by_hadm[rec.hadm_id] = rec
    deduped = list(by_hadm.values())
    eval_scores = load_eval_scores(model_dir)
    return [
        ScoredResult(generation=rec, scores=eval_scores.get(rec.hadm_id, {})) for rec in deduped
    ]


def get_result_record(model_dir: Path, hadm_id: int) -> ScoredResult | None:
    """Retrieve the full record for a single hadm_id.

    Returns None if the file is missing or the hadm_id is not found.
    """
    jsonl_path = model_dir / RUN_RESULTS_FILENAME
    if not jsonl_path.exists():
        return None
    id_str = str(hadm_id)
    with jsonl_path.open() as f:
        for line in f:
            if id_str not in line:
                continue
            data = json.loads(line)
            if data["hadm_id"] == hadm_id:
                record = ResultRecord.from_dict(data)
                eval_scores = load_eval_scores(model_dir)
                return ScoredResult(generation=record, scores=eval_scores.get(hadm_id, {}))
    return None


def load_results_df(model_dir: Path) -> pd.DataFrame:
    """Load one model's results into a flat per-admission DataFrame."""
    return build_per_admission_df([(load_run_record(model_dir), load_result_records(model_dir))])


def load_generations(model_dir: Path) -> dict[int, tuple[bool, str]]:
    """Return {hadm_id: (truncated, text)} for one model.

    Truncated generations are included, with the truncation flag. Admissions
    with no llm_result are skipped.
    """
    out: dict[int, tuple[bool, str]] = {}
    for record in iter_result_records(model_dir):
        if record.llm_result is None:
            continue
        out[record.hadm_id] = (record.llm_result.truncated, record.llm_result.text or "")
    return out


def load_reference_bhc(bhc_path: Path, hadm_ids: set[int]) -> dict[int, str]:
    """Load reference BHC text for a set of hadm_ids from bhc.jsonl."""
    if not hadm_ids:
        return {}
    id_strings = {str(hid) for hid in hadm_ids}
    result: dict[int, str] = {}
    with bhc_path.open() as f:
        for line in f:
            if not any(s in line for s in id_strings):
                continue
            rec = json.loads(line)
            if rec["hadm_id"] in hadm_ids:
                result[rec["hadm_id"]] = rec["bhc_text"]
                if len(result) == len(hadm_ids):
                    break
    return result


def extract_run_label(meta: dict) -> tuple[str, str]:
    """Extract (run_name, model_key) from a run record."""
    cli_parameters = CliParameters(**meta["cli_parameters"])
    return cli_parameters.run_name, cli_parameters.model


def _mean_or_none(values: list[float]) -> float | None:
    """Mean of the values, or None when empty, which marks no data."""
    return sum(values) / len(values) if values else None


def _mean_or_zero(values: list[float]) -> float:
    """Mean of the values, or zero when empty (token and latency means)."""
    return sum(values) / len(values) if values else 0


def build_summary_df(models: list[tuple[dict, list[ScoredResult]]]) -> pd.DataFrame:
    """Build a summary DataFrame with one row per model (mean scores).

    Every mean and total covers the records that completed. Errored and truncated
    records are counted in ``n_errors`` and ``n_truncated`` and contribute nothing else.
    """
    rows = []
    for meta, scored in models:
        run_name, model_name = extract_run_label(meta)
        if not scored:
            continue

        all_scores: dict[str, list[float]] = {}
        input_tokens = []
        output_tokens = []
        cached_tokens = []
        latencies = []
        prefill_ms: list[float] = []
        decode_ms: list[float] = []
        prefill_tok_s: list[float] = []
        decode_tok_s: list[float] = []
        n_errors = 0
        n_truncated = 0

        for sr in scored:
            if sr.error:
                n_errors += 1
                continue
            if sr.truncated:
                n_truncated += 1
                continue
            for k, v in sr.scores.items():
                all_scores.setdefault(k, []).append(v)
            llm = sr.llm_result
            if llm:
                input_tokens.append(llm.input_tokens)
                output_tokens.append(llm.output_tokens)
                cached_tokens.append(llm.cached_tokens or 0)
                latencies.append(llm.latency_ms)
                if llm.timings:
                    bd = timing_breakdown(llm.timings)
                    prefill_ms.append(bd["prefill_ms"])
                    decode_ms.append(bd["decode_ms"])
                    prefill_tok_s.append(bd["prefill_tok_s"])
                    decode_tok_s.append(bd["decode_tok_s"])

        row: dict = {
            "run": run_name,
            "model": model_name,
            "n_admissions": len(scored) - n_errors - n_truncated,
            "n_errors": n_errors,
            "n_truncated": n_truncated,
        }
        for k, vals in sorted(all_scores.items()):
            row[k] = sum(vals) / len(vals)
        row.update(
            {
                "input_tokens_mean": _mean_or_zero(input_tokens),
                "output_tokens_mean": _mean_or_zero(output_tokens),
                "cached_tokens_mean": _mean_or_zero(cached_tokens),
                "latency_ms_mean": _mean_or_zero(latencies),
            }
        )
        # A missing timing breakdown means it was not measured, so these means fall
        # back to None.
        row.update(
            {
                "prefill_ms_mean": _mean_or_none(prefill_ms),
                "decode_ms_mean": _mean_or_none(decode_ms),
                "prefill_tok_s_mean": _mean_or_none(prefill_tok_s),
                "decode_tok_s_mean": _mean_or_none(decode_tok_s),
            }
        )
        # "completed" because the loop above skips errored and truncated records, so
        # these sums cover the same records as the means. A truncated generation stops
        # at the output cap, which would make both the mean and the total say more
        # about the cap than about the run.
        row["input_tokens_completed_total"] = sum(input_tokens)
        row["output_tokens_completed_total"] = sum(output_tokens)
        rows.append(row)

    return pd.DataFrame(rows)


def attach_cohort(results_df: pd.DataFrame, cohort: Cohort, *, how: str = "inner") -> pd.DataFrame:
    """Attach cohort attributes to a per-admission results DataFrame.

    Inner join is the default for EDA: stratification needs cohort
    attributes present, so unmatched rows would have NaN in every cohort
    column and break faceting. Pass ``how="left"`` to keep results with no
    cohort match.

    Returns an empty DataFrame when either side has zero rows. Without
    this check, ``pd.merge`` raises ``KeyError: 'hadm_id'`` because an
    empty DataFrame has no columns to join on.
    """
    if results_df.empty or not cohort.admissions:
        return pd.DataFrame()
    cohort_df = pd.DataFrame([asdict(a) for a in cohort.admissions])
    return results_df.merge(cohort_df, on="hadm_id", how=how)


def merge_run_pair(
    base: pd.DataFrame,
    treat: pd.DataFrame,
    *,
    on: Sequence[str] = ("hadm_id", "model"),
    suffixes: tuple[str, str] = ("__base", "__treat"),
) -> pd.DataFrame:
    """Inner-join two per-admission run frames into one paired base/treat frame.

    Drops any admission only one run scored, so the ``__base`` and ``__treat``
    columns cover the same paired set.
    """
    return base.merge(treat, on=list(on), how="inner", suffixes=suffixes)


def load_run_df(run_dir: Path) -> pd.DataFrame:
    """Load a run as a cohort-joined, series-labeled per-admission DataFrame.

    Returns an empty DataFrame when the run is missing or has no results.
    """
    models = load_run(run_dir)
    if not models:
        return pd.DataFrame()
    per_admission = build_per_admission_df(models)
    if per_admission.empty:
        return per_admission
    cohort = load_cohort(COHORTS_DIR / CliParameters(**models[0][0]["cli_parameters"]).cohort_file)
    merged = attach_cohort(per_admission, cohort)
    return compose_series_labels(merged)


def worst_by_composite_z(
    df: pd.DataFrame, n: int, *, metrics: Sequence[str] = PRIMARY_METRICS
) -> list[tuple[int, float, str]]:
    """Rank admissions by composite z-score, worst first, and return the worst n.

    An admission's composite z is its largest gap below the cohort mean across
    ``metrics``, in standard deviations, so the top of the ranking is the
    admissions that fall furthest below the cohort on their weakest metric.

    Models are averaged per admission first. Metrics with no cohort spread are
    dropped. The result is empty if none have spread. Each entry is
    ``(hadm_id, composite_z, driver_metric)``. The driver is the metric that
    set the composite.
    """
    present = [m for m in metrics if m in df.columns]
    if not present:
        return []
    means = df.groupby("hadm_id")[present].mean()
    cohort_std = means.std(ddof=0)
    with_spread = cohort_std[cohort_std > 0].index.tolist()
    if not with_spread:
        return []
    # Cohort mean minus the admission (not the reverse) so a below-cohort
    # admission scores positive: higher z = worse.
    z = (means[with_spread].mean() - means[with_spread]) / cohort_std[with_spread]
    composite = z.max(axis=1)
    ranked = composite.dropna().sort_values(ascending=False).head(n)
    # idxmax only on the kept rows: each has a finite composite, so no all-NaN
    # row reaches idxmax, which a future pandas will raise on.
    driver = z.loc[ranked.index].idxmax(axis=1)
    worst: list[tuple[int, float, str]] = []
    for hadm_id in ranked.index:
        worst.append((int(hadm_id), float(ranked[hadm_id]), str(driver[hadm_id])))
    return worst


def build_per_admission_df(models: list[tuple[dict, list[ScoredResult]]]) -> pd.DataFrame:
    """Build a DataFrame with one row per (model, admission) pair."""
    rows = []
    for meta, scored in models:
        run_name, model_name = extract_run_label(meta)

        for sr in scored:
            row: dict = {
                "run": run_name,
                "model": model_name,
                "hadm_id": sr.hadm_id,
                "error": sr.error,
                "truncated": sr.truncated,
            }
            row |= dict(sorted(sr.scores.items()))
            llm = sr.llm_result
            if llm:
                row["input_tokens"] = llm.input_tokens
                row["output_tokens"] = llm.output_tokens
                row["cached_tokens"] = llm.cached_tokens or 0
                row["latency_ms"] = llm.latency_ms
                row.update(timing_breakdown(llm.timings))
            rows.append(row)

    return pd.DataFrame(rows)
