"""Experiment cohort sampling logic and admission-list file writers.

Splits the filtered pool into LOS quartile bins, samples a balanced cohort
across the token strata, and writes the filtered-pool and experiment-cohort
JSON files.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mimic_data_prep.cohort import compute_mesh_age_group

logger = logging.getLogger(__name__)


# Edges are upper bounds in tokens. The pool filter caps admissions at
# 112K, so 112K is also the largest possible value.
TOKEN_STRATA: tuple[str, ...] = ("4K", "8K", "16K", "32K", "64K", "96K", "112K")
TOKEN_STRATUM_EDGES: tuple[int, ...] = (4_000, 8_000, 16_000, 32_000, 64_000, 96_000, 112_000)


def assign_token_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each admission to one of the token strata.

    Requires a ``max_total_tokens`` column.
    Bins are left-open and right-closed, so an admission at exactly 4,000
    tokens goes into the 4K stratum and 4,001 goes into the 8K stratum.

    Raises ``ValueError`` if any admission exceeds 112K.
    """
    over = df["max_total_tokens"] > TOKEN_STRATUM_EDGES[-1]
    if over.any():
        msg = (
            f"{int(over.sum())} admission(s) exceed the 112K pool budget. "
            "The pool filter must be applied before stratification."
        )
        raise ValueError(msg)
    # The -inf sentinel lets the smallest stratum also capture zero and
    # any positive count below the first edge.
    bins = [float("-inf"), *TOKEN_STRATUM_EDGES]
    df = df.copy()
    df["token_stratum"] = pd.cut(
        df["max_total_tokens"], bins=bins, labels=list(TOKEN_STRATA), right=True
    )
    return df


def assign_los_quartiles(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Assign each row to a LOS quartile bin (Q1-Q4).

    Bins are left-inclusive: [min, P25), [P25, P50), [P50, P75), [P75, max].

    Returns the DataFrame with a new 'los_quartile' column and the
    [P25, P50, P75] edges.
    """
    edges = df["los_days"].quantile([0.25, 0.5, 0.75]).tolist()
    # -inf/inf avoid assumptions about the data range.
    bins = [float("-inf"), edges[0], edges[1], edges[2], float("inf")]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    df["los_quartile"] = pd.cut(df["los_days"], bins=bins, labels=labels, right=False)
    return df, edges


def sample_stratified(
    df: pd.DataFrame,
    *,
    grouping_columns: list[str],
    samples_per_group: int | dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """Sample a balanced number of admissions from each cell of a multi-column grid.

    Each cell is a unique combination of values across `grouping_columns`.
    `samples_per_group` is an int for a uniform target across all cells, or a
    dict keyed by the grouping-column value for an asymmetric per-stratum
    target (e.g., the token-strata allocation).

    If a cell has fewer than its target rows, all rows are taken and a
    warning is logged, for both the int and dict paths.

    Returns a DataFrame with the sampled rows, sorted by hadm_id.
    """
    samples = []
    # observed=True avoids pandas enumerating every category combination,
    # including empty ones, when a grouping column is a Categorical (e.g.,
    # los_quartile from pd.cut). Enumerating them otherwise fails at sampling.
    for cell_key, group in df.groupby(grouping_columns, observed=True):
        # groupby returns a 1-tuple when grouping on one column. Unwrap it so the
        # dict lookup can use the bare label.
        lookup_key = cell_key[0] if isinstance(cell_key, tuple) and len(cell_key) == 1 else cell_key
        target = (
            samples_per_group[lookup_key]
            if isinstance(samples_per_group, dict)
            else samples_per_group
        )
        n_available = len(group)
        n_sample = min(target, n_available)
        if n_available < target:
            logger.warning(
                "Cell %s has %d admissions, fewer than target %d. Taking all.",
                lookup_key,
                n_available,
                target,
            )
        sampled = group.sample(n=n_sample, random_state=seed)
        samples.append(sampled)
    return pd.concat(samples).sort_values("hadm_id").reset_index(drop=True)


def write_cohort_file(
    df: pd.DataFrame,
    *,
    path: Path,
    seed: int,
    stratification_axes: list[str],
    los_quartile_edges: list[float],
    cohort_size: int,
    token_strata: dict,
    per_stratum_counts: dict[str, dict[str, int]],
    bhc_jsonl_path: str,
    bhc_jsonl_sha256: str,
    bhc_extraction_commit: tuple[str, str],
    sampling_script_commit: tuple[str, str],
    tokenizer: dict,
    pool_file: dict,
) -> None:
    """Write the cohort file as JSON with full provenance for reproducibility.

    Args:
        df: Sampled admissions from ``filtered_pool`` with token_stratum,
            LOS quartile, and note-category-availability covariates attached.
        los_quartile_edges: [P25, P50, P75] boundaries used to assign the
            ``los_quartile`` covariate, which is reported per admission and
            never sampled on.
        cohort_size: Total size of the filtered pool the sample was drawn from.
        token_strata: The seven stratum ``labels`` and ``edges`` plus the
            binning rule, captured verbatim.
        per_stratum_counts: ``{label: {target, actual}}`` per stratum. The
            asymmetric allocation gives each stratum its own target, and
            the long-tail strata (notably 4K, structurally underflowing
            in the pool) record actual < target here.
        tokenizer: Tokenizer block forwarded from the pool manifest.
        pool_file: ``{path, sha256}`` of the filtered-pool JSON the cohort
            was drawn from.
    """
    cols = [
        "hadm_id",
        "subject_id",
        "bhc_category",
        "token_stratum",
        "los_quartile",
        "los_days",
        "note_category_availability",
        "bhc_numbering_convention",
        "bhc_modifiers",
        "note_count",
        "note_categories",
        "lab_count",
        "lab_abnormal_count",
        "prescription_count",
        "bhc_word_count",
        "bhc_token_counts",
        "charting_days",
        "record_token_counts",
        "max_total_tokens",
        "narrative_record_token_counts",
        "hospital_expire_flag",
        "temporal_density",
        "timestamp_completeness",
        "source_diversity",
        "pre_admission_notes",
        "gender",
        "age",
        "service",
        "admittime",
        "dischtime",
    ]
    admissions = df[cols].sort_values("hadm_id").to_dict(orient="records")
    for row in admissions:
        # Convert pandas and numpy types to plain Python so json.dumps
        # produces a clean record with no numpy.int64 values in it.
        row["hadm_id"] = int(row["hadm_id"])
        row["los_days"] = int(row["los_days"])
        row["los_quartile"] = str(row["los_quartile"])
        row["token_stratum"] = str(row["token_stratum"])
        # bhc_numbering_convention is None for narrative BHCs, string otherwise.
        # pandas may store None as float NaN for object columns, so normalize to None.
        nc = row["bhc_numbering_convention"]
        row["bhc_numbering_convention"] = None if pd.isna(nc) else str(nc)
        row["bhc_modifiers"] = list(row["bhc_modifiers"])
        for int_col in (
            "subject_id",
            "age",
            "note_count",
            "lab_count",
            "lab_abnormal_count",
            "prescription_count",
            "bhc_word_count",
            "charting_days",
            "max_total_tokens",
            "hospital_expire_flag",
        ):
            row[int_col] = int(row[int_col])
        row["gender"] = str(row["gender"])
        # service is None for admissions without a SERVICES row, string otherwise.
        svc = row["service"]
        row["service"] = None if pd.isna(svc) else str(svc)
        row["admittime"] = str(row["admittime"])
        row["dischtime"] = str(row["dischtime"])
        # Computed here so callers do not have to pre-populate mesh_age_group
        # in the input DataFrame.
        row["mesh_age_group"] = compute_mesh_age_group(row["age"])

    cohort_data = {
        "metadata": {
            "seed": seed,
            "sample_size": len(df),
            "stratification_axes": stratification_axes,
            "token_strata": token_strata,
            "per_stratum_counts": per_stratum_counts,
            "los_quartile_edges": los_quartile_edges,
            "cohort_size": cohort_size,
            "data_sources": {
                "mimic": "MIMIC-III v1.4",
                "bhc_jsonl": {"path": bhc_jsonl_path, "sha256": bhc_jsonl_sha256},
                "bhc_extraction_repo": {
                    "commit": bhc_extraction_commit[0],
                    "subject": bhc_extraction_commit[1],
                },
                "sampling_script_repo": {
                    "commit": sampling_script_commit[0],
                    "subject": sampling_script_commit[1],
                },
                "pool_file": pool_file,
            },
            "tokenizer": tokenizer,
            "script": "scripts/pipeline/sample_cohort.py",
            "created_at": datetime.now(tz=UTC).isoformat(),
        },
        "admissions": admissions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cohort_data, indent=2) + "\n")


def write_filtered_pool_file(
    df: pd.DataFrame,
    *,
    path: Path,
    bhc_jsonl_path: str,
    bhc_jsonl_sha256: str,
    bhc_extraction_commit: tuple[str, str],
    build_script_commit: tuple[str, str],
    pool_definition: dict,
    tokenizer: dict,
    pool_filter_stats: dict[str, int],
) -> None:
    """Write the filtered pool to JSON.

    Schema: a metadata block with provenance and pool/tokenizer definitions,
    plus an ``admissions`` array sorted by ``hadm_id``. Each admission
    has only record-level fields, with no stratification. Stratification
    happens at sampling time from this file.

    Args:
        pool_filter_stats: ``before_token_budget`` (admissions passing the
            pool filters in ``create_filtered_pool``, before the token-budget
            cap) and ``dropped_above_token_budget`` (admissions cut by the cap).
    """
    cols = [
        "hadm_id",
        "subject_id",
        "bhc_char_count",
        "bhc_word_count",
        "bhc_token_counts",
        "bhc_category",
        "bhc_numbering_convention",
        "bhc_modifiers",
        "los_days",
        "note_count",
        "charting_days",
        "lab_count",
        "lab_abnormal_count",
        "prescription_count",
        "admittime",
        "dischtime",
        "hospital_expire_flag",
        "gender",
        "age",
        "mesh_age_group",
        "service",
        "record_token_counts",
        "max_total_tokens",
        "narrative_record_token_counts",
        "temporal_density",
        "timestamp_completeness",
        "source_diversity",
        "pre_admission_notes",
        "note_categories",
    ]
    admissions = df[cols].sort_values("hadm_id").to_dict(orient="records")
    for row in admissions:
        # numpy.int64 and friends are not JSON-serializable.
        for int_col in (
            "hadm_id",
            "subject_id",
            "bhc_char_count",
            "bhc_word_count",
            "los_days",
            "note_count",
            "charting_days",
            "lab_count",
            "lab_abnormal_count",
            "prescription_count",
            "hospital_expire_flag",
            "age",
            "max_total_tokens",
        ):
            row[int_col] = int(row[int_col])
        row["bhc_category"] = str(row["bhc_category"])
        # bhc_numbering_convention is None for narrative BHCs, string otherwise.
        nc = row["bhc_numbering_convention"]
        row["bhc_numbering_convention"] = None if pd.isna(nc) else str(nc)
        # DuckDB returns LIST columns as arrays. Make it a plain Python list.
        row["bhc_modifiers"] = list(row["bhc_modifiers"])
        row["admittime"] = str(row["admittime"])
        row["dischtime"] = str(row["dischtime"])
        row["gender"] = str(row["gender"])
        row["mesh_age_group"] = str(row["mesh_age_group"])
        svc = row["service"]
        row["service"] = None if pd.isna(svc) else str(svc)

    pool_data = {
        "metadata": {
            "script": "scripts/pipeline/build_filtered_pool.py",
            "script_commit": {
                "hash": build_script_commit[0],
                "subject": build_script_commit[1],
            },
            "created_at": datetime.now(tz=UTC).isoformat(),
            "pool_size": len(df),
            "pool_filter_stats": pool_filter_stats,
            "data_sources": {
                "mimic": "MIMIC-III v1.4",
                "bhc_jsonl": {"path": bhc_jsonl_path, "sha256": bhc_jsonl_sha256},
                "bhc_extraction_repo": {
                    "commit": bhc_extraction_commit[0],
                    "subject": bhc_extraction_commit[1],
                },
            },
            "pool_definition": pool_definition,
            "tokenizer": tokenizer,
        },
        "admissions": admissions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool_data, indent=2) + "\n")
