"""Cohort admissions as Python values.

- Pool loader: builds the filtered pool as a DataFrame on top of ``mimic.py``,
  and reads a saved pool back from JSON.
- Per-admission metrics computed during that build: token counts per record
  type, narrative tokens, temporal density, source diversity, and
  pre-admission notes.
- Token budget sums over the per-model record counts.
- Derived attributes: MeSH age group and note category availability, shared by
  the pool build and the cohort write.
- Cohort file reader: reads a cohort JSON file into typed values.
"""

import json
import logging
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pandas as pd

from mimic_data_prep.mimic import (
    create_filtered_pool,
    note_order_sql,
    noteevents_input_filter,
    table_ref,
)
from mimic_data_prep.models import MODEL_KEYS
from mimic_data_prep.tokenizers import count_per_model

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _default_n_workers() -> int:
    """Default tokenizer worker count: leave one core for the orchestrator.

    ``os.cpu_count()`` can return None in unusual environments, so we
    coerce that to 1.
    """
    return max(1, (os.cpu_count() or 1) - 1)


# Record components that sum to one model's total token count.
# ``labs_abnormal`` is deliberately absent: it counts a subset of the same
# LABEVENTS rows as ``labs_all`` (only those with ``FLAG = 'abnormal'``), so
# summing both would double-count the abnormal labs.
_TOTAL_TOKENS_COMPONENTS: tuple[str, ...] = (
    "notes",
    "labs_all",
    "prescriptions",
    "demographics",
)


def total_tokens(record_token_counts: "dict[str, int]") -> int:
    """Total tokens the generation input uses.

    The argument is one model's counts, so the total is specific to that model.
    """
    return sum(int(record_token_counts[k]) for k in _TOTAL_TOKENS_COMPONENTS)


def max_total_tokens(
    per_model_counts: "dict[str, dict[str, int]]",
) -> int:
    """Total tokens for the model with the largest sum.

    The maximum is the conservative pick: if the largest model's total fits
    the budget, every model fits.
    """
    return max(total_tokens(rtc) for rtc in per_model_counts.values())


@dataclass(frozen=True)
class Admission:
    """One admission's cohort record."""

    hadm_id: int
    subject_id: int
    bhc_category: str
    token_stratum: str
    los_quartile: str
    los_days: int
    note_category_availability: str
    bhc_numbering_convention: str | None
    bhc_modifiers: list[str]
    note_count: int
    note_categories: dict[str, int]
    lab_count: int
    lab_abnormal_count: int
    prescription_count: int
    bhc_word_count: int
    bhc_token_counts: dict[str, int]
    charting_days: int
    record_token_counts: dict[str, dict[str, int]]
    max_total_tokens: int
    narrative_record_token_counts: dict[str, int]
    hospital_expire_flag: int
    temporal_density: dict
    timestamp_completeness: dict
    source_diversity: dict
    pre_admission_notes: dict
    gender: str
    age: int
    service: str | None
    admittime: str
    dischtime: str
    mesh_age_group: str

    @classmethod
    def from_dict(cls, data: dict) -> "Admission":
        return cls(
            hadm_id=data["hadm_id"],
            subject_id=data["subject_id"],
            bhc_category=data["bhc_category"],
            token_stratum=data["token_stratum"],
            los_quartile=data["los_quartile"],
            los_days=data["los_days"],
            note_category_availability=data["note_category_availability"],
            bhc_numbering_convention=data["bhc_numbering_convention"],
            bhc_modifiers=data["bhc_modifiers"],
            note_count=data["note_count"],
            note_categories=data["note_categories"],
            lab_count=data["lab_count"],
            lab_abnormal_count=data["lab_abnormal_count"],
            prescription_count=data["prescription_count"],
            bhc_word_count=data["bhc_word_count"],
            bhc_token_counts={k: int(v) for k, v in data["bhc_token_counts"].items()},
            charting_days=data["charting_days"],
            record_token_counts={
                model: {k: int(v) for k, v in per_component.items()}
                for model, per_component in data["record_token_counts"].items()
            },
            max_total_tokens=data["max_total_tokens"],
            narrative_record_token_counts={
                k: int(v) for k, v in data["narrative_record_token_counts"].items()
            },
            hospital_expire_flag=data["hospital_expire_flag"],
            temporal_density=data["temporal_density"],
            timestamp_completeness=data["timestamp_completeness"],
            source_diversity=data["source_diversity"],
            pre_admission_notes=data["pre_admission_notes"],
            gender=data["gender"],
            age=data["age"],
            service=data["service"],
            admittime=data["admittime"],
            dischtime=data["dischtime"],
            mesh_age_group=data["mesh_age_group"],
        )


# Boundary table for compute_mesh_age_group. Each entry is
# (max_age_exclusive, label): an admission with age < max_age_exclusive falls
# into that bucket. Order matters, since the first match wins.
_MESH_AGE_BUCKETS: tuple[tuple[int, str], ...] = (
    (2, "Infant"),
    (6, "Preschool"),
    (13, "Child"),
    (19, "Adolescent"),
    (25, "Young Adult"),
    (45, "Adult"),
    (65, "Middle Aged"),
    (80, "Aged"),
)
_MESH_AGE_DEFAULT = "Aged, 80 and over"


def derive_note_category_availability(note_categories: "dict[str, int]") -> str:
    """Classify an admission by the kinds of clinical notes it has.

    Returns ``"clinical_narrative"`` if the admission has at least one
    Nursing or Physician note, else ``"imaging_only"``.
    """
    if any(c == "Nursing" or c.startswith("Physician") for c in note_categories):
        return "clinical_narrative"
    return "imaging_only"


def compute_mesh_age_group(age: int) -> str:
    """Map an admission age (in years) to a MeSH life-stage label.

    MeSH defines overlapping age categories ("Young Adult" 19-24 vs "Adult"
    19-44). The boundary table above resolves this by listing Young Adult
    first, so 19-24 maps to "Young Adult" and 25-44 maps to "Adult".

    MIMIC-III caps real ages at 90, and age 90 falls into "Aged, 80 and over".

    MeSH also defines "Newborn" (≤1 month), but age arrives in whole years, so
    age=0 spans both Newborn (0-1 month) and Infant (1-11 months). Age=0 maps
    to the more inclusive "Infant".
    """
    for max_age, label in _MESH_AGE_BUCKETS:
        if age < max_age:
            return label
    return _MESH_AGE_DEFAULT


@dataclass(frozen=True)
class Cohort:
    """A cohort file: metadata dict plus a list of typed admissions.

    Metadata stays an untyped dict because subset cohort files (smoke, prompt test)
    have different layouts. Admissions share a schema across cohort files, so
    that is where typing is worthwhile.
    """

    metadata: dict
    admissions: list[Admission]

    @classmethod
    def from_dict(cls, data: dict) -> "Cohort":
        return cls(
            metadata=data["metadata"],
            admissions=[Admission.from_dict(a) for a in data["admissions"]],
        )


def load_cohort(path: Path) -> Cohort:
    """Load and validate a cohort JSON file."""
    return Cohort.from_dict(json.loads(path.read_text()))


def _tokenize_strings[K](
    items: Mapping[K, str],
    tokenizers: dict[str, "PreTrainedTokenizerBase"],
    n_workers: int,
) -> dict[K, dict[str, int]]:
    """Tokenize a {key: text} mapping, returning {key: {model_key: count}}.

    Runs on a thread pool when ``n_workers > 1``. Hugging Face's fast
    tokenizers release the GIL during ``encode``, so threads scale on
    this work.
    """

    def count_one(item: tuple[K, str]) -> tuple[K, dict[str, int]]:
        key, text = item
        return key, count_per_model(tokenizers, text)

    if n_workers <= 1:
        return dict(count_one(item) for item in items.items())
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return dict(ex.map(count_one, items.items()))


_NOTES = "notes"
_LABS_ALL = "labs_all"
_LABS_ABNORMAL = "labs_abnormal"
_PRESCRIPTIONS = "prescriptions"
_DEMOGRAPHICS = "demographics"
_RECORD_TYPES = (_NOTES, _LABS_ALL, _LABS_ABNORMAL, _PRESCRIPTIONS, _DEMOGRAPHICS)

# Dict key for BHC text in the combined tokenization pass. Kept out of
# _RECORD_TYPES so a BHC entry never collides with a record entry.
_BHC_TYPE = "_bhc"


def _pull_record_text_parts(
    conn: duckdb.DuckDBPyConnection, hadm_ids: list[int]
) -> dict[str, dict[int, str]]:
    """Pull the record-text components per admission, keyed by record type.

    Returns ``{record_type: {hadm_id: concatenated_text}}`` for notes, labs,
    prescriptions, and demographics. Each inner dict may omit admissions that
    have no rows in that source table.
    """
    noteevents = table_ref("NOTEEVENTS.csv.gz")
    labevents = table_ref("LABEVENTS.csv.gz")
    d_labitems = table_ref("D_LABITEMS.csv.gz")
    prescriptions = table_ref("PRESCRIPTIONS.csv.gz")
    admissions = table_ref("ADMISSIONS.csv.gz")
    patients = table_ref("PATIENTS.csv.gz")

    # Notes: usable source notes per admission.
    notes_rows = conn.execute(
        f"""
        SELECT HADM_ID,
               -- ORDER BY inside string_agg pins the concatenation order so the
               -- text is reproducible. Token counts do not depend on that order.
               -- COALESCE returns '' when every note in the group has NULL text.
               COALESCE(
                   string_agg(TEXT, E'\n' ORDER BY {note_order_sql()}), ''
               ) AS text
        FROM {noteevents}
        WHERE HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
          AND {noteevents_input_filter()}  -- keep only usable source notes
        GROUP BY HADM_ID
        """,
        [hadm_ids],
    ).fetchall()
    notes_by_id = {int(h): t for h, t in notes_rows}

    # Labs: join with D_LABITEMS for the human-readable test name. One pull
    # for all labs, one filtered pull for FLAG = 'abnormal'.
    labs_all_rows = conn.execute(
        f"""
        SELECT l.HADM_ID,
               COALESCE(string_agg(
                   CONCAT_WS(' ',
                       CAST(l.CHARTTIME AS VARCHAR),
                       d.LABEL,
                       l.VALUE,
                       l.VALUEUOM,
                       l.FLAG),
                   E'\n' ORDER BY l.CHARTTIME NULLS LAST, l.ROW_ID), '') AS text
        FROM {labevents} l
        -- LEFT JOIN keeps a lab even when D_LABITEMS has no matching name for its ITEMID.
        -- An inner join would drop labs whose item code is not in the dictionary.
        LEFT JOIN {d_labitems} d USING (ITEMID)
        WHERE l.HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
        GROUP BY l.HADM_ID
        """,
        [hadm_ids],
    ).fetchall()
    labs_all_by_id = {int(h): t for h, t in labs_all_rows}

    labs_abn_rows = conn.execute(
        f"""
        SELECT l.HADM_ID,
               COALESCE(string_agg(
                   CONCAT_WS(' ',
                       CAST(l.CHARTTIME AS VARCHAR),
                       d.LABEL,
                       l.VALUE,
                       l.VALUEUOM,
                       l.FLAG),
                   E'\n' ORDER BY l.CHARTTIME NULLS LAST, l.ROW_ID), '') AS text
        FROM {labevents} l
        -- LEFT JOIN keeps a lab even when D_LABITEMS has no matching name for its ITEMID.
        LEFT JOIN {d_labitems} d USING (ITEMID)
        WHERE l.HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
          -- 'abnormal' is the FLAG value MIMIC uses for out-of-range results.
          AND l.FLAG = 'abnormal'
        GROUP BY l.HADM_ID
        """,
        [hadm_ids],
    ).fetchall()
    labs_abn_by_id = {int(h): t for h, t in labs_abn_rows}

    rx_rows = conn.execute(
        f"""
        SELECT HADM_ID,
               COALESCE(string_agg(
                   CONCAT_WS(' ',
                       CAST(STARTDATE AS VARCHAR),
                       DRUG,
                       DOSE_VAL_RX,
                       DOSE_UNIT_RX,
                       ROUTE),
                   E'\n' ORDER BY STARTDATE NULLS LAST, ROW_ID), '') AS text
        FROM {prescriptions}
        WHERE HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
        GROUP BY HADM_ID
        """,
        [hadm_ids],
    ).fetchall()
    rx_by_id = {int(h): t for h, t in rx_rows}

    demo_rows = conn.execute(
        f"""
        SELECT a.HADM_ID,
               CONCAT_WS(' ',
                   CAST(p.GENDER AS VARCHAR),
                   CAST(a.ADMITTIME AS VARCHAR),
                   CAST(a.DISCHTIME AS VARCHAR),
                   a.ADMISSION_TYPE,
                   a.INSURANCE) AS text
        FROM {admissions} a
        -- LEFT JOIN keeps the admission even when PATIENTS has no matching row.
        LEFT JOIN {patients} p USING (SUBJECT_ID)
        WHERE a.HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
        """,
        [hadm_ids],
    ).fetchall()
    demo_by_id = {int(h): t for h, t in demo_rows}

    return {
        _NOTES: notes_by_id,
        _LABS_ALL: labs_all_by_id,
        _LABS_ABNORMAL: labs_abn_by_id,
        _PRESCRIPTIONS: rx_by_id,
        _DEMOGRAPHICS: demo_by_id,
    }


def _zero_record_counts() -> dict[str, dict[str, int]]:
    """Give every model a fresh set of record counts, all starting at zero."""
    return {model: dict.fromkeys(_RECORD_TYPES, 0) for model in MODEL_KEYS}


def compute_token_counts(
    conn: duckdb.DuckDBPyConnection,
    hadm_ids: list[int],
    bhc_texts: dict[int, str],
    tokenizers: dict[str, "PreTrainedTokenizerBase"],
    *,
    n_workers: int = 1,
) -> tuple[dict[int, dict[str, dict[str, int]]], dict[int, dict[str, int]]]:
    """Tokenize record components and BHC ground truth per admission and per model.

    Returns ``(record_token_counts, bhc_token_counts)``:
    - ``record_token_counts[hadm_id][model_key]`` is a dict of per-record-type
      token counts. Missing admissions are filled with zeros so the caller
      always gets a complete per-model dict per HADM_ID.
    - ``bhc_token_counts[hadm_id][model_key]`` is the BHC ground-truth
      token count under that model's tokenizer.
    """
    texts_by_type = _pull_record_text_parts(conn, hadm_ids)
    texts_by_type[_BHC_TYPE] = bhc_texts

    work: dict[tuple[int, str], str] = {}
    for text_type, texts in texts_by_type.items():
        for hid, txt in texts.items():
            work[(hid, text_type)] = txt

    counts = _tokenize_strings(work, tokenizers, n_workers)

    record_counts: dict[int, dict[str, dict[str, int]]] = {
        int(hid): _zero_record_counts() for hid in hadm_ids
    }
    bhc_counts: dict[int, dict[str, int]] = {
        int(hid): dict.fromkeys(MODEL_KEYS, 0) for hid in hadm_ids
    }
    for (hid, text_type), counts_per_model in counts.items():
        hid_int = int(hid)
        if text_type == _BHC_TYPE:
            for model, n in counts_per_model.items():
                bhc_counts[hid_int][model] = int(n)
        else:
            for model, n in counts_per_model.items():
                record_counts[hid_int][model][text_type] = int(n)
    return record_counts, bhc_counts


def compute_narrative_token_counts(
    conn: duckdb.DuckDBPyConnection,
    hadm_ids: list[int],
    tokenizers: dict[str, "PreTrainedTokenizerBase"],
    *,
    n_workers: int = 1,
) -> dict[int, dict[str, int]]:
    """Count each admission's narrative-note tokens, once per model.

    A narrative note is a usable note (passes ``noteevents_input_filter``)
    whose CATEGORY is exactly ``Nursing`` or starts with ``Physician``.

    Returns ``{hadm_id: {model_key: int}}``. Admissions with no narrative
    notes get a per-model dict of zeros so every key is always present.
    """
    noteevents = table_ref("NOTEEVENTS.csv.gz")
    rows = conn.execute(
        f"""
        SELECT HADM_ID,
               COALESCE(string_agg(TEXT, E'\n'), '') AS text
        FROM {noteevents}
        WHERE HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
          AND {noteevents_input_filter()}
          AND (CATEGORY = 'Nursing' OR CATEGORY LIKE 'Physician%')
        GROUP BY HADM_ID
        """,
        [hadm_ids],
    ).fetchall()
    text_by_id: dict[object, str] = {int(hid): text for hid, text in rows}
    counts = _tokenize_strings(text_by_id, tokenizers, n_workers)
    out: dict[int, dict[str, int]] = {int(hid): dict.fromkeys(MODEL_KEYS, 0) for hid in hadm_ids}
    for hid, per_model_counts in counts.items():
        out[int(hid)] = {model: int(n) for model, n in per_model_counts.items()}
    return out


def compute_temporal_metrics(
    conn: duckdb.DuckDBPyConnection, hadm_ids: list[int]
) -> dict[int, dict]:
    """Compute per-admission temporal_density and timestamp_completeness.

    Returns ``{hadm_id: {"temporal_density": {...}, "timestamp_completeness": {...}}}``.

    Throughout this function, "usable note" means a note that passes
    ``noteevents_input_filter()``.

    temporal_density:
    - first_note_offset_hours: first usable note time - admittime, in
      hours. Negative when an ED note pre-dates the formal admission.
      Tells how much pre-admission context the BHC ground truth could
      have drawn on.
    - last_note_offset_hours: last usable note time - dischtime, in
      hours. Typically negative. A large negative value flags a quiet
      period before discharge (placement waits, social work delays)
      where the BHC may describe events that left no note record.
    - clinical_span_hours: last note time - first note time, in hours.
      The documentation footprint, distinct from LOS (admit to
      discharge). Distinguishes long admissions with sparse notes from
      short admissions with dense notes that share the same note_count.
    - gap_hours_between_notes: dict with p50, p75, p90, and max of
      consecutive-note gaps. Tells whether documentation is steady
      (regular rounds, narrow gaps) or bursty (procedures, wide gaps).
      Restricted to notes with non-null CHARTTIME because gap precision
      requires a real timestamp. Empty dict if fewer than two such notes.
    - notes_per_day: dict with p50, p75, p90, and max of per-day note
      counts. Days with zero notes are excluded, so the percentiles
      describe intensity on documented days only.

    timestamp_completeness:
    - null_charttime_note_count: usable notes where CHARTTIME is NULL.
    - null_charttime_note_categories: per-CATEGORY count of those
      notes. ECG and Echo notes never populate CHARTTIME, so they
      dominate this count.
    """
    noteevents = table_ref("NOTEEVENTS.csv.gz")
    admissions_ref = table_ref("ADMISSIONS.csv.gz")

    # Step 1: notes that pass noteevents_input_filter (discharge summaries
    # and error-marked entries excluded), joined with admit times.
    conn.execute(
        f"""
        CREATE TEMP TABLE pool_usable_notes AS
        SELECT n.HADM_ID,
               n.CATEGORY AS category,
               n.CHARTTIME::TIMESTAMP AS charttime,
               n.CHARTDATE::DATE AS chartdate,
               -- effective_time falls back to CHARTDATE at 00:00:00 when
               -- CHARTTIME is NULL. Without this fallback, ECG and Echo
               -- notes (which never populate CHARTTIME) would have NULL
               -- effective_time and drop out of first/last/span.
               COALESCE(n.CHARTTIME::TIMESTAMP, n.CHARTDATE::TIMESTAMP) AS effective_time,
               a.ADMITTIME::TIMESTAMP AS admittime,
               a.DISCHTIME::TIMESTAMP AS dischtime
        FROM {noteevents} n
        JOIN {admissions_ref} a USING (HADM_ID)
        WHERE n.HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
          AND {noteevents_input_filter("n")}
    """,
        [hadm_ids],
    )

    # Step 2: per-admission offsets and span.
    offset_rows = conn.execute("""
        SELECT HADM_ID,
               -- (first usable note - admit), in hours. EXTRACT(EPOCH ...)
               -- returns the timedelta in seconds, then / 3600 converts to hours.
               -- MIN(admittime) reduces the per-row admittime to a scalar
               -- inside the GROUP BY (every row in a HADM_ID partition
               -- carries the same admittime, so MIN equals MAX equals the
               -- value itself).
               EXTRACT(EPOCH FROM (MIN(effective_time) - MIN(admittime))) / 3600.0
                   AS first_note_offset_hours,
               -- (last usable note - discharge), in hours. Typically negative
               -- because usable notes precede discharge.
               EXTRACT(EPOCH FROM (MAX(effective_time) - MIN(dischtime))) / 3600.0
                   AS last_note_offset_hours,
               -- (last usable note - first usable note), in hours. The
               -- documentation footprint, distinct from LOS (admit to
               -- discharge).
               EXTRACT(EPOCH FROM (MAX(effective_time) - MIN(effective_time))) / 3600.0
                   AS clinical_span_hours
        FROM pool_usable_notes
        GROUP BY HADM_ID
    """).fetchall()
    offsets_by_id: dict[int, tuple[float, float, float]] = {
        int(r[0]): (float(r[1]), float(r[2]), float(r[3])) for r in offset_rows
    }

    # Step 3: consecutive-note gaps. Skip notes without CHARTTIME because
    # two date-only notes on the same day would both sit at 00:00:00 and
    # produce a false zero-hour gap.
    conn.execute("""
        CREATE TEMP TABLE pool_gap_intervals AS
        SELECT HADM_ID,
               EXTRACT(EPOCH FROM (
                   charttime - LAG(charttime) OVER (PARTITION BY HADM_ID ORDER BY charttime)
               )) / 3600.0 AS gap_hours
        FROM pool_usable_notes
        WHERE charttime IS NOT NULL
    """)
    gap_rows = conn.execute("""
        SELECT HADM_ID,
               QUANTILE_CONT(gap_hours, 0.5) AS p50,
               QUANTILE_CONT(gap_hours, 0.75) AS p75,
               QUANTILE_CONT(gap_hours, 0.9) AS p90,
               MAX(gap_hours) AS max
        FROM pool_gap_intervals
        WHERE gap_hours IS NOT NULL
        GROUP BY HADM_ID
    """).fetchall()
    gaps_by_id: dict[int, dict[str, float]] = {
        int(r[0]): {"p50": float(r[1]), "p75": float(r[2]), "p90": float(r[3]), "max": float(r[4])}
        for r in gap_rows
    }

    # Step 4: notes per day. GROUP BY (HADM_ID, chartdate) implicitly drops
    # zero-note days, so the percentiles describe density on documented days.
    conn.execute("""
        CREATE TEMP TABLE pool_notes_per_day AS
        SELECT HADM_ID, chartdate, COUNT(*) AS n
        FROM pool_usable_notes
        WHERE chartdate IS NOT NULL
        GROUP BY HADM_ID, chartdate
    """)
    notes_per_day_rows = conn.execute("""
        SELECT HADM_ID,
               QUANTILE_CONT(n, 0.5) AS p50,
               QUANTILE_CONT(n, 0.75) AS p75,
               QUANTILE_CONT(n, 0.9) AS p90,
               MAX(n) AS max
        FROM pool_notes_per_day
        GROUP BY HADM_ID
    """).fetchall()
    notes_per_day_by_id: dict[int, dict[str, float]] = {
        int(r[0]): {"p50": float(r[1]), "p75": float(r[2]), "p90": float(r[3]), "max": float(r[4])}
        for r in notes_per_day_rows
    }

    # Step 5: timestamp completeness counts.
    null_count_rows = conn.execute("""
        SELECT HADM_ID, COUNT(*) AS null_count
        FROM pool_usable_notes
        WHERE charttime IS NULL
        GROUP BY HADM_ID
    """).fetchall()
    null_counts_by_id: dict[int, int] = {int(r[0]): int(r[1]) for r in null_count_rows}

    null_by_category_rows = conn.execute("""
        SELECT HADM_ID, category, COUNT(*) AS n
        FROM pool_usable_notes
        WHERE charttime IS NULL
        GROUP BY HADM_ID, category
    """).fetchall()
    null_by_category: dict[int, dict[str, int]] = {}
    for hid, category, n in null_by_category_rows:
        null_by_category.setdefault(int(hid), {})[str(category)] = int(n)
    # Sort by category so the JSON serialization order is stable. DuckDB's
    # GROUP BY does not guarantee row order, so the insertion order would
    # otherwise drift between runs and produce noise in pool-file diffs.
    null_by_category = {hid: dict(sorted(d.items())) for hid, d in null_by_category.items()}

    # Drop the temp tables so a second call on the same connection starts clean.
    for tbl in (
        "pool_notes_per_day",
        "pool_gap_intervals",
        "pool_usable_notes",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {tbl}")

    # Step 6: assemble per-admission nested dicts.
    out: dict[int, dict] = {}
    for hid in hadm_ids:
        hid_int = int(hid)
        first_off, last_off, span = offsets_by_id.get(hid_int, (0.0, 0.0, 0.0))
        out[hid_int] = {
            "temporal_density": {
                "first_note_offset_hours": first_off,
                "last_note_offset_hours": last_off,
                "clinical_span_hours": span,
                "gap_hours_between_notes": gaps_by_id.get(hid_int, {}),
                "notes_per_day": notes_per_day_by_id.get(hid_int, {}),
            },
            "timestamp_completeness": {
                "null_charttime_note_count": null_counts_by_id.get(hid_int, 0),
                "null_charttime_note_categories": null_by_category.get(hid_int, {}),
            },
        }
    return out


def compute_source_diversity(
    conn: duckdb.DuckDBPyConnection, hadm_ids: list[int]
) -> dict[int, dict[str, int]]:
    """Compute distinct lab items, distinct drugs, and distinct services per admission.

    Returns ``{hadm_id: {"distinct_lab_items": int, "distinct_drugs": int,
    "distinct_services_touching": int}}``.

    Each field counts distinct values in a different source table:

    - distinct_lab_items: distinct ITEMIDs in LABEVENTS, independent of
      lab_count, which counts result rows: 5 tests drawn 10x each gives
      lab_count 50, distinct_lab_items 5.
    - distinct_drugs: distinct DRUG values in PRESCRIPTIONS, independent
      of prescription_count, which counts rows: 5 drugs refilled 6x each
      gives prescription_count 30, distinct_drugs 5.
    - distinct_services_touching: distinct CURR_SERVICE values in
      SERVICES (one row per transfer), independent of the ``service``
      field on the pool record, which has only the discharging service.
    """
    labevents = table_ref("LABEVENTS.csv.gz")
    prescriptions = table_ref("PRESCRIPTIONS.csv.gz")
    services_ref = table_ref("SERVICES.csv.gz")

    lab_rows = conn.execute(
        f"""
        SELECT HADM_ID, COUNT(DISTINCT ITEMID) AS n
        FROM {labevents}
        WHERE HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
        GROUP BY HADM_ID
        """,
        [hadm_ids],
    ).fetchall()
    lab_by_id = {int(r[0]): int(r[1]) for r in lab_rows}

    rx_rows = conn.execute(
        f"""
        SELECT HADM_ID, COUNT(DISTINCT DRUG) AS n
        FROM {prescriptions}
        WHERE HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
        GROUP BY HADM_ID
        """,
        [hadm_ids],
    ).fetchall()
    rx_by_id = {int(r[0]): int(r[1]) for r in rx_rows}

    svc_rows = conn.execute(
        f"""
        SELECT HADM_ID, COUNT(DISTINCT CURR_SERVICE) AS n
        FROM {services_ref}
        WHERE HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
        GROUP BY HADM_ID
        """,
        [hadm_ids],
    ).fetchall()
    svc_by_id = {int(r[0]): int(r[1]) for r in svc_rows}

    return {
        int(hid): {
            "distinct_lab_items": lab_by_id.get(int(hid), 0),
            "distinct_drugs": rx_by_id.get(int(hid), 0),
            "distinct_services_touching": svc_by_id.get(int(hid), 0),
        }
        for hid in hadm_ids
    }


def compute_pre_admission_notes(
    conn: duckdb.DuckDBPyConnection,
    hadm_ids: list[int],
    tokenizers: dict[str, "PreTrainedTokenizerBase"],
    *,
    n_workers: int = 1,
) -> dict[int, dict]:
    """Compute pre-admission usable note count, per-category breakdown, and per-model token counts.

    Returns ``{hadm_id: {"count": int, "token_counts": {model_key: int},
    "categories": {category: count}}}``.

    A pre-admission note is a usable note (passes ``noteevents_input_filter``)
    whose CHARTDATE is strictly earlier than the calendar day of ADMITTIME.
    The ``categories`` field exposes which NOTEEVENTS categories contribute
    per admission. The boundary uses dates because many pre-admission notes are date-only
    (no CHARTTIME). A note on the admission day counts as in-admission.

    ``token_counts`` is a subset of ``record_token_counts.<model>.notes``,
    not an addition to it. Pre-admission notes are also usable notes, so
    their tokens already appear inside ``record_token_counts.<model>.notes``
    (which has no admit-time boundary).
    """
    noteevents = table_ref("NOTEEVENTS.csv.gz")
    admissions_ref = table_ref("ADMISSIONS.csv.gz")

    rows = conn.execute(
        f"""
        SELECT n.HADM_ID,
               n.CATEGORY AS category,
               n.TEXT AS text
        FROM {noteevents} n
        JOIN {admissions_ref} a USING (HADM_ID)
        WHERE n.HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
          AND {noteevents_input_filter("n")}
          -- Strictly before the admission day. A note on admittime's date
          -- is treated as in-admission, not pre-admission.
          AND n.CHARTDATE::DATE < a.ADMITTIME::DATE
        """,
        [hadm_ids],
    ).fetchall()

    # Build per-admission text (concatenated), category counts, and total counts
    # in a single pass over the result rows.
    texts_by_id: dict[int, list[str]] = {}
    cats_by_id: dict[int, dict[str, int]] = {}
    counts_by_id: dict[int, int] = {}
    for hid, category, text in rows:
        hid_int = int(hid)
        texts_by_id.setdefault(hid_int, []).append(text or "")
        cat_str = str(category)
        cats_by_id.setdefault(hid_int, {})
        cats_by_id[hid_int][cat_str] = cats_by_id[hid_int].get(cat_str, 0) + 1
        counts_by_id[hid_int] = counts_by_id.get(hid_int, 0) + 1
    # Sort by category so the JSON serialization order is stable across runs.
    cats_by_id = {hid: dict(sorted(d.items())) for hid, d in cats_by_id.items()}

    # Concatenate so each admission tokenizes as one string per model.
    concat_by_id = {hid: "\n".join(parts) for hid, parts in texts_by_id.items()}
    token_counts = _tokenize_strings(concat_by_id, tokenizers, n_workers)
    zero_per_model = dict.fromkeys(MODEL_KEYS, 0)

    return {
        int(hid): {
            "count": counts_by_id.get(int(hid), 0),
            "token_counts": dict(token_counts.get(int(hid), zero_per_model)),
            "categories": cats_by_id.get(int(hid), {}),
        }
        for hid in hadm_ids
    }


def load_filtered_pool(
    conn: duckdb.DuckDBPyConnection,
    *,
    tokenizers: dict[str, "PreTrainedTokenizerBase"],
) -> pd.DataFrame:
    """Create the filtered pool as a DataFrame with all record properties.

    Token counting runs on a thread pool sized from the host's CPU count.
    """
    n_workers = _default_n_workers()
    logger.info("  Tokenizing with %d thread(s)", n_workers)

    logger.info("  Phase 1/6: Building filtered_pool table")
    create_filtered_pool(conn)
    df = conn.execute("SELECT * FROM filtered_pool").fetchdf()
    hadm_ids = df["hadm_id"].tolist()

    logger.info(
        "  Phase 2/6: Per-model token counts (%d tokenizers x %d admissions, "
        "takes several minutes)",
        len(MODEL_KEYS),
        len(hadm_ids),
    )
    bhc_texts = dict(zip(df["hadm_id"].astype(int), df["bhc_text"], strict=True))
    record_counts, bhc_counts = compute_token_counts(
        conn,
        hadm_ids,
        bhc_texts,
        tokenizers,
        n_workers=n_workers,
    )
    df["record_token_counts"] = df["hadm_id"].map(lambda h: record_counts[int(h)])
    df["bhc_token_counts"] = df["hadm_id"].map(lambda h: bhc_counts[int(h)])
    # Store the per-admission max total once here so it lives on the pool.
    df["max_total_tokens"] = df["record_token_counts"].map(max_total_tokens)
    narrative_counts = compute_narrative_token_counts(
        conn, hadm_ids, tokenizers, n_workers=n_workers
    )
    df["narrative_record_token_counts"] = df["hadm_id"].map(lambda h: narrative_counts[int(h)])

    logger.info("  Phase 3/6: Temporal metrics")
    temporal = compute_temporal_metrics(conn, hadm_ids)
    df["temporal_density"] = df["hadm_id"].map(lambda h: temporal[int(h)]["temporal_density"])
    df["timestamp_completeness"] = df["hadm_id"].map(
        lambda h: temporal[int(h)]["timestamp_completeness"]
    )

    logger.info("  Phase 4/6: Source diversity")
    diversity = compute_source_diversity(conn, hadm_ids)
    df["source_diversity"] = df["hadm_id"].map(lambda h: diversity[int(h)])

    logger.info(
        "  Phase 5/6: Pre-admission note counts (%d admissions, takes several minutes)",
        len(hadm_ids),
    )
    pre_admission = compute_pre_admission_notes(conn, hadm_ids, tokenizers, n_workers=n_workers)
    df["pre_admission_notes"] = df["hadm_id"].map(lambda h: pre_admission[int(h)])

    logger.info("  Phase 6/6: Note categories and MeSH age group")
    # bhc_text is not persisted in the pool file. See create_eligible_pool.
    df = df.drop(columns=["bhc_text"])

    # Per-category note counts as a nested dict per admission.
    cat_rows = conn.execute(
        f"""
        SELECT HADM_ID, CATEGORY, COUNT(*) AS n
        FROM {table_ref("NOTEEVENTS.csv.gz")}
        WHERE HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
          AND {noteevents_input_filter()}
        GROUP BY HADM_ID, CATEGORY
        ORDER BY HADM_ID, CATEGORY
        """,
        [hadm_ids],
    ).fetchall()
    note_categories: dict[int, dict[str, int]] = {}
    for hid, category, n in cat_rows:
        note_categories.setdefault(int(hid), {})[str(category)] = int(n)
    df["note_categories"] = df["hadm_id"].map(lambda h: note_categories[int(h)])

    df["mesh_age_group"] = df["age"].astype(int).map(compute_mesh_age_group)

    return df


def load_filtered_pool_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read a persisted filtered-pool JSON file into (DataFrame, metadata)."""
    data = json.loads(path.read_text())
    df = pd.DataFrame(data["admissions"])
    return df, data["metadata"]
