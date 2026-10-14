"""Data access for the spot-check review app: MIMIC queries, review-case
selection, and reading and writing the review file.
"""

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from mimic_data_prep.mimic import note_order_sql
from mimic_data_prep.sampling import TOKEN_STRATA


def fetch_bhc_data_batch(
    conn: duckdb.DuckDBPyConnection,
    bhc_jsonl: Path,
    noteevents_csv: Path,
    *,
    hadm_ids: tuple[int, ...] | list[int],
) -> dict[int, dict[str, Any]]:
    """Fetch BHC extraction data and discharge note text for many admissions.

    Reads bhc.jsonl once with ``hadm_id IN (...)``, then reads NOTEEVENTS once
    with ``ROW_ID IN (...)`` for the discharge note text, which the extraction
    does not embed. HADM_IDs not present in bhc.jsonl are omitted from the
    returned dict.

    """
    ids = list(hadm_ids)
    bhc_df = conn.execute(
        f"""
        SELECT hadm_id, subject_id, row_id,
               bhc_text, bhc_start, bhc_end, note_length,
               bhc_char_count, bhc_word_count,
               bhc_structure.category AS bhc_category
        FROM read_json_auto('{bhc_jsonl}')
        WHERE hadm_id IN (SELECT UNNEST($1::INTEGER[]))
        """,
        [ids],
    ).fetchdf()

    # Every extraction row has its discharge note's ROW_ID, so one
    # NOTEEVENTS scan fetches the note text for all admissions.
    note_texts: dict[int, str] = {}
    if len(bhc_df):
        row_ids = [int(r) for r in bhc_df["row_id"].tolist()]
        note_df = conn.execute(
            f"""
            SELECT ROW_ID, TEXT FROM read_csv_auto('{noteevents_csv}')
            WHERE ROW_ID IN (SELECT UNNEST($1::INTEGER[]))
            """,
            [row_ids],
        ).fetchdf()
        note_texts = {int(r["ROW_ID"]): r["TEXT"] for _, r in note_df.iterrows()}

    result: dict[int, dict[str, Any]] = {}
    for _, row in bhc_df.iterrows():
        hid = int(row["hadm_id"])
        result[hid] = {
            "hadm_id": hid,
            "subject_id": int(row["subject_id"]) if pd.notna(row["subject_id"]) else None,
            "bhc_text": row["bhc_text"],
            "bhc_start": int(row["bhc_start"]) if pd.notna(row["bhc_start"]) else None,
            "bhc_end": int(row["bhc_end"]) if pd.notna(row["bhc_end"]) else None,
            "note_text": note_texts.get(int(row["row_id"])),
            "note_length": int(row["note_length"]) if pd.notna(row["note_length"]) else None,
            "bhc_char_count": int(row["bhc_char_count"]),
            "bhc_word_count": int(row["bhc_word_count"]),
            "bhc_category": row["bhc_category"],
        }
    return result


def fetch_demographics_batch(
    conn: duckdb.DuckDBPyConnection,
    patients_csv: Path,
    admissions_csv: Path,
    *,
    hadm_ids: tuple[int, ...] | list[int],
) -> dict[int, dict[str, Any]]:
    """Fetch demographics for many admissions in one pair of scans.

    HADM_IDs not found are omitted from the returned dict.
    """
    ids = list(hadm_ids)
    df = conn.execute(
        f"""
        SELECT
            a.HADM_ID,
            a.SUBJECT_ID,
            p.GENDER,
            -- Cap age at 90: MIMIC shifts DOB ~300 years for patients over 89, so raw age above the
            -- cap is a de-identification artifact.
            LEAST(DATE_DIFF('year', p.DOB::DATE, a.ADMITTIME::DATE), 90) AS age,
            a.ADMITTIME::VARCHAR AS admittime,
            a.DISCHTIME::VARCHAR AS dischtime,
            DATE_DIFF('day', a.ADMITTIME::TIMESTAMP, a.DISCHTIME::TIMESTAMP) AS los_days
        FROM read_csv_auto('{admissions_csv}') a
        -- Every admission has exactly one PATIENTS row, so an inner join never drops one.
        JOIN read_csv_auto('{patients_csv}') p ON a.SUBJECT_ID = p.SUBJECT_ID
        WHERE a.HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
        """,
        [ids],
    ).fetchdf()
    demographics: dict[int, dict] = {}
    for _, row in df.iterrows():
        hadm_id = int(row["HADM_ID"])
        demographics[hadm_id] = {
            "hadm_id": hadm_id,
            "subject_id": int(row["SUBJECT_ID"]),
            "gender": row["GENDER"],
            "age": int(row["age"]),
            "admittime": row["admittime"],
            "dischtime": row["dischtime"],
            "los_days": int(row["los_days"]),
        }
    return demographics


def fetch_notes_batch(
    conn: duckdb.DuckDBPyConnection,
    noteevents_csv: Path,
    admissions_csv: Path,
    *,
    hadm_ids: tuple[int, ...] | list[int],
) -> dict[int, pd.DataFrame]:
    """Fetch all notes for many admissions in one scan of NOTEEVENTS.

    Returns {hadm_id: DataFrame} with columns ROW_ID, CHARTDATE, CHARTTIME, CATEGORY,
    TEXT, note_length, is_error, is_pre_admission. HADM_IDs with no notes are omitted.
    Rows come back in charting-time order (see ``mimic.note_order_sql``).

    CHARTTIME is in the output because the rows are sorted by it. A reader checking
    an admission's timeline needs to see the column that decided it.
    """
    ids = list(hadm_ids)
    df = conn.execute(
        f"""
        SELECT n.HADM_ID, n.ROW_ID, n.CHARTDATE, n.CHARTTIME, n.CATEGORY, n.TEXT,
               LENGTH(n.TEXT) AS note_length,
               -- ISERROR is NULL for normal notes and '1' for flagged errors.
               CASE WHEN n.ISERROR = '1' THEN TRUE ELSE FALSE END AS is_error,
               -- A note charted before the admission day is pre-admission. CHARTDATE is
               -- date-only, so compare it against ADMITTIME cast to a date. The comparison
               -- is strict, so a note charted on the admission day is not pre-admission.
               CASE WHEN n.CHARTDATE < a.ADMITTIME::DATE THEN TRUE ELSE FALSE END
                   AS is_pre_admission
        FROM read_csv_auto('{noteevents_csv}') n
        -- Inner join drops notes whose HADM_ID has no ADMISSIONS row,
        -- because is_pre_admission needs the admission's ADMITTIME.
        JOIN read_csv_auto('{admissions_csv}') a ON n.HADM_ID = a.HADM_ID
        WHERE n.HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
        ORDER BY n.HADM_ID, {note_order_sql("n")}
        """,
        [ids],
    ).fetchdf()
    return {
        int(hid): sub.drop(columns=["HADM_ID"]).reset_index(drop=True)
        for hid, sub in df.groupby("HADM_ID")
    }


def fetch_lab_count_batch(
    conn: duckdb.DuckDBPyConnection,
    labevents_csv: Path,
    *,
    hadm_ids: tuple[int, ...] | list[int],
) -> dict[int, dict[str, int]]:
    """Count total and abnormal labs per admission in one scan.

    Returns one entry per requested HADM_ID, with zeros for admissions that
    have no labs.
    """
    ids = list(hadm_ids)
    df = conn.execute(
        f"""
        SELECT HADM_ID,
               COUNT(*) AS n,
               -- 'abnormal' is the literal FLAG value LABEVENTS stores for out-of-range results.
               COUNT(*) FILTER (WHERE FLAG = 'abnormal') AS n_abnormal
        FROM read_csv_auto('{labevents_csv}')
        WHERE HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
        GROUP BY HADM_ID
        """,
        [ids],
    ).fetchdf()
    counts = {
        int(r["HADM_ID"]): {"total": int(r["n"]), "abnormal": int(r["n_abnormal"])}
        for _, r in df.iterrows()
    }
    return {hid: counts.get(hid, {"total": 0, "abnormal": 0}) for hid in ids}


def longest_no_note_gap(days_with_notes: set[int], stay_days: int) -> int:
    """Longest run of consecutive stay days with no note of any kind.

    ``days_with_notes`` has 1-based hospital-day indices. ``stay_days`` is
    the inclusive stay length (admit day through discharge day). Returns 0
    when every day in the stay has a note.
    """
    longest_gap = 0
    current_gap = 0
    for day in range(1, stay_days + 1):
        if day in days_with_notes:
            current_gap = 0
        else:
            current_gap += 1
            longest_gap = max(longest_gap, current_gap)
    return longest_gap


def _pick_grid_cases(
    admissions: list[dict[str, Any]],
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """One admission per (token_stratum, bhc_category) cell, deterministically shuffled."""
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for adm in admissions:
        by_cell.setdefault((adm["token_stratum"], adm["bhc_category"]), []).append(adm)

    selected: list[dict[str, Any]] = []
    for key in sorted(by_cell.keys(), key=lambda k: (TOKEN_STRATA.index(k[0]), k[1])):
        pool = by_cell[key].copy()
        rng.shuffle(pool)
        selected.append(pool[0])
    return selected


def _pick_stratum_extras(
    admissions: list[dict[str, Any]],
    *,
    extra_per_stratum: dict[str, int],
    already_selected: set[int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Draw extras from named strata, skipping any HADM already in the grid."""
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for adm in admissions:
        by_stratum.setdefault(adm["token_stratum"], []).append(adm)

    selected: list[dict[str, Any]] = []
    for stratum, n_extra in extra_per_stratum.items():
        if stratum not in by_stratum:
            continue
        pool = [a for a in by_stratum[stratum] if a["hadm_id"] not in already_selected]
        rng.shuffle(pool)
        selected.extend(pool[:n_extra])
    return selected


def select_review_cases(
    admissions: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic subset of admissions for spot-check review.

    Picks one admission from each (token_stratum, bhc_category) cell so
    every stratum is sampled at both BHC formats, then adds one more
    admission each from 96K and 112K (no overlap with the grid picks):
    the two strata closest to the 112K pool budget, where the budget
    leaves the least headroom for prompt-size or tokenizer-count drift.

    Results are sorted by (token_stratum order, bhc_category, hadm_id) so
    the reviewer walks short-to-long admissions.
    """
    extra_per_stratum = {"96K": 1, "112K": 1}
    rng = random.Random(seed)

    grid = _pick_grid_cases(admissions, rng=rng)
    grid_ids = {a["hadm_id"] for a in grid}
    extras = _pick_stratum_extras(
        admissions,
        extra_per_stratum=extra_per_stratum,
        already_selected=grid_ids,
        rng=rng,
    )

    return sorted(
        grid + extras,
        key=lambda a: (TOKEN_STRATA.index(a["token_stratum"]), a["bhc_category"], a["hadm_id"]),
    )


def load_reviews(path: Path) -> dict[str, Any]:
    """Load existing reviews from JSON file. Returns empty structure if file doesn't exist."""
    if not path.exists():
        return {"metadata": {}, "reviews": {}}
    return json.loads(path.read_text())


def save_review(
    path: Path,
    *,
    hadm_id: int,
    coherence: dict[str, bool],
    coverage: dict[str, bool],
    notes: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save or update a single review. Preserves existing reviews in the file.

    If metadata is provided and the file has no metadata yet, it is written once.
    """
    reviews_data = load_reviews(path)
    if metadata and not reviews_data.get("metadata"):
        reviews_data["metadata"] = metadata
    reviews_data["reviews"][str(hadm_id)] = {
        "coherence": coherence,
        "coverage": coverage,
        "notes": notes,
        "reviewed_at": datetime.now(tz=UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reviews_data, indent=2) + "\n")
