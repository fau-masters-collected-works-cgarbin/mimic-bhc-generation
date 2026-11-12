"""MIMIC-III data access configuration and shared cohort definitions.

Holds the paths and SQL fragments the queries share (table references, note
filters and ordering, age and service SQL), and the pool pipeline that applies
the eligibility and threshold filters to build the admission pools.

See AGENTS.md for the data access pattern.
"""

from pathlib import Path

import duckdb

# All sibling repositories and data live at the same filesystem level as this
# repo. PROJECTS_DIR is the directory that contains this repo and its siblings.
PROJECTS_DIR = Path(__file__).resolve().parents[3]
MIMIC_DIR = PROJECTS_DIR / "mimic-iii/physionet.org/files/mimiciii/1.4"

# Output of the mimic-bhc-extraction pipeline: one extracted Brief Hospital
# Course per admission, with quality flags and structural metadata.
BHC_JSONL = PROJECTS_DIR / "mimic-bhc-extraction/bhc.jsonl"

# Experiment data lives in a separate repo to comply with the PhysioNet DUA.
# No MIMIC data (including patient IDs) may be stored in this repo.
_experiment_data_repo = PROJECTS_DIR / "phd-dissertation-experiment-results"
COHORTS_DIR = _experiment_data_repo / "cohorts"
RESULTS_DIR = _experiment_data_repo / "results"
# The concept artifact: per-text UMLS concept extractions, one file per
# extractor fingerprint.
CONCEPTS_DIR = _experiment_data_repo / "concepts"
# The prepared-input artifact: filtered, deduplicated admission bundles, one
# file per cohort/dedup combination.
PREPARED_DIR = _experiment_data_repo / "prepared"
# Retired runs, moved out of RESULTS_DIR. Split into obsolete/ (superseded
# earlier versions) and one-off/ (narrow probes).
ARCHIVE_DIR = _experiment_data_repo / "archive"


def table_ref(name: str, *, all_varchar: bool = False) -> str:
    """SQL fragment to read a MIMIC-III CSV.gz file directly with DuckDB.

    ``all_varchar`` reads every column as text and skips type inference. A failed
    inference prints the offending row, MIMIC identifiers included.
    """
    options = ", all_varchar=true" if all_varchar else ""
    return f"read_csv_auto('{MIMIC_DIR / name}'{options})"


def bhc_ref() -> str:
    """SQL fragment to read the BHC extraction JSONL with DuckDB."""
    return f"read_json_auto('{BHC_JSONL}')"


def noteevents_input_filter(alias: str = "") -> str:
    """SQL WHERE fragment selecting the source notes of an admission.

    Keeps everything charted during the admission except the discharge summary,
    which has the ground-truth Brief Hospital Course and would leak the answer
    back into the input. It also drops the notes MIMIC marks as entered in error. Pass a
    table alias (e.g., ``"n"``) to prefix column references.

    ISERROR is NULL for most notes and '1' for notes entered in error.
    DuckDB may infer it as INTEGER from CSV, so the CAST ensures a
    safe string comparison. The IS NULL check is needed because
    ``NULL != '1'`` evaluates to NULL (not TRUE) under SQL three-valued
    logic, which would incorrectly exclude the majority of notes.
    """
    p = f"{alias}." if alias else ""
    return (
        f"{p}CATEGORY != 'Discharge summary' "
        f"AND ({p}ISERROR IS NULL "
        f"OR CAST({p}ISERROR AS VARCHAR) != '1')"
    )


def note_order_sql(alias: str = "") -> str:
    """The order every step of the pipeline reads an admission's notes in.

    A single shared definition, so no two steps disagree about which note
    came first.

    Charting time and ROW_ID together fix the order, so the first note is always the
    earliest charted one. CHARTTIME places a note to the second. MIMIC records it
    for every category except ECG and Echo. ROW_ID settles the notes charted in the
    same second, of which MIMIC has many.

    This orders the notes of one admission. A query spanning several has to group
    them first, as in ``ORDER BY n.HADM_ID, {note_order_sql("n")}``.
    """
    p = f"{alias}." if alias else ""

    # CHARTTIME is the charting timestamp. DuckDB infers it as VARCHAR from the CSV,
    # so it has to be cast before it compares as a time.
    charted_at = f"{p}CHARTTIME::TIMESTAMP"

    # ECG and Echo have a chart date and no time. Placing them at noon is our
    # choice, with no basis in MIMIC. It puts them among that day's timed notes
    # instead of before or after all of them.
    undated_at_noon = f"{p}CHARTDATE::TIMESTAMP + INTERVAL 12 HOUR"

    # COALESCE returns the first argument that is not null, so it takes the most precise
    # time first (CHARTTIME) and falls back to the least precise (CHARTDATE) when needed.
    return f"COALESCE({charted_at}, {undated_at_noon}), {p}ROW_ID"


def pull_notes_for_batch(
    conn: duckdb.DuckDBPyConnection, hadm_ids: list[int]
) -> dict[int, list[dict]]:
    """Every source note for these admissions, in charting-time order.

    See ``noteevents_input_filter`` for what counts as a source note.

    Returns ``{hadm_id: [note, ...]}``, one dict per note.
    """
    noteevents = table_ref("NOTEEVENTS.csv.gz")
    rows = conn.execute(
        f"""
        SELECT n.HADM_ID,
               n.ROW_ID,
               n.CHARTDATE::VARCHAR AS chartdate,
               n.CHARTTIME::VARCHAR AS charttime,
               n.CATEGORY AS category,
               n.DESCRIPTION AS description,
               n.TEXT AS text
        FROM {noteevents} n
        -- The batch of admission IDs arrives as a single bound parameter, $1, holding
        -- the whole Python list as one DuckDB array. UNNEST turns that array into rows,
        -- which is what IN needs to match against. Binding one array keeps the query
        -- text fixed no matter how many admissions are in the batch, so DuckDB is never
        -- handed a 500-literal IN list and the IDs are never spliced into SQL by hand.
        WHERE n.HADM_ID IN (SELECT UNNEST($1::INTEGER[]))
          AND {noteevents_input_filter("n")}
        ORDER BY n.HADM_ID, {note_order_sql("n")}
        """,
        [hadm_ids],
    ).fetchall()
    by_adm: dict[int, list[dict]] = {}
    for hid, row_id, chartdate, charttime, category, description, text in rows:
        by_adm.setdefault(int(hid), []).append(
            {
                "row_id": int(row_id),
                "chartdate": chartdate,
                "charttime": charttime,
                "category": category,
                "description": description,
                "text": text,
            }
        )
    return by_adm


def age_at_admission_sql(admit_alias: str = "", patients_alias: str = "") -> str:
    """SQL fragment computing patient age at admission, capped at 90.

    MIMIC-III shifts dates of birth ~300 years for patients older than 89
    to comply with HIPAA. ``LEAST(..., 90)`` collapses those ages to 90.

    Pass aliases (e.g., ``"a"`` for ADMISSIONS, ``"p"`` for PATIENTS) to
    prefix column references when joining multiple tables. The fragment
    expects ``DOB`` on the patients side and ``ADMITTIME`` on the
    admissions side.
    """
    a = f"{admit_alias}." if admit_alias else ""
    p = f"{patients_alias}." if patients_alias else ""
    return f"LEAST(DATE_DIFF('year', {p}DOB::DATE, {a}ADMITTIME::DATE), 90)"


def last_service_per_admission_sql(services_ref: str) -> str:
    """SQL subquery returning (HADM_ID, CURR_SERVICE) for the discharging service.

    A patient may transfer between services during a single admission. The
    discharging service is the one responsible for writing the discharge
    summary, defined here as the ``CURR_SERVICE`` on the latest
    ``TRANSFERTIME`` per ``HADM_ID``.

    ``services_ref`` is a SQL fragment for the SERVICES table (typically
    from ``table_ref('SERVICES.csv.gz')``). The returned fragment is the
    inner SELECT only. Callers wrap it as a subquery (``FROM (...) alias``)
    or as a CTE body, whichever their query needs.
    """
    return f"""SELECT HADM_ID, CURR_SERVICE
        FROM (
            -- Rank each admission's service transfers newest-first
            SELECT HADM_ID, CURR_SERVICE,
                   ROW_NUMBER() OVER (PARTITION BY HADM_ID ORDER BY TRANSFERTIME DESC) AS rn
            FROM {services_ref}
        )
        -- rn = 1 is the latest transfer, i.e. the discharging service
        WHERE rn = 1"""


def create_icustay_admissions(
    conn: duckdb.DuckDBPyConnection, *, metavision_only: bool = False
) -> None:
    """Create a temp table with one row per admission and ICU charting system.

    ICUSTAYS has one row per ICU stay, but an admission can have multiple
    stays. This table reduces them to one row per (HADM_ID, DBSOURCE), so an
    admission with stays in both systems keeps a row for each.

    Creates: temp table ``icustay_admissions`` with columns HADM_ID, DBSOURCE.
    """
    if metavision_only:
        where = "WHERE DBSOURCE = 'metavision'"
    else:
        # Exclude 'both': 108 transition-period stays, too few to change any result.
        where = "WHERE DBSOURCE IN ('carevue', 'metavision')"

    conn.execute(f"""
        CREATE TEMP TABLE icustay_admissions AS
        SELECT HADM_ID, DBSOURCE
        FROM {table_ref("ICUSTAYS.csv.gz")}
        {where}
        -- GROUP BY deduplicates the ICU stays an admission has in the same
        -- charting system.
        GROUP BY HADM_ID, DBSOURCE
    """)


def create_eligible_pool(conn: duckdb.DuckDBPyConnection) -> int:
    """Create the eligible pool: MetaVision admissions with a valid BHC.

    Creates: temp tables ``icustay_admissions`` (MetaVision only) and
    ``eligible_pool`` with admission identifiers, BHC quality metrics, and
    BHC structural metadata (category, sub-format details).

    Returns the number of eligible admissions.
    """
    # Step 1: Deduplicate ICUSTAYS to one row per MetaVision admission.
    create_icustay_admissions(conn, metavision_only=True)

    # Step 2: Join BHC results against MetaVision admissions and apply
    # quality filters.
    conn.execute(f"""
        CREATE TEMP TABLE eligible_pool AS
        SELECT b.hadm_id, b.subject_id, b.bhc_char_count, b.bhc_word_count,
               -- BHC organizational structure: top-level category and
               -- sub-format details from the extraction pipeline.
               b.bhc_structure.category AS bhc_category,
               b.bhc_structure.numbering_convention AS bhc_numbering_convention,
               COALESCE(b.bhc_structure.modifiers, []) AS bhc_modifiers,
               -- Ground-truth BHC text, carried on the temp table so it can
               -- be tokenized per model without re-reading the JSONL. Not
               -- written to the JSON output. The BHC JSONL on disk holds the
               -- text.
               b.bhc_text AS bhc_text
        FROM {bhc_ref()} b
        -- JOIN (not LEFT JOIN): only keep admissions that have an ICU stay
        -- in MetaVision. Admissions without ICU stays are out of scope, since
        -- MIMIC-III records the charting system with each ICU stay.
        JOIN icustay_admissions i ON b.hadm_id = i.HADM_ID
        -- BHC quality filters (carried forward from the extraction pipeline):
        WHERE b.extraction_success           -- recognizable BHC header pattern found
          AND b.bhc_char_count >= 100        -- exclude trivially short extractions
          AND NOT list_contains(             -- exclude notes flagged as entered in error
                b.quality_flags, 'iserror')  -- (quality_flags is a list/array column)
    """)
    return conn.execute("SELECT COUNT(*) FROM eligible_pool").fetchone()[0]


def create_filtered_pool(conn: duckdb.DuckDBPyConnection) -> int:
    """Create the filtered pool by applying threshold filters to the eligible pool.

    Builds on ``create_eligible_pool`` (MetaVision + valid BHC), then applies:

    1. LOS >= 3 days (task-driven: need a hospital course to summarize)
    2. Distinct charting days >= 3 (documentation spread over time, so one burst of
       charting on a single day does not qualify)
    3. BHC <= 15,000 chars (outlier removal)
    4. Non-DS notes >= 2 and labs >= 10 (safety nets)

    Creates: temp tables ``icustay_admissions``, ``eligible_pool``, and
    ``filtered_pool`` with everything from ``eligible_pool`` plus enrichment
    columns (LOS, note/lab/charting-day counts) used by threshold filters.

    Returns the number of admissions in the filtered pool.
    """
    # Step 1: Build the eligible pool (MetaVision + valid BHC).
    create_eligible_pool(conn)

    noteevents = table_ref("NOTEEVENTS.csv.gz")
    admissions_ref = table_ref("ADMISSIONS.csv.gz")
    labevents = table_ref("LABEVENTS.csv.gz")
    patients_ref = table_ref("PATIENTS.csv.gz")
    services_ref = table_ref("SERVICES.csv.gz")

    # Step 2: Admission dates, LOS, and full timestamps.
    # ADMITTIME and DISCHTIME are kept as VARCHAR so they serialize cleanly to
    # JSON, without DuckDB-specific timestamp types in the cohort schema.
    conn.execute(f"""
        CREATE TEMP TABLE cohort_admit_dates AS
        SELECT HADM_ID,
               -- Calendar day boundaries crossed, not elapsed hours.
               DATE_DIFF('day', ADMITTIME::TIMESTAMP, DISCHTIME::TIMESTAMP)
                   AS los_days,
               ADMITTIME::VARCHAR AS admittime,
               DISCHTIME::VARCHAR AS dischtime,
               -- In-hospital death flag (0/1).
               HOSPITAL_EXPIRE_FLAG::INTEGER AS hospital_expire_flag
        FROM {admissions_ref}
    """)

    # Step 2b: Patient demographics (gender, age at admission).
    conn.execute(f"""
        CREATE TEMP TABLE cohort_patient_demo AS
        SELECT a.HADM_ID,
               p.GENDER AS gender,
               {age_at_admission_sql(admit_alias="a", patients_alias="p")} AS age
        FROM {admissions_ref} a
        JOIN {patients_ref} p USING (SUBJECT_ID)
    """)

    # Step 2c: Discharging service per admission.
    # One row per admission, the discharging service.
    conn.execute(f"""
        CREATE TEMP TABLE cohort_service AS
        SELECT HADM_ID, CURR_SERVICE AS service
        FROM ({last_service_per_admission_sql(services_ref)}) ls
    """)

    # Step 3: Non-DS note counts per admission, excluding errors.
    # Includes pre-admission notes (e.g., ED notes before formal admission).
    conn.execute(f"""
        CREATE TEMP TABLE cohort_note_counts AS
        SELECT n.HADM_ID, COUNT(*) AS note_count
        FROM {noteevents} n
        WHERE {noteevents_input_filter("n")}
          AND n.HADM_ID IN (SELECT hadm_id FROM eligible_pool)
        GROUP BY n.HADM_ID
    """)

    # Step 4: Distinct charting days per admission (days with at least one
    # non-DS, non-error note).
    # This is a proxy for temporal signal: if all notes are on the same day,
    # there's no temporal progression to summarize.
    # Requires the same filters as step 3 to ensure consistency with note
    # counts and exclude error notes.
    conn.execute(f"""
        CREATE TEMP TABLE cohort_charting_days AS
        SELECT n.HADM_ID, COUNT(DISTINCT n.CHARTDATE) AS charting_days
        FROM {noteevents} n
        WHERE {noteevents_input_filter("n")}
          AND n.CHARTDATE IS NOT NULL
          AND n.HADM_ID IN (SELECT hadm_id FROM eligible_pool)
        GROUP BY n.HADM_ID
    """)

    # Step 5: Lab counts per admission, with abnormal-only sibling count.
    # lab_count: indicator of temporal signal.
    # lab_abnormal_count: strict FLAG = 'abnormal' count. It excludes unflagged rows
    # and 'delta', which flags a change from the previous value for that test.
    conn.execute(f"""
        CREATE TEMP TABLE cohort_lab_counts AS
        SELECT HADM_ID,
               COUNT(*) AS lab_count,
               COUNT(*) FILTER (WHERE FLAG = 'abnormal') AS lab_abnormal_count
        FROM {labevents}
        GROUP BY HADM_ID
    """)

    # Step 5b: Prescription counts per admission.
    # Counted here so the pool has its own record properties (one
    # authoritative number per admission). Not used as a filter threshold.
    prescriptions = table_ref("PRESCRIPTIONS.csv.gz")
    conn.execute(f"""
        CREATE TEMP TABLE cohort_rx_counts AS
        SELECT HADM_ID, COUNT(*) AS prescription_count
        FROM {prescriptions}
        GROUP BY HADM_ID
    """)

    # Step 6: Join into enriched table and apply all filters.
    conn.execute("""
        CREATE TEMP TABLE filtered_pool AS
        SELECT
            e.hadm_id,
            e.subject_id,
            e.bhc_char_count,
            e.bhc_word_count,
            e.bhc_text,
            e.bhc_category,
            e.bhc_numbering_convention,
            e.bhc_modifiers,
            COALESCE(ad.los_days, 0) AS los_days,
            COALESCE(nc.note_count, 0) AS note_count,
            COALESCE(cd.charting_days, 0) AS charting_days,
            COALESCE(lc.lab_count, 0) AS lab_count,
            COALESCE(lc.lab_abnormal_count, 0) AS lab_abnormal_count,
            COALESCE(rx.prescription_count, 0) AS prescription_count,
            ad.admittime,
            ad.dischtime,
            ad.hospital_expire_flag,
            pd.gender,
            pd.age,
            sv.service
        FROM eligible_pool e
        LEFT JOIN cohort_admit_dates ad ON e.hadm_id = ad.HADM_ID
        LEFT JOIN cohort_note_counts nc ON e.hadm_id = nc.HADM_ID
        LEFT JOIN cohort_charting_days cd ON e.hadm_id = cd.HADM_ID
        LEFT JOIN cohort_lab_counts lc ON e.hadm_id = lc.HADM_ID
        LEFT JOIN cohort_rx_counts rx ON e.hadm_id = rx.HADM_ID
        LEFT JOIN cohort_patient_demo pd ON e.hadm_id = pd.HADM_ID
        LEFT JOIN cohort_service sv ON e.hadm_id = sv.HADM_ID
        WHERE COALESCE(ad.los_days, 0) >= 3           -- enough days for temporal progression
          AND COALESCE(cd.charting_days, 0) >= 3      -- minimal charting data for temporal signal
          AND COALESCE(nc.note_count, 0) >= 2         -- minimum source material for generation
          AND COALESCE(lc.lab_count, 0) >= 10         -- minimum structured data for context
          AND e.bhc_char_count <= 15000               -- outlier removal
    """)

    # Clean up intermediate tables.
    for t in [
        "cohort_admit_dates",
        "cohort_note_counts",
        "cohort_charting_days",
        "cohort_lab_counts",
        "cohort_rx_counts",
        "cohort_patient_demo",
        "cohort_service",
    ]:
        conn.execute(f"DROP TABLE IF EXISTS {t}")

    return conn.execute("SELECT COUNT(*) FROM filtered_pool").fetchone()[0]
