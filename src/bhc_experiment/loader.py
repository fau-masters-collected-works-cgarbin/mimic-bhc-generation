"""Data loader for experiment admission bundles.

Two-phase loader: prepare_cohort() batch-loads MIMIC-III tables into DuckDB
temp tables (one scan per large CSV.gz file), then load_admission_bundle() returns a
single AdmissionBundle from the temp tables.

The two-phase design exists for performance. NOTEEVENTS (~2M rows) and
LABEVENTS (~27M rows) are compressed CSV files that require a full file scan
per query. Loading one admission at a time would mean scanning these files
once per admission.

The loader returns all records with flag columns. It does not filter by
content, because the right filter varies by experiment.

Tables not loaded: CHARTEVENTS (vitals) and INPUTEVENTS (fluids). They are the
highest-volume MIMIC-III tables. The working hypothesis is that their
clinically relevant signal is already abstracted into the notes.

Naming convention: MIMIC-III column names are UPPERCASE, matching MIMIC-III
documentation and DuckDB debugging. Computed flags and derived values are
lowercase.

Ground-truth BHC comes from the extraction JSONL via DuckDB (read_json_auto).
That file is the interface between repos, avoiding a cross-repo runtime
dependency.
"""

from dataclasses import dataclass

import duckdb

from mimic_data_prep.mimic import (
    age_at_admission_sql,
    bhc_ref,
    last_service_per_admission_sql,
    note_order_sql,
    table_ref,
)


@dataclass(frozen=True)
class NoteEvent:
    """A single note from NOTEEVENTS."""

    ROW_ID: int
    CHARTDATE: str | None
    CHARTTIME: str | None
    CATEGORY: str
    TEXT: str
    ISERROR: str | None
    is_pre_admission: bool
    is_discharge_summary: bool
    # Not shown in the prompt. It is loaded because the radiology dedup rule
    # pairs a preliminary read with its final report by chart date and description.
    DESCRIPTION: str | None = None


@dataclass(frozen=True)
class LabEvent:
    """A single lab result from LABEVENTS.

    Carries VALUE (raw result string) but not VALUENUM. About 10% of
    LABEVENTS rows are qualitative (microscopy ranges like "11-20",
    categorical results like NEG, POS, NORMAL) and have no VALUENUM.
    Loading VALUE keeps those rows available to the experiment.
    """

    ROW_ID: int
    CHARTTIME: str | None
    ITEMID: int
    LABEL: str  # Human-readable test name from D_LABITEMS (e.g., "Creatinine")
    VALUE: str | None
    VALUEUOM: str | None
    FLAG: str | None


@dataclass(frozen=True)
class Prescription:
    """A single prescription from PRESCRIPTIONS."""

    ROW_ID: int
    STARTDATE: str | None
    ENDDATE: str | None
    DRUG: str
    DOSE_VAL_RX: str | None
    DOSE_UNIT_RX: str | None
    ROUTE: str | None


@dataclass(frozen=True)
class BhcStructure:
    """BHC organizational structure from the extraction pipeline.

    Fields mirror bhc_structure in bhc.jsonl (see mimic-bhc-extraction
    StructureResult for the authoritative definitions).
    """

    category: str
    numbering_convention: str | None
    modifiers: list[str]


@dataclass(frozen=True)
class AdmissionMetadata:
    """Admission demographics from ADMISSIONS + PATIENTS + SERVICES."""

    HADM_ID: int
    SUBJECT_ID: int
    ADMITTIME: str
    DISCHTIME: str
    GENDER: str
    SERVICE: str | None  # CURR_SERVICE from SERVICES.csv.gz. None if no SERVICES row
    los_days: int
    age: int


@dataclass(frozen=True)
class AdmissionBundle:
    """All data for a single admission, ready for the experiment pipeline.

    Records are chronologically ordered within each list.
    """

    admission: AdmissionMetadata
    ground_truth_bhc: str
    bhc_category: str
    bhc_structure: BhcStructure
    noteevents: list[NoteEvent]
    labevents: list[LabEvent]
    prescriptions: list[Prescription]


def prepare_cohort(conn: duckdb.DuckDBPyConnection, hadm_ids: list[int]) -> None:
    """Load only the records for the requested admissions into temp tables.

    Can be called again on the same connection, e.g. to restart a run
    without reconnecting.

    Temp tables, so the phase-1 result stays queryable on the same connection
    when debugging one admission.
    """
    noteevents = table_ref("NOTEEVENTS.csv.gz")
    admissions_ref = table_ref("ADMISSIONS.csv.gz")
    patients_ref = table_ref("PATIENTS.csv.gz")
    services_ref = table_ref("SERVICES.csv.gz")
    labevents = table_ref("LABEVENTS.csv.gz")
    d_labitems = table_ref("D_LABITEMS.csv.gz")
    prescriptions_ref = table_ref("PRESCRIPTIONS.csv.gz")
    bhc = bhc_ref()

    # Noteevents: all notes for the cohort, with computed flag columns.
    # JOIN to ADMISSIONS for ADMITTIME (needed for is_pre_admission flag).
    # No content filtering: error notes, pre-admission notes, and discharge
    # summaries are all included with their respective flags.
    conn.execute(
        f"""
        -- OR REPLACE: safe to re-call on the same connection (e.g., run restart).
        CREATE OR REPLACE TEMP TABLE cohort_noteevents AS
        SELECT n.ROW_ID, n.HADM_ID,
               n.CHARTDATE::VARCHAR AS CHARTDATE,
               n.CHARTTIME::VARCHAR AS CHARTTIME,
               n.CATEGORY, n.DESCRIPTION, n.TEXT,
               -- Cast ISERROR to VARCHAR because DuckDB may infer it as INTEGER
               -- from CSV (most rows are NULL, the non-NULL value is '1').
               CAST(n.ISERROR AS VARCHAR) AS ISERROR,
               -- Flag notes dated before the admission date.
               CASE WHEN n.CHARTDATE < a.ADMITTIME::DATE THEN TRUE ELSE FALSE END
                   AS is_pre_admission,
               -- Flag discharge summaries because these are the notes with BHC, what we will
               -- generate (adding them would contaminate the input with the output).
               CASE WHEN n.CATEGORY = 'Discharge summary' THEN TRUE ELSE FALSE END
                   AS is_discharge_summary
        FROM {noteevents} n
        JOIN {admissions_ref} a ON n.HADM_ID = a.HADM_ID
        -- UNNEST passes the list as a query parameter instead of a fixed list to
        -- avoid SQL injection and allow DuckDB to cache the query plan across calls.
        WHERE n.HADM_ID IN (SELECT UNNEST($1::INT[]))
        ORDER BY n.HADM_ID, {note_order_sql("n")}
        """,
        [hadm_ids],
    )

    # Labevents: all labs for the cohort, chronological.
    # JOIN D_LABITEMS for human-readable test name (e.g., "Creatinine").
    # Without it, the LLM only sees a numeric value and unit with no
    # indication of what lab test it represents.
    # Inner JOIN is safe: D_LABITEMS is the dictionary for every LABEVENTS.ITEMID,
    # so no lab row is dropped.
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE cohort_labevents AS
        SELECT l.ROW_ID, l.HADM_ID,
               l.CHARTTIME::VARCHAR AS CHARTTIME,
               l.ITEMID, d.LABEL,
               l.VALUE, l.VALUEUOM, l.FLAG
        FROM {labevents} l
        JOIN {d_labitems} d ON l.ITEMID = d.ITEMID
        WHERE l.HADM_ID IN (SELECT UNNEST($1::INT[]))
        -- ROW_ID is the final tiebreaker so labs sharing a CHARTTIME come back
        -- in a fixed, reproducible order across runs.
        ORDER BY l.HADM_ID, l.CHARTTIME NULLS LAST, l.ROW_ID
        """,
        [hadm_ids],
    )

    # Prescriptions: all prescriptions for the cohort, chronological.
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE cohort_prescriptions AS
        SELECT ROW_ID, HADM_ID,
               -- MIMIC stores these as midnight datetimes ("2150-01-03
               -- 00:00:00"). Cast to DATE so these stay date-only values.
               STARTDATE::DATE::VARCHAR AS STARTDATE,
               ENDDATE::DATE::VARCHAR AS ENDDATE,
               DRUG, DOSE_VAL_RX, DOSE_UNIT_RX, ROUTE
        FROM {prescriptions_ref}
        WHERE HADM_ID IN (SELECT UNNEST($1::INT[]))
        -- ROW_ID is the final tiebreaker so prescriptions sharing a STARTDATE
        -- come back in a fixed, reproducible order across runs.
        ORDER BY HADM_ID, STARTDATE NULLS LAST, ROW_ID
        """,
        [hadm_ids],
    )

    # Admissions: demographics with computed LOS, age, and discharging service.
    # age_at_admission_sql and last_service_per_admission_sql centralize the
    # MIMIC-III age cap and the ROW_NUMBER discharging-service pattern shared
    # with the cohort sampler.
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE cohort_admissions AS
        SELECT
            a.HADM_ID, a.SUBJECT_ID,
            a.ADMITTIME::VARCHAR AS ADMITTIME,
            a.DISCHTIME::VARCHAR AS DISCHTIME,
            p.GENDER,
            s.CURR_SERVICE AS SERVICE,
            -- DATE_DIFF('day', ...) counts calendar day boundaries crossed, not
            -- elapsed hours. Same-day admit/discharge = 0.
            DATE_DIFF('day', a.ADMITTIME::TIMESTAMP, a.DISCHTIME::TIMESTAMP) AS los_days,
            {age_at_admission_sql(admit_alias="a", patients_alias="p")} AS age
        FROM {admissions_ref} a
        JOIN {patients_ref} p ON a.SUBJECT_ID = p.SUBJECT_ID
        LEFT JOIN ({last_service_per_admission_sql(services_ref)}) s
            ON a.HADM_ID = s.HADM_ID
        WHERE a.HADM_ID IN (SELECT UNNEST($1::INT[]))
        """,
        [hadm_ids],
    )

    # BHC ground truth: text, format category, and full structure metadata.
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE cohort_bhc AS
        SELECT hadm_id, bhc_text,
               bhc_structure.category AS bhc_category,
               bhc_structure.numbering_convention AS numbering_convention,
               -- COALESCE so modifiers is always a list, never NULL.
               COALESCE(bhc_structure.modifiers, []) AS modifiers
        FROM {bhc}
        WHERE hadm_id IN (SELECT UNNEST($1::INT[]))
        """,
        [hadm_ids],
    )


def _fetch_one_row(conn: duckdb.DuckDBPyConnection, sql: str, params: list, *, label: str) -> tuple:
    """Execute a query expected to return exactly one row, raising ValueError otherwise."""
    rows = conn.execute(sql, params).fetchall()
    if len(rows) == 0:
        msg = f"{label}: no row found"
        raise ValueError(msg)
    if len(rows) > 1:
        msg = f"{label}: expected 1 row, got {len(rows)}"
        raise ValueError(msg)
    return rows[0]


def load_admission_bundle(conn: duckdb.DuckDBPyConnection, hadm_id: int) -> AdmissionBundle:
    """Load a single AdmissionBundle from the prepared temp tables.

    Call prepare_cohort() first to create the temp tables.

    Raises ValueError if the HADM_ID is not found or has duplicate rows.
    """
    # Admission metadata
    adm_row = _fetch_one_row(
        conn,
        "SELECT HADM_ID, SUBJECT_ID, ADMITTIME, DISCHTIME, GENDER, SERVICE, los_days, age "
        "FROM cohort_admissions WHERE HADM_ID = $1",
        [hadm_id],
        label=f"HADM_ID {hadm_id} in cohort_admissions",
    )
    admission = AdmissionMetadata(
        HADM_ID=adm_row[0],
        SUBJECT_ID=adm_row[1],
        ADMITTIME=adm_row[2],
        DISCHTIME=adm_row[3],
        GENDER=adm_row[4],
        SERVICE=adm_row[5],
        los_days=adm_row[6],
        age=adm_row[7],
    )

    # BHC ground truth
    bhc_row = _fetch_one_row(
        conn,
        "SELECT bhc_text, bhc_category, numbering_convention, modifiers "
        "FROM cohort_bhc WHERE hadm_id = $1",
        [hadm_id],
        label=f"HADM_ID {hadm_id} in cohort_bhc",
    )
    ground_truth_bhc = bhc_row[0]
    bhc_category = bhc_row[1]
    bhc_structure = BhcStructure(
        category=bhc_category,
        numbering_convention=bhc_row[2],
        modifiers=list(bhc_row[3]) if bhc_row[3] is not None else [],
    )

    # Notes
    # Timestamps are already VARCHAR from prepare_cohort. NULL stays None.
    note_rows = conn.execute(
        "SELECT ROW_ID, CHARTDATE, CHARTTIME, CATEGORY, DESCRIPTION, TEXT, ISERROR, "
        "is_pre_admission, is_discharge_summary "
        "FROM cohort_noteevents WHERE HADM_ID = $1 "
        f"ORDER BY {note_order_sql()}",
        [hadm_id],
    ).fetchall()
    noteevents = [
        NoteEvent(
            ROW_ID=r[0],
            CHARTDATE=r[1],
            CHARTTIME=r[2],
            CATEGORY=r[3],
            DESCRIPTION=r[4],
            TEXT=r[5],
            ISERROR=r[6],
            is_pre_admission=bool(r[7]),
            is_discharge_summary=bool(r[8]),
        )
        for r in note_rows
    ]

    # Labs
    lab_rows = conn.execute(
        "SELECT ROW_ID, CHARTTIME, ITEMID, LABEL, VALUE, VALUEUOM, FLAG "
        "FROM cohort_labevents WHERE HADM_ID = $1 "
        "ORDER BY CHARTTIME NULLS LAST, ROW_ID",
        [hadm_id],
    ).fetchall()
    labevents = [
        LabEvent(
            ROW_ID=r[0],
            CHARTTIME=r[1],
            ITEMID=r[2],
            LABEL=r[3],
            VALUE=r[4],
            VALUEUOM=r[5],
            FLAG=r[6],
        )
        for r in lab_rows
    ]

    # Prescriptions
    rx_rows = conn.execute(
        "SELECT ROW_ID, STARTDATE, ENDDATE, DRUG, DOSE_VAL_RX, DOSE_UNIT_RX, ROUTE "
        "FROM cohort_prescriptions WHERE HADM_ID = $1 "
        "ORDER BY STARTDATE NULLS LAST, ROW_ID",
        [hadm_id],
    ).fetchall()
    prescriptions = [
        Prescription(
            ROW_ID=r[0],
            STARTDATE=r[1],
            ENDDATE=r[2],
            DRUG=r[3],
            DOSE_VAL_RX=r[4],
            DOSE_UNIT_RX=r[5],
            ROUTE=r[6],
        )
        for r in rx_rows
    ]

    return AdmissionBundle(
        admission=admission,
        ground_truth_bhc=ground_truth_bhc,
        bhc_category=bhc_category,
        bhc_structure=bhc_structure,
        noteevents=noteevents,
        labevents=labevents,
        prescriptions=prescriptions,
    )


def load_reference_texts(hadm_ids: list[int]) -> dict[int, str]:
    """Load each admission's reference Brief Hospital Course, keyed by HADM_ID."""
    conn = duckdb.connect()
    try:
        prepare_cohort(conn, hadm_ids)
        texts: dict[int, str] = {}
        for hadm_id in hadm_ids:
            texts[hadm_id] = load_admission_bundle(conn, hadm_id).ground_truth_bhc
        return texts
    finally:
        conn.close()
