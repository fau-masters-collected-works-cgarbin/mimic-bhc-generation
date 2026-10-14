"""Tests for spot_check_queries using synthetic JSONL fixtures."""

import csv
import gzip
import io
import json
from pathlib import Path

import duckdb
import pytest

_BHC_RECORDS = [
    {
        "hadm_id": 100,
        "subject_id": 10,
        "bhc_text": "Patient was admitted for pneumonia.",
        "bhc_char_count": 35,
        "bhc_word_count": 5,
        "bhc_start": 50,
        "bhc_end": 85,
        "note_length": 130,
        "bhc_structure": {
            "category": "narrative",
            "numbering_convention": None,
            "modifiers": [],
            "nicu_domain_count": 0,
            "labeled_section_count": 0,
        },
        "extraction_success": True,
        "failure_reason": None,
        "start_header": "Brief Hospital Course:",
        "end_header": "Medications on Discharge:",
        "quality_flags": [],
        "row_id": "1001",
        "chartdate": "2150-01-05",
        "note_row_count": 1,
    },
    {
        "hadm_id": 200,
        "subject_id": 20,
        "bhc_text": "Patient presented with chest pain.",
        "bhc_char_count": 34,
        "bhc_word_count": 5,
        "bhc_start": 50,
        "bhc_end": 84,
        "note_length": 129,
        "bhc_structure": {
            "category": "domain_organized",
            "numbering_convention": "hash",
            "modifiers": [],
            "nicu_domain_count": 0,
            "labeled_section_count": 3,
        },
        "extraction_success": True,
        "failure_reason": None,
        "start_header": "Brief Hospital Course:",
        "end_header": "Medications on Discharge:",
        "quality_flags": [],
        "row_id": "2001",
        "chartdate": "2150-02-10",
        "note_row_count": 1,
    },
]

# Discharge note text served by the NOTEEVENTS fixture.
_NOTE_TEXTS = {
    100: (
        "Discharge Summary\n\nAdmission Date: ...\n\n"
        "Brief Hospital Course:\nPatient was admitted for pneumonia."
        "\n\nMedications on Discharge: ..."
    ),
    200: (
        "Discharge Summary\n\nAdmission Date: ...\n\n"
        "Brief Hospital Course:\nPatient presented with chest pain."
        "\n\nMedications on Discharge: ..."
    ),
}


@pytest.fixture
def bhc_jsonl(tmp_path: Path) -> Path:
    """bhc.jsonl in the production extraction format (note text lives in NOTEEVENTS)."""
    path = tmp_path / "bhc.jsonl"
    with path.open("w") as f:
        for r in _BHC_RECORDS:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def discharge_noteevents_csv(tmp_path: Path) -> Path:
    """NOTEEVENTS with discharge summaries matching bhc_jsonl row_ids."""
    path = tmp_path / "NOTEEVENTS_DS.csv.gz"
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "ROW_ID",
            "SUBJECT_ID",
            "HADM_ID",
            "CHARTDATE",
            "CHARTTIME",
            "STORETIME",
            "CATEGORY",
            "DESCRIPTION",
            "CGID",
            "ISERROR",
            "TEXT",
        ]
    )
    writer.writerow(
        [
            1001,
            10,
            100,
            "2150-01-05",
            "2150-01-05 12:00:00",
            "",
            "Discharge summary",
            "Report",
            2,
            "",
            _NOTE_TEXTS[100],
        ]
    )
    writer.writerow(
        [
            2001,
            20,
            200,
            "2150-02-10",
            "2150-02-10 12:00:00",
            "",
            "Discharge summary",
            "Report",
            2,
            "",
            _NOTE_TEXTS[200],
        ]
    )
    with gzip.open(path, "wt") as f:
        f.write(buf.getvalue())
    return path


def test_fetch_bhc_data_batch(bhc_jsonl: Path, discharge_noteevents_csv: Path):
    """Note text is fetched from NOTEEVENTS via row_id, in a single
    NOTEEVENTS scan for all requested HADM_IDs."""
    from mimic_data_prep.spot_check_queries import fetch_bhc_data_batch

    conn = duckdb.connect()
    result = fetch_bhc_data_batch(conn, bhc_jsonl, discharge_noteevents_csv, hadm_ids=[100, 200])

    assert set(result) == {100, 200}
    assert result[100]["bhc_text"] == "Patient was admitted for pneumonia."
    assert result[100]["bhc_start"] == 50
    assert result[100]["bhc_end"] == 85
    assert result[100]["note_text"].startswith("Discharge Summary")
    assert result[100]["bhc_char_count"] == 35
    assert result[100]["bhc_word_count"] == 5
    assert result[100]["bhc_category"] == "narrative"
    assert result[200]["note_text"].startswith("Discharge Summary")
    assert result[200]["bhc_category"] == "domain_organized"


def test_fetch_bhc_data_batch_omits_missing(bhc_jsonl: Path, discharge_noteevents_csv: Path):
    """Missing HADM_IDs are omitted from the result (not returned as None)."""
    from mimic_data_prep.spot_check_queries import fetch_bhc_data_batch

    conn = duckdb.connect()
    result = fetch_bhc_data_batch(conn, bhc_jsonl, discharge_noteevents_csv, hadm_ids=[100, 999])

    assert set(result) == {100}


@pytest.fixture
def patients_csv(tmp_path: Path) -> Path:
    """PATIENTS table with two patients. Subject 20 has shifted DOB (age > 89)."""
    path = tmp_path / "PATIENTS.csv.gz"
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "ROW_ID",
            "SUBJECT_ID",
            "GENDER",
            "DOB",
            "DOD",
            "DOD_HOSP",
            "DOD_SSN",
            "EXPIRE_FLAG",
        ]
    )
    writer.writerow([1, 10, "M", "2080-01-01 00:00:00", "", "", "", 0])
    writer.writerow([2, 20, "F", "1850-01-01 00:00:00", "", "", "", 0])
    with gzip.open(path, "wt") as f:
        f.write(buf.getvalue())
    return path


@pytest.fixture
def admissions_csv(tmp_path: Path) -> Path:
    """ADMISSIONS table with one admission per patient."""
    path = tmp_path / "ADMISSIONS.csv.gz"
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "ROW_ID",
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
            "DISCHTIME",
            "DEATHTIME",
            "ADMISSION_TYPE",
            "ADMISSION_LOCATION",
            "DISCHARGE_LOCATION",
            "INSURANCE",
            "LANGUAGE",
            "RELIGION",
            "MARITAL_STATUS",
            "ETHNICITY",
            "EDREGTIME",
            "EDOUTTIME",
            "DIAGNOSIS",
            "HOSPITAL_EXPIRE_FLAG",
            "HAS_CHARTEVENTS_DATA",
        ]
    )
    writer.writerow(
        [
            1,
            10,
            100,
            "2150-01-01 08:00:00",
            "2150-01-17 14:00:00",
            "",
            "EMERGENCY",
            "EMERGENCY ROOM ADMIT",
            "HOME",
            "Medicare",
            "",
            "",
            "",
            "WHITE",
            "",
            "",
            "PNEUMONIA",
            0,
            1,
        ]
    )
    writer.writerow(
        [
            2,
            20,
            200,
            "2150-02-05 10:00:00",
            "2150-02-12 11:00:00",
            "",
            "ELECTIVE",
            "PHYS REFERRAL",
            "HOME",
            "Private",
            "",
            "",
            "",
            "BLACK",
            "",
            "",
            "CHEST PAIN",
            0,
            1,
        ]
    )
    with gzip.open(path, "wt") as f:
        f.write(buf.getvalue())
    return path


def test_fetch_demographics_batch(patients_csv: Path, admissions_csv: Path):
    """Batch returns demographics for every requested HADM_ID. HADM 200 is the
    age-cap case (DOB 1850 → 300 years, capped at 90)."""
    from mimic_data_prep.spot_check_queries import fetch_demographics_batch

    conn = duckdb.connect()
    result = fetch_demographics_batch(conn, patients_csv, admissions_csv, hadm_ids=[100, 200])

    assert set(result) == {100, 200}
    assert result[100]["gender"] == "M"
    assert result[100]["age"] == 70  # 2150 - 2080
    assert "2150-01-01" in result[100]["admittime"]
    assert "2150-01-17" in result[100]["dischtime"]
    assert result[100]["los_days"] == 16
    assert result[200]["age"] == 90  # 2150 - 1850 = 300, capped


def test_fetch_demographics_batch_omits_missing(patients_csv: Path, admissions_csv: Path):
    """Missing HADM_IDs are omitted from the result."""
    from mimic_data_prep.spot_check_queries import fetch_demographics_batch

    conn = duckdb.connect()
    result = fetch_demographics_batch(conn, patients_csv, admissions_csv, hadm_ids=[100, 999])

    assert set(result) == {100}


@pytest.fixture
def noteevents_csv(tmp_path: Path) -> Path:
    """NOTEEVENTS with 4 notes: 2 Nursing, 1 Physician, 1 Discharge summary. Plus 1 error."""
    path = tmp_path / "NOTEEVENTS.csv.gz"
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "ROW_ID",
            "SUBJECT_ID",
            "HADM_ID",
            "CHARTDATE",
            "CHARTTIME",
            "STORETIME",
            "CATEGORY",
            "DESCRIPTION",
            "CGID",
            "ISERROR",
            "TEXT",
        ]
    )
    writer.writerow(
        [
            1,
            10,
            100,
            "2150-01-01",
            "2150-01-01 10:00:00",
            "",
            "Nursing",
            "Nursing Note",
            1,
            "",
            "Patient resting comfortably.",
        ]
    )
    writer.writerow(
        [
            2,
            10,
            100,
            "2150-01-03",
            "2150-01-03 08:00:00",
            "",
            "Nursing",
            "Nursing Note",
            1,
            "",
            "Patient improving.",
        ]
    )
    writer.writerow(
        [
            3,
            10,
            100,
            "2150-01-03",
            "2150-01-03 14:00:00",
            "",
            "Physician ",
            "Physician Note",
            2,
            "",
            "Assessment and plan updated.",
        ]
    )
    writer.writerow(
        [
            4,
            10,
            100,
            "2150-01-17",
            "2150-01-17 12:00:00",
            "",
            "Discharge summary",
            "Report",
            2,
            "",
            "Discharge summary text.",
        ]
    )
    # Error note (should be excluded).
    writer.writerow(
        [
            5,
            10,
            100,
            "2150-01-02",
            "2150-01-02 09:00:00",
            "",
            "Nursing",
            "Nursing Note",
            1,
            "1",
            "This was entered in error.",
        ]
    )
    with gzip.open(path, "wt") as f:
        f.write(buf.getvalue())
    return path


def test_fetch_notes_batch_returns_all_unfiltered(noteevents_csv: Path, admissions_csv: Path):
    """All notes are returned per admission, including the error note, nothing filtered."""
    from mimic_data_prep.spot_check_queries import fetch_notes_batch

    conn = duckdb.connect()
    result = fetch_notes_batch(conn, noteevents_csv, admissions_csv, hadm_ids=[100])

    df = result[100]
    assert len(df) == 5
    assert "Discharge summary" in df["CATEGORY"].to_numpy()
    assert df["TEXT"].str.contains("entered in error").any()
    assert set(df.columns) >= {
        "ROW_ID",
        "CHARTDATE",
        "CATEGORY",
        "TEXT",
        "note_length",
        "is_error",
        "is_pre_admission",
    }
    assert "HADM_ID" not in df.columns  # dropped in per-admission DataFrame


def test_fetch_notes_batch_flags_error_notes(noteevents_csv: Path, admissions_csv: Path):
    from mimic_data_prep.spot_check_queries import fetch_notes_batch

    conn = duckdb.connect()
    df = fetch_notes_batch(conn, noteevents_csv, admissions_csv, hadm_ids=[100])[100]

    error_rows = df[df["is_error"]]
    assert len(error_rows) == 1
    assert error_rows.iloc[0]["TEXT"] == "This was entered in error."
    assert len(df[~df["is_error"]]) == 4


def test_fetch_notes_batch_flags_pre_admission(noteevents_csv: Path, admissions_csv: Path):
    """Pre-admission notes are returned but flagged."""
    from mimic_data_prep.spot_check_queries import fetch_notes_batch

    conn = duckdb.connect()
    # Add a pre-admission note.
    conn.execute(f"""
        CREATE TABLE pre_admit_notes AS
        SELECT * FROM read_csv_auto('{noteevents_csv}')
        UNION ALL
        SELECT 6, 10, 100, '2149-12-31', '2149-12-31 10:00:00', '',
               'Nursing', 'Nursing Note', 1, '', 'Pre-admission note.'
    """)
    tmp_note_path = noteevents_csv.parent / "NOTEEVENTS_WITH_PRE.csv.gz"
    conn.execute(
        f"COPY pre_admit_notes TO '{tmp_note_path}' (FORMAT CSV, HEADER TRUE, COMPRESSION 'gzip')"
    )

    df = fetch_notes_batch(conn, tmp_note_path, admissions_csv, hadm_ids=[100])[100]
    assert len(df) == 6
    pre_adm = df[df["is_pre_admission"]]
    assert len(pre_adm) == 1
    assert pre_adm.iloc[0]["TEXT"] == "Pre-admission note."


def test_fetch_notes_batch_splits_across_admissions(noteevents_csv: Path, admissions_csv: Path):
    """Batch groups notes by HADM_ID into separate per-admission DataFrames."""
    from mimic_data_prep.spot_check_queries import fetch_notes_batch

    conn = duckdb.connect()
    # Extend the HADM 100 fixture with one note for HADM 200.
    conn.execute(f"""
        CREATE TABLE multi_hadm_notes AS
        SELECT * FROM read_csv_auto('{noteevents_csv}')
        UNION ALL
        SELECT 7, 20, 200, '2150-02-06', '2150-02-06 12:00:00', '',
               'Nursing', 'Nursing Note', 1, '', 'Note for HADM 200.'
    """)
    multi_path = noteevents_csv.parent / "NOTEEVENTS_MULTI_HADM.csv.gz"
    conn.execute(
        f"COPY multi_hadm_notes TO '{multi_path}' (FORMAT CSV, HEADER TRUE, COMPRESSION 'gzip')"
    )

    result = fetch_notes_batch(conn, multi_path, admissions_csv, hadm_ids=[100, 200])

    assert set(result) == {100, 200}
    assert len(result[100]) == 5  # original HADM 100 notes, unaffected
    assert len(result[200]) == 1
    assert result[200].iloc[0]["TEXT"] == "Note for HADM 200."
    assert "HADM_ID" not in result[100].columns
    assert "HADM_ID" not in result[200].columns


@pytest.fixture
def labevents_csv(tmp_path: Path) -> Path:
    """LABEVENTS with 3 rows for HADM_ID 100."""
    path = tmp_path / "LABEVENTS.csv.gz"
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "ROW_ID",
            "SUBJECT_ID",
            "HADM_ID",
            "ITEMID",
            "CHARTTIME",
            "VALUE",
            "VALUENUM",
            "VALUEUOM",
            "FLAG",
        ]
    )
    writer.writerow([1, 10, 100, 50811, "2150-01-01 10:00:00", "12.5", 12.5, "g/dL", ""])
    writer.writerow([2, 10, 100, 50912, "2150-01-02 06:00:00", "1.0", 1.0, "mg/dL", "abnormal"])
    writer.writerow([3, 10, 100, 50811, "2150-01-03 06:00:00", "11.8", 11.8, "g/dL", ""])
    with gzip.open(path, "wt") as f:
        f.write(buf.getvalue())
    return path


def test_fetch_lab_count_batch(labevents_csv: Path):
    """Returns total and abnormal counts for every requested HADM_ID, with
    zeros for admissions that have no labs (unlike the notes batch, which
    omits missing HADM_IDs)."""
    from mimic_data_prep.spot_check_queries import fetch_lab_count_batch

    conn = duckdb.connect()
    result = fetch_lab_count_batch(conn, labevents_csv, hadm_ids=[100, 999])

    assert result == {
        100: {"total": 3, "abnormal": 1},
        999: {"total": 0, "abnormal": 0},
    }


_TOKEN_STRATA = ("4K", "8K", "16K", "32K", "64K", "96K", "112K")


def _make_admissions(n_per_cell=25):
    """Build a cohort-style admission list: n_per_cell per cell of the
    (token_stratum x bhc_category) grid.
    """
    admissions = []
    hadm_counter = 1000
    for stratum in _TOKEN_STRATA:
        for cat in ["narrative", "domain_organized"]:
            for _ in range(n_per_cell):
                admissions.append(
                    {
                        "hadm_id": hadm_counter,
                        "bhc_category": cat,
                        "token_stratum": stratum,
                        "los_quartile": "Q2",
                        "note_category_availability": "clinical_narrative",
                        "los_days": 7,
                    }
                )
                hadm_counter += 1
    return admissions


def test_select_review_cases():
    from collections import Counter

    from mimic_data_prep.spot_check_queries import select_review_cases

    admissions = _make_admissions()
    selected = select_review_cases(admissions, seed=99)

    # 7 strata x 2 formats = 14 grid + 2 extras (96K, 112K) = 16.
    assert len(selected) == 16
    assert len({s["hadm_id"] for s in selected}) == 16

    stratum_counts = Counter(s["token_stratum"] for s in selected)
    # Strata 4K-64K: 2 cases each (one per format). 96K/112K: 3 (grid + extra).
    for stratum in ("4K", "8K", "16K", "32K", "64K"):
        assert stratum_counts[stratum] == 2, f"{stratum} expected 2, got {stratum_counts[stratum]}"
    assert stratum_counts["96K"] == 3
    assert stratum_counts["112K"] == 3

    # Every (stratum, format) cell appears at least once.
    cell_counts = Counter((s["token_stratum"], s["bhc_category"]) for s in selected)
    assert len(cell_counts) == 14


def test_select_review_cases_deterministic():
    from mimic_data_prep.spot_check_queries import select_review_cases

    admissions = _make_admissions()
    a = select_review_cases(admissions, seed=99)
    b = select_review_cases(admissions, seed=99)
    assert [s["hadm_id"] for s in a] == [s["hadm_id"] for s in b]


def test_longest_no_note_gap_no_gap_when_every_day_has_notes():
    from mimic_data_prep.spot_check_queries import longest_no_note_gap

    assert longest_no_note_gap({1, 2, 3, 4, 5}, stay_days=5) == 0


def test_longest_no_note_gap_picks_longest_run():
    from mimic_data_prep.spot_check_queries import longest_no_note_gap

    # Days with notes: 1, 5, 10. Stay length 10.
    # Gaps: days 2-4 (3 days), days 6-9 (4 days). Longest is 4.
    assert longest_no_note_gap({1, 5, 10}, stay_days=10) == 4


def test_longest_no_note_gap_trailing_gap():
    from mimic_data_prep.spot_check_queries import longest_no_note_gap

    # No notes after day 2 in a 7-day stay leaves a trailing gap of 5.
    assert longest_no_note_gap({1, 2}, stay_days=7) == 5


def test_save_and_load_review(tmp_path: Path):
    from mimic_data_prep.spot_check_queries import load_reviews, save_review

    review_path = tmp_path / "reviews.json"
    test_metadata = {"cohort_file_path": "experiment_cohort.json", "selection_seed": 99}

    save_review(
        review_path,
        hadm_id=100,
        coherence={"text_coherent": True, "not_truncated": True, "boundaries_correct": True},
        coverage={"source_notes_present": True, "temporal_spread": True},
        notes="Looks good.",
        metadata=test_metadata,
    )

    data = load_reviews(review_path)
    assert data["metadata"]["selection_seed"] == 99
    assert "100" in data["reviews"]
    review = data["reviews"]["100"]
    assert review["coherence"]["text_coherent"] is True
    assert review["notes"] == "Looks good."
    assert "reviewed_at" in review


def test_save_review_preserves_existing(tmp_path: Path):
    from mimic_data_prep.spot_check_queries import load_reviews, save_review

    review_path = tmp_path / "reviews.json"
    test_metadata = {"cohort_file_path": "experiment_cohort.json", "selection_seed": 99}

    save_review(
        review_path,
        hadm_id=100,
        coherence={"text_coherent": True, "not_truncated": True, "boundaries_correct": True},
        coverage={"source_notes_present": True, "temporal_spread": True},
        notes="First.",
        metadata=test_metadata,
    )
    # Second save without metadata: existing metadata preserved.
    save_review(
        review_path,
        hadm_id=200,
        coherence={"text_coherent": False, "not_truncated": True, "boundaries_correct": True},
        coverage={"source_notes_present": True, "temporal_spread": True},
        notes="Second.",
    )

    data = load_reviews(review_path)
    assert "100" in data["reviews"]
    assert "200" in data["reviews"]
    assert data["metadata"]["selection_seed"] == 99
