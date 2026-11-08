"""Tests for the shared MIMIC module (mimic.py).

Fixtures are synthetic CSV.gz and JSONL files written to a tmp_path.
"""

import gzip
import json
from pathlib import Path

import duckdb
import pytest

from mimic_data_prep import mimic
from mimic_data_prep.cohort import (
    compute_narrative_token_counts,
    compute_pre_admission_notes,
    compute_source_diversity,
    compute_temporal_metrics,
)


@pytest.fixture
def mimic_dir(tmp_path: Path) -> Path:
    """Create a temp directory with synthetic MIMIC-III files."""
    # ICUSTAYS: two MetaVision stays for the same admission (dedup test),
    # one CareVue admission, one 'both' admission.
    icustays_csv = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,DBSOURCE,FIRST_CAREUNIT,"
        "LAST_CAREUNIT,FIRST_WARDID,LAST_WARDID,INTIME,OUTTIME,LOS\n"
        # Admission 100: two MetaVision ICU stays (should dedup to one row)
        "1,10,100,1001,metavision,MICU,MICU,1,1,2150-01-01,2150-01-05,4.0\n"
        "2,10,100,1002,metavision,SICU,SICU,2,2,2150-01-05,2150-01-07,2.0\n"
        # Admission 200: one CareVue stay
        "3,20,200,2001,carevue,CCU,CCU,3,3,2145-03-01,2145-03-04,3.0\n"
        # Admission 300: 'both' (transition period, should be excluded)
        "4,30,300,3001,both,MICU,MICU,1,1,2148-06-01,2148-06-03,2.0\n"
        # Admission 400: one MetaVision stay
        "5,40,400,4001,metavision,TSICU,TSICU,4,4,2151-02-01,2151-02-06,5.0\n"
    )
    path = tmp_path / "ICUSTAYS.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write(icustays_csv)
    return tmp_path


@pytest.fixture
def bhc_jsonl(tmp_path: Path) -> Path:
    """Create a synthetic BHC extraction JSONL file."""
    records = [
        # Admission 100: valid MetaVision BHC
        {
            "hadm_id": 100,
            "subject_id": 10,
            "bhc_char_count": 2500,
            "bhc_word_count": 400,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "narrative",
                "numbering_convention": None,
                "modifiers": [],
            },
        },
        # Admission 200: valid CareVue BHC (should be excluded by MetaVision filter)
        {
            "hadm_id": 200,
            "subject_id": 20,
            "bhc_char_count": 1800,
            "bhc_word_count": 300,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "narrative",
                "numbering_convention": None,
                "modifiers": [],
            },
        },
        # Admission 400: valid MetaVision BHC
        {
            "hadm_id": 400,
            "subject_id": 40,
            "bhc_char_count": 3000,
            "bhc_word_count": 500,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "domain_organized",
                "numbering_convention": "#",
                "modifiers": ["acute_chronic"],
            },
        },
        # Admission 500: MetaVision BHC but too short (< 100 chars)
        {
            "hadm_id": 500,
            "subject_id": 50,
            "bhc_char_count": 50,
            "bhc_word_count": 8,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "narrative",
                "numbering_convention": None,
                "modifiers": [],
            },
        },
        # Admission 600: MetaVision BHC but extraction failed
        {
            "hadm_id": 600,
            "subject_id": 60,
            "bhc_char_count": 0,
            "bhc_word_count": 0,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": False,
            "quality_flags": [],
            "bhc_structure": {
                "category": "narrative",
                "numbering_convention": None,
                "modifiers": [],
            },
        },
        # Admission 700: MetaVision BHC but flagged as error
        {
            "hadm_id": 700,
            "subject_id": 70,
            "bhc_char_count": 2000,
            "bhc_word_count": 350,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": ["iserror"],
            "bhc_structure": {
                "category": "domain_organized",
                "numbering_convention": "N.",
                "modifiers": [],
            },
        },
    ]
    path = tmp_path / "bhc.jsonl"
    with path.open("w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)
    return path


@pytest.fixture
def _patch_paths(monkeypatch: pytest.MonkeyPatch, mimic_dir: Path, bhc_jsonl: Path) -> None:
    """Point mimic module paths at synthetic test data."""
    monkeypatch.setattr(mimic, "MIMIC_DIR", mimic_dir)
    monkeypatch.setattr(mimic, "BHC_JSONL", bhc_jsonl)

    # Add ICUSTAYS entries for the BHC-only test admissions (500, 600, 700)
    # so they exist in ICUSTAYS for the eligible pool join.
    extra_csv = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,DBSOURCE,FIRST_CAREUNIT,"
        "LAST_CAREUNIT,FIRST_WARDID,LAST_WARDID,INTIME,OUTTIME,LOS\n"
        "1,10,100,1001,metavision,MICU,MICU,1,1,2150-01-01,2150-01-05,4.0\n"
        "2,10,100,1002,metavision,SICU,SICU,2,2,2150-01-05,2150-01-07,2.0\n"
        "3,20,200,2001,carevue,CCU,CCU,3,3,2145-03-01,2145-03-04,3.0\n"
        "4,30,300,3001,both,MICU,MICU,1,1,2148-06-01,2148-06-03,2.0\n"
        "5,40,400,4001,metavision,TSICU,TSICU,4,4,2151-02-01,2151-02-06,5.0\n"
        "6,50,500,5001,metavision,MICU,MICU,1,1,2151-03-01,2151-03-03,2.0\n"
        "7,60,600,6001,metavision,SICU,SICU,2,2,2151-04-01,2151-04-04,3.0\n"
        "8,70,700,7001,metavision,MICU,MICU,1,1,2151-05-01,2151-05-03,2.0\n"
    )
    icustays_path = mimic_dir / "ICUSTAYS.csv.gz"
    with gzip.open(icustays_path, "wt") as f:
        f.write(extra_csv)


def _write_cohort_fixtures(mimic_dir: Path, bhc_path: Path) -> None:
    """Write synthetic MIMIC files for final cohort tests.

    Creates five MetaVision admissions with different filter outcomes:
    - 1001: passes all filters (LOS=5, 3 charting days, 3 notes, 15 labs)
    - 1002: passes all filters (LOS=4, 4 charting days, 4 notes, 15 labs).
      One of the four notes is a pre-admission ED note on 2150-01-31 used
      to exercise compute_pre_admission_notes.
    - 1003: fails LOS < 3 (LOS=2)
    - 1004: fails charting days < 3 (only 2 charting days)
    - 1005: fails BHC > 15,000 chars
    """
    # ICUSTAYS
    icustays = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,DBSOURCE,FIRST_CAREUNIT,"
        "LAST_CAREUNIT,FIRST_WARDID,LAST_WARDID,INTIME,OUTTIME,LOS\n"
        "1,11,1001,9001,metavision,MICU,MICU,1,1,2150-01-01,2150-01-06,5.0\n"
        "2,12,1002,9002,metavision,SICU,SICU,2,2,2150-02-01,2150-02-05,4.0\n"
        "3,13,1003,9003,metavision,CCU,CCU,3,3,2150-03-01,2150-03-03,2.0\n"
        "4,14,1004,9004,metavision,TSICU,TSICU,4,4,2150-04-01,2150-04-05,4.0\n"
        "5,15,1005,9005,metavision,MICU,MICU,1,1,2150-05-01,2150-05-06,5.0\n"
    )
    with gzip.open(mimic_dir / "ICUSTAYS.csv.gz", "wt") as f:
        f.write(icustays)

    # ADMISSIONS (timestamps for LOS calculation)
    admissions = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ADMITTIME,DISCHTIME,DEATHTIME,"
        "ADMISSION_TYPE,ADMISSION_LOCATION,DISCHARGE_LOCATION,"
        "INSURANCE,LANGUAGE,RELIGION,MARITAL_STATUS,ETHNICITY,"
        "EDREGTIME,EDOUTTIME,DIAGNOSIS,HOSPITAL_EXPIRE_FLAG,HAS_CHARTEVENTS_DATA\n"
        "1,11,1001,2150-01-01 08:00:00,2150-01-06 14:00:00,,"
        "EMERGENCY,ER,HOME,Private,,,,WHITE,,,,0,1\n"
        "2,12,1002,2150-02-01 10:00:00,2150-02-05 12:00:00,,"
        "EMERGENCY,ER,HOME,Private,,,,WHITE,,,,0,1\n"
        "3,13,1003,2150-03-01 06:00:00,2150-03-03 10:00:00,,"
        "EMERGENCY,ER,HOME,Private,,,,WHITE,,,,0,1\n"
        "4,14,1004,2150-04-01 09:00:00,2150-04-05 16:00:00,,"
        "EMERGENCY,ER,HOME,Private,,,,WHITE,,,,0,1\n"
        "5,15,1005,2150-05-01 07:00:00,2150-05-06 11:00:00,,"
        "EMERGENCY,ER,HOME,Private,,,,WHITE,,,,0,1\n"
    )
    with gzip.open(mimic_dir / "ADMISSIONS.csv.gz", "wt") as f:
        f.write(admissions)

    # NOTEEVENTS: non-DS notes on distinct charting days.
    # 1001: 3 notes on 3 days (passes)
    # 1002: 4 notes on 4 days (one pre-admission, three in-admission)
    # 1003: 3 notes on 3 days (passes notes, but fails LOS)
    # 1004: 2 notes on 2 days (fails charting days)
    # 1005: 3 notes on 3 days (passes notes, but fails BHC)
    noteevents = (
        "ROW_ID,SUBJECT_ID,HADM_ID,CHARTDATE,CHARTTIME,STORETIME,"
        "CATEGORY,DESCRIPTION,CGID,ISERROR,TEXT\n"
        # 1001: notes on days 1, 3, 5
        '1,11,1001,2150-01-01,,,"Nursing","Note",1,,"Day 1 note"\n'
        '2,11,1001,2150-01-03,,,"Physician","Note",2,,"Day 3 note"\n'
        '3,11,1001,2150-01-05,,,"Nursing","Note",1,,"Day 5 note"\n'
        # 1002: pre-admission ED note on 2150-01-31, then in-admission on days 1, 3, 4.
        # The pre-admission note exercises the populated path for
        # compute_pre_admission_notes. Admission 1002 still passes the
        # note_count >= 2 and charting_days >= 3 filters.
        '15,12,1002,2150-01-31,,,"Nursing","Pre-admission ED note",1,,"ED note text"\n'
        '4,12,1002,2150-02-01,,,"Nursing","Note",1,,"Day 1 note"\n'
        '5,12,1002,2150-02-03,,,"Physician","Note",2,,"Day 3 note"\n'
        '6,12,1002,2150-02-04,,,"Nursing","Note",1,,"Day 4 note"\n'
        # 1003: notes on days 1, 2, 3
        '7,13,1003,2150-03-01,,,"Nursing","Note",1,,"Day 1 note"\n'
        '8,13,1003,2150-03-02,,,"Physician","Note",2,,"Day 2 note"\n'
        '9,13,1003,2150-03-03,,,"Nursing","Note",1,,"Day 3 note"\n'
        # 1004: notes on only 2 days
        '10,14,1004,2150-04-01,,,"Nursing","Note",1,,"Day 1 note"\n'
        '11,14,1004,2150-04-02,,,"Physician","Note",2,,"Day 2 note"\n'
        # 1005: notes on days 1, 3, 5
        '12,15,1005,2150-05-01,,,"Nursing","Note",1,,"Day 1 note"\n'
        '13,15,1005,2150-05-03,,,"Physician","Note",2,,"Day 3 note"\n'
        '14,15,1005,2150-05-05,,,"Nursing","Note",1,,"Day 5 note"\n'
    )
    with gzip.open(mimic_dir / "NOTEEVENTS.csv.gz", "wt") as f:
        f.write(noteevents)

    # LABEVENTS: enough labs for each admission to pass/fail.
    # All admissions get 15 labs (above the >= 10 safety net).
    lab_rows = ["ROW_ID,SUBJECT_ID,HADM_ID,ITEMID,CHARTTIME,VALUE,VALUENUM,VALUEUOM,FLAG\n"]
    row_id = 1
    for hadm_id, subject_id, date_prefix in [
        (1001, 11, "2150-01"),
        (1002, 12, "2150-02"),
        (1003, 13, "2150-03"),
        (1004, 14, "2150-04"),
        (1005, 15, "2150-05"),
    ]:
        for i in range(15):
            lab_rows.append(
                f"{row_id},{subject_id},{hadm_id},50811,"
                f"{date_prefix}-01 {8 + i}:00:00,12.0,12.0,g/dL,\n"
            )
            row_id += 1
    with gzip.open(mimic_dir / "LABEVENTS.csv.gz", "wt") as f:
        f.write("".join(lab_rows))

    # PRESCRIPTIONS: per-admission counts become prescription_count in
    # filtered_pool. Admission 1001 has 4 prescriptions. Admission 1002 has no
    # PRESCRIPTIONS row at all, which exercises the LEFT JOIN + COALESCE-to-0
    # path for an otherwise-passing admission.
    prescriptions = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,STARTDATE,ENDDATE,DRUG_TYPE,"
        "DRUG,DRUG_NAME_POE,DRUG_NAME_GENERIC,FORMULARY_DRUG_CD,GSN,NDC,"
        "PROD_STRENGTH,DOSE_VAL_RX,DOSE_UNIT_RX,FORM_VAL_DISP,FORM_UNIT_DISP,ROUTE\n"
        "1,11,1001,9001,2150-01-01,2150-01-02,MAIN,Aspirin,,,,,,81mg,81,mg,1,TAB,PO\n"
        "2,11,1001,9001,2150-01-02,2150-01-03,MAIN,Lisinopril,,,,,,10mg,10,mg,1,TAB,PO\n"
        "3,11,1001,9001,2150-01-03,2150-01-04,MAIN,Metoprolol,,,,,,25mg,25,mg,1,TAB,PO\n"
        "4,11,1001,9001,2150-01-04,2150-01-05,MAIN,Atorvastatin,,,,,,40mg,40,mg,1,TAB,PO\n"
    )
    with gzip.open(mimic_dir / "PRESCRIPTIONS.csv.gz", "wt") as f:
        f.write(prescriptions)

    # PATIENTS: DOB drives the age calculation (DATE_DIFF year, capped at 90).
    # Subjects line up 1:1 with the admissions above.
    # Subject 15: DOB shifted ~300 years back (HIPAA convention for >89), so
    # LEAST(DATE_DIFF, 90) clamps to 90.
    patients = (
        "ROW_ID,SUBJECT_ID,GENDER,DOB,DOD,DOD_HOSP,DOD_SSN,EXPIRE_FLAG\n"
        "1,11,M,2090-01-01 00:00:00,,,,0\n"  # age 60 at 2150
        "2,12,F,2075-01-01 00:00:00,,,,0\n"  # age 75 at 2150
        "3,13,M,2100-01-01 00:00:00,,,,0\n"  # age 50 at 2150
        "4,14,F,2120-01-01 00:00:00,,,,0\n"  # age 30 at 2150
        "5,15,M,1850-01-01 00:00:00,,,,0\n"  # censored DOB, capped to age 90
    )
    with gzip.open(mimic_dir / "PATIENTS.csv.gz", "wt") as f:
        f.write(patients)

    # SERVICES: discharging service per admission (last CURR_SERVICE before discharge).
    # Admission 1002 has a service transfer to exercise the ROW_NUMBER logic.
    services = (
        "ROW_ID,SUBJECT_ID,HADM_ID,TRANSFERTIME,PREV_SERVICE,CURR_SERVICE\n"
        "1,11,1001,2150-01-01 09:00:00,,MED\n"
        "2,12,1002,2150-02-01 11:00:00,,MED\n"
        "3,12,1002,2150-02-04 09:00:00,MED,CSURG\n"  # transfer (CSURG is the discharging service)
        "4,13,1003,2150-03-01 07:00:00,,MED\n"
        "5,14,1004,2150-04-01 10:00:00,,NSURG\n"
        # 1005 deliberately has no SERVICES row to exercise the LEFT JOIN -> NULL path.
    )
    with gzip.open(mimic_dir / "SERVICES.csv.gz", "wt") as f:
        f.write(services)

    # BHC JSONL
    # The first record has a non-empty quality_flags so DuckDB can infer the
    # array element type (VARCHAR). Without this, all-empty arrays cause a
    # type inference failure when list_contains() compares against 'iserror'.
    # Admission 9999 has no ICUSTAYS entry, so it's excluded by the JOIN.
    records = [
        {
            "hadm_id": 9999,
            "subject_id": 99,
            "bhc_char_count": 100,
            "bhc_word_count": 20,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": ["placeholder"],
            "bhc_structure": {
                "category": "narrative",
                "numbering_convention": None,
                "modifiers": [],
            },
        },
        {
            "hadm_id": 1001,
            "subject_id": 11,
            "bhc_char_count": 2500,
            "bhc_word_count": 400,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "narrative",
                "numbering_convention": None,
                "modifiers": [],
            },
        },
        {
            "hadm_id": 1002,
            "subject_id": 12,
            "bhc_char_count": 3000,
            "bhc_word_count": 500,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "domain_organized",
                "numbering_convention": "#",
                "modifiers": ["acute_chronic"],
            },
        },
        {
            "hadm_id": 1003,
            "subject_id": 13,
            "bhc_char_count": 1800,
            "bhc_word_count": 300,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "narrative",
                "numbering_convention": None,
                "modifiers": [],
            },
        },
        {
            "hadm_id": 1004,
            "subject_id": 14,
            "bhc_char_count": 2000,
            "bhc_word_count": 350,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "domain_organized",
                "numbering_convention": "N.",
                "modifiers": [],
            },
        },
        {
            "hadm_id": 1005,
            "subject_id": 15,
            "bhc_char_count": 16000,
            "bhc_word_count": 2800,
            "bhc_text": "Brief hospital course narrative for testing.",
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "domain_organized",
                "numbering_convention": "#",
                "modifiers": [],
            },
        },
    ]
    with bhc_path.open("w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)


@pytest.fixture
def _patch_cohort_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point mimic module at synthetic data for final cohort tests."""
    bhc_path = tmp_path / "bhc.jsonl"
    monkeypatch.setattr(mimic, "MIMIC_DIR", tmp_path)
    monkeypatch.setattr(mimic, "BHC_JSONL", bhc_path)
    _write_cohort_fixtures(tmp_path, bhc_path)


@pytest.mark.usefixtures("_patch_paths")
class TestCreateIcustayAdmissions:
    """Tests for create_icustay_admissions."""

    def test_deduplicates_multiple_stays(self) -> None:
        """Admission with two ICU stays should produce one row."""
        conn = duckdb.connect()
        mimic.create_icustay_admissions(conn)
        rows = conn.execute("SELECT * FROM icustay_admissions WHERE HADM_ID = 100").fetchall()
        assert len(rows) == 1

    def test_excludes_both_by_default(self) -> None:
        """'both' admissions are excluded even without metavision_only."""
        conn = duckdb.connect()
        mimic.create_icustay_admissions(conn)
        rows = conn.execute("SELECT * FROM icustay_admissions WHERE HADM_ID = 300").fetchall()
        assert len(rows) == 0

    def test_includes_carevue_and_metavision_by_default(self) -> None:
        """Default mode includes both CareVue and MetaVision."""
        conn = duckdb.connect()
        mimic.create_icustay_admissions(conn)
        sources = conn.execute(
            "SELECT DISTINCT DBSOURCE FROM icustay_admissions ORDER BY DBSOURCE"
        ).fetchall()
        assert [r[0] for r in sources] == ["carevue", "metavision"]

    def test_metavision_only_excludes_carevue(self) -> None:
        """metavision_only=True excludes CareVue admissions."""
        conn = duckdb.connect()
        mimic.create_icustay_admissions(conn, metavision_only=True)
        sources = conn.execute("SELECT DISTINCT DBSOURCE FROM icustay_admissions").fetchall()
        assert [r[0] for r in sources] == ["metavision"]
        # CareVue admission 200 should not appear.
        rows = conn.execute("SELECT * FROM icustay_admissions WHERE HADM_ID = 200").fetchall()
        assert len(rows) == 0


@pytest.mark.usefixtures("_patch_paths")
class TestCreateEligiblePool:
    """Tests for create_eligible_pool."""

    def test_returns_correct_count(self) -> None:
        """Only MetaVision admissions with valid BHC pass all filters."""
        conn = duckdb.connect()
        count = mimic.create_eligible_pool(conn)
        # 100: MetaVision + valid BHC -> eligible
        # 200: CareVue -> excluded by MetaVision filter
        # 300: 'both' -> excluded, no BHC anyway
        # 400: MetaVision + valid BHC -> eligible
        # 500: MetaVision but bhc_char_count < 100 -> excluded
        # 600: MetaVision but extraction_success=False -> excluded
        # 700: MetaVision but iserror flag -> excluded
        assert count == 2

    @pytest.mark.parametrize(
        ("hadm_id", "reason"),
        [
            (500, "BHC < 100 chars"),
            (600, "failed extraction"),
            (700, "BHC flagged as iserror"),
        ],
    )
    def test_excludes_admission(self, hadm_id: int, reason: str) -> None:
        """Each row is one rejection rule from the eligibility funnel."""
        conn = duckdb.connect()
        mimic.create_eligible_pool(conn)
        rows = conn.execute("SELECT * FROM eligible_pool WHERE hadm_id = ?", [hadm_id]).fetchall()
        assert rows == [], f"hadm_id {hadm_id} ({reason}) should be excluded"


@pytest.mark.usefixtures("_patch_cohort_paths")
class TestCreateFilteredPool:
    """Tests for create_filtered_pool."""

    def test_returns_correct_count(self) -> None:
        """Only admissions passing all filters are in the filtered pool."""
        conn = duckdb.connect()
        count = mimic.create_filtered_pool(conn)
        # 1001: LOS=5, 3 chart days, BHC=2500, 3 notes, 15 labs -> passes
        # 1002: LOS=4, 4 chart days, BHC=3000, 4 notes, 15 labs -> passes
        # 1003: LOS=2 -> fails LOS >= 3
        # 1004: 2 chart days -> fails charting days >= 3
        # 1005: BHC=16000 -> fails BHC <= 15000
        assert count == 2

    @pytest.mark.parametrize(
        ("hadm_id", "reason"),
        [
            (1003, "LOS < 3"),
            (1004, "< 3 charting days"),
            (1005, "BHC > 15,000 chars"),
        ],
    )
    def test_excludes_admission(self, hadm_id: int, reason: str) -> None:
        """Each row is one rejection rule the filtered pool applies."""
        conn = duckdb.connect()
        mimic.create_filtered_pool(conn)
        rows = conn.execute("SELECT * FROM filtered_pool WHERE hadm_id = ?", [hadm_id]).fetchall()
        assert rows == [], f"hadm_id {hadm_id} ({reason}) should be excluded"

    def test_enrichment_values(self) -> None:
        """Verify enrichment columns have correct values for a passing admission."""
        conn = duckdb.connect()
        mimic.create_filtered_pool(conn)
        row = conn.execute(
            "SELECT los_days, note_count, charting_days, lab_count, prescription_count, "
            "gender, age, service, hospital_expire_flag "
            "FROM filtered_pool WHERE hadm_id = 1001"
        ).fetchone()
        assert row[0] == 5  # LOS
        assert row[1] == 3  # notes
        assert row[2] == 3  # charting days
        assert row[3] == 15  # labs
        assert row[4] == 4  # prescriptions from the synthetic fixture
        assert row[5] == "M"  # gender from PATIENTS
        assert row[6] == 60  # age = 2150 - 2090
        assert row[7] == "MED"  # discharging service from SERVICES
        assert row[8] == 0  # HOSPITAL_EXPIRE_FLAG from ADMISSIONS

    def test_admission_without_prescriptions_gets_zero(self) -> None:
        """COALESCE on the LEFT JOIN gives prescription_count = 0 when none exist."""
        conn = duckdb.connect()
        mimic.create_filtered_pool(conn)
        # 1002 passes all filters but has no PRESCRIPTIONS row in the fixture.
        # The LEFT JOIN returns NULL for rx.prescription_count, and COALESCE clamps that to 0.
        row = conn.execute(
            "SELECT prescription_count FROM filtered_pool WHERE hadm_id = 1002"
        ).fetchone()
        assert row[0] == 0

    def test_discharging_service_is_last_transfer(self) -> None:
        """When an admission transfers services, the discharging service is the last one."""
        conn = duckdb.connect()
        mimic.create_filtered_pool(conn)
        # Admission 1002 has MED -> CSURG transfer, so CSURG is the discharging service.
        row = conn.execute("SELECT service FROM filtered_pool WHERE hadm_id = 1002").fetchone()
        assert row[0] == "CSURG"


@pytest.mark.usefixtures("_patch_cohort_paths")
class TestComputeTemporalMetrics:
    """compute_temporal_metrics on the synthetic cohort fixture (admission 1001).

    Fixture admission 1001:
    - admittime: 2150-01-01 08:00, dischtime: 2150-01-06 14:00
    - 3 usable notes on chartdate 2150-01-01, 2150-01-03, 2150-01-05
    - All three notes have NULL CHARTTIME (date-only)

    With no CHARTTIME, effective_time falls back to CHARTDATE at midnight.
    """

    def test_offsets_and_completeness(self) -> None:
        conn = duckdb.connect()
        result = compute_temporal_metrics(conn, [1001])
        td = result[1001]["temporal_density"]
        tc = result[1001]["timestamp_completeness"]

        assert td["first_note_offset_hours"] == -8.0
        assert td["last_note_offset_hours"] == -38.0
        assert td["clinical_span_hours"] == 96.0
        assert td["gap_hours_between_notes"] == {}
        assert td["notes_per_day"] == {"p50": 1.0, "p75": 1.0, "p90": 1.0, "max": 1}

        assert tc["null_charttime_note_count"] == 3
        assert tc["null_charttime_note_categories"] == {"Nursing": 2, "Physician": 1}


@pytest.mark.usefixtures("_patch_cohort_paths")
class TestComputeSourceDiversity:
    """compute_source_diversity counts distinct labs/drugs/services per admission.

    Fixture admissions exercised here:
    - 1001: 15 LABEVENTS rows all on ITEMID=50811 (1 distinct item),
      4 PRESCRIPTIONS rows on 4 distinct DRUG values,
      1 SERVICES row with CURR_SERVICE=MED.
    - 1002: 15 LABEVENTS rows all on ITEMID=50811 (1 distinct item),
      0 PRESCRIPTIONS rows (no entry at all),
      2 SERVICES rows: MED then CSURG. The distinct count is 2 because
      every CURR_SERVICE row counts, the discharging one included.
    - 1005: has labs but no PRESCRIPTIONS or SERVICES row at all, so both
      counts are 0.
    """

    def test_distinct_counts_across_three_admissions(self) -> None:
        conn = duckdb.connect()
        result = compute_source_diversity(conn, [1001, 1002, 1005])

        assert result[1001] == {
            "distinct_lab_items": 1,
            "distinct_drugs": 4,
            "distinct_services_touching": 1,
        }
        # 1002 has two CURR_SERVICE rows (MED, CSURG). The distinct count
        # captures both, including the non-discharging MED.
        assert result[1002] == {
            "distinct_lab_items": 1,
            "distinct_drugs": 0,
            "distinct_services_touching": 2,
        }
        # 1005 has no PRESCRIPTIONS or SERVICES row. Counts default to 0.
        assert result[1005] == {
            "distinct_lab_items": 1,
            "distinct_drugs": 0,
            "distinct_services_touching": 0,
        }


@pytest.mark.usefixtures("_patch_cohort_paths")
class TestComputePreAdmissionNotes:
    """compute_pre_admission_notes counts notes whose CHARTDATE precedes admittime.

    Fixture admissions exercised here:
    - 1001: admittime 2150-01-01. Notes on 2150-01-01, 03, 05 (none
      pre-admission). Empty path.
    - 1002: admittime 2150-02-01. One Nursing note on 2150-01-31 (pre)
      plus three on 02-01, 02-03, 02-04. Populated path.

    A fake tokenizer dict is passed in.
    """

    @staticmethod
    def _fake_tokenizers() -> dict:
        from mimic_data_prep.models import MODEL_KEYS

        class _FakeTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG002, FBT001, FBT002
                return [0] * len(text.split())

        return dict.fromkeys(MODEL_KEYS, _FakeTokenizer())

    def test_no_pre_admission_notes(self) -> None:
        conn = duckdb.connect()
        result = compute_pre_admission_notes(conn, [1001], self._fake_tokenizers(), n_workers=1)
        assert result[1001]["count"] == 0
        assert result[1001]["categories"] == {}
        assert all(v == 0 for v in result[1001]["token_counts"].values())

    def test_pre_admission_note_present(self) -> None:
        conn = duckdb.connect()
        result = compute_pre_admission_notes(conn, [1002], self._fake_tokenizers(), n_workers=1)
        # One pre-admission Nursing note. token_counts is per-model and
        # depends on each tokenizer, so asserting > 0 across all keys
        # is enough to confirm the path executed.
        assert result[1002]["count"] == 1
        assert result[1002]["categories"] == {"Nursing": 1}
        assert all(v > 0 for v in result[1002]["token_counts"].values())


class TestComputeNarrativeTokenCounts:
    """compute_narrative_token_counts per-model token totals for narrative notes.

    Narrative notes are usable notes whose CATEGORY is exactly 'Nursing'
    or starts with 'Physician'.
    """

    @staticmethod
    def _fake_tokenizers() -> dict:
        from mimic_data_prep.models import MODEL_KEYS

        class _FakeTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG002, FBT001, FBT002
                return [0] * len(text.split())

        return dict.fromkeys(MODEL_KEYS, _FakeTokenizer())

    @staticmethod
    def _write_mixed_notes(mimic_dir: Path) -> None:
        """One admission with mixed note categories.

        Admission 800 has one Nursing note ('alpha beta'), one Physician
        note ('gamma'), one Radiology note (excluded), and one ECG note
        (excluded). The narrative-token count under the fake tokenizer
        (whitespace word count) should be 3 across all three models.
        """
        rows = (
            "ROW_ID,SUBJECT_ID,HADM_ID,CHARTDATE,CHARTTIME,STORETIME,"
            "CATEGORY,DESCRIPTION,CGID,ISERROR,TEXT\n"
            '1,80,800,2150-01-01,,,"Nursing","Note",1,,"alpha beta"\n'
            '2,80,800,2150-01-02,,,"Physician","Note",1,,"gamma"\n'
            '3,80,800,2150-01-03,,,"Radiology","Note",1,,"radiology one two three"\n'
            '4,80,800,2150-01-04,,,"ECG","Note",1,,"ecg only"\n'
        )
        with gzip.open(mimic_dir / "NOTEEVENTS.csv.gz", "wt") as f:
            f.write(rows)

    def test_excludes_non_narrative_categories(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(mimic, "MIMIC_DIR", tmp_path)
        self._write_mixed_notes(tmp_path)
        conn = duckdb.connect()
        result = compute_narrative_token_counts(conn, [800], self._fake_tokenizers(), n_workers=1)
        for count in result[800].values():
            assert count == 3


class TestNoteOrder:
    """The SQL note order and the Python chronological sort key must agree.

    Two implementations of one order: SQL for the loader and the cohort
    aggregates, Python for the chronological prompt. If they disagree, the two
    orders name a different note as first.
    """

    # One admission day, deliberately adversarial. ROW_ID is assigned in a
    # scrambled order so it cannot stand in for time.
    _NOTES_CSV = (
        "ROW_ID,SUBJECT_ID,HADM_ID,CHARTDATE,CHARTTIME,CATEGORY,DESCRIPTION,ISERROR,TEXT\n"
        "5,10,100,2150-01-02,2150-01-02 14:30:00,Nursing,Note,,n1\n"
        "3,10,100,2150-01-02,2150-01-02 08:15:00,Physician,Note,,p1\n"
        "9,10,100,2150-01-02,,ECG,Note,,e1\n"  # untimed: sorts at noon
        "1,10,100,2150-01-02,2150-01-02 17:00:00,Radiology,Note,,r1\n"
        "7,10,100,2150-01-02,2150-01-02 08:15:00,Respiratory,Note,,s1\n"  # ties p1 on time
        "2,10,100,2150-01-01,2150-01-01 23:00:00,Radiology,Note,,r0\n"  # previous day
    )

    def _conn(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> duckdb.DuckDBPyConnection:
        with gzip.open(tmp_path / "NOTEEVENTS.csv.gz", "wt") as f:
            f.write(self._NOTES_CSV)
        monkeypatch.setattr(mimic, "MIMIC_DIR", tmp_path)
        return duckdb.connect()

    def test_sql_order_is_chronological_not_alphabetical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = self._conn(tmp_path, monkeypatch)
        rows = conn.execute(f"""
            SELECT TEXT FROM {mimic.table_ref("NOTEEVENTS.csv.gz")} n
            WHERE n.HADM_ID = 100 AND {mimic.noteevents_input_filter("n")}
            ORDER BY {mimic.note_order_sql("n")}
        """).fetchall()
        # Previous day first. Then, within 2150-01-02: the two 08:15 notes in
        # ROW_ID order (3 before 7), the untimed ECG at noon, then 14:30 and 17:00.
        assert [r[0] for r in rows] == ["r0", "p1", "s1", "e1", "n1", "r1"]

    def test_sql_order_matches_the_chronological_sort_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from bhc_experiment.prompt_builder import chronological_sort_key

        from .conftest import make_note

        conn = self._conn(tmp_path, monkeypatch)
        rows = conn.execute(f"""
            SELECT ROW_ID, CHARTDATE::VARCHAR, CHARTTIME::VARCHAR
            FROM {mimic.table_ref("NOTEEVENTS.csv.gz")} n
            WHERE n.HADM_ID = 100 AND {mimic.noteevents_input_filter("n")}
            ORDER BY {mimic.note_order_sql("n")}
        """).fetchall()
        notes = [make_note(r[0], CHARTDATE=r[1], CHARTTIME=r[2]) for r in rows]
        assert [n.ROW_ID for n in notes] == [
            n.ROW_ID for n in sorted(notes, key=chronological_sort_key)
        ]


class TestTableRef:
    def test_all_varchar_reaches_the_reader(self) -> None:
        path = mimic.MIMIC_DIR / "CPTEVENTS.csv.gz"
        ref = mimic.table_ref("CPTEVENTS.csv.gz", all_varchar=True)
        assert ref == f"read_csv_auto('{path}', all_varchar=true)"
