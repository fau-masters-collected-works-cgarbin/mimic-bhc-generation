"""Tests for the experiment data loader (bhc_experiment.loader).

Uses synthetic CSV.gz and JSONL fixtures.
"""

import gzip
import json
from pathlib import Path

import duckdb
import pytest

from bhc_experiment.loader import (
    LabEvent,
    Prescription,
    load_admission_bundle,
    load_reference_texts,
    prepare_cohort,
)
from mimic_data_prep import mimic


def _write_loader_fixtures(mimic_dir: Path, bhc_path: Path) -> None:
    """Write synthetic MIMIC files for loader tests.

    Creates four admissions with different characteristics:
    - 1001: normal admission with notes, labs, prescriptions. Has one error
      note, one pre-admission note, and one note with NULL CHARTDATE (all
      should appear with flags set). Patient DOB is normal (age < 90).
    - 1002: admission with no labs and no prescriptions. Tests empty lists.
    - 1003: admission with multiple notes on different dates and labs at
      different times. Tests chronological ordering. Patient DOB is shifted
      ~300 years back (age should cap at 90).
    - 1004: two notes that share a CHARTDATE and CATEGORY, written to the CSV
      in reverse ROW_ID order. Tests that the loader breaks the tie by ROW_ID
      so the note order is deterministic and reproducible across runs.
    """
    # ICUSTAYS (all MetaVision)
    icustays = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,DBSOURCE,FIRST_CAREUNIT,"
        "LAST_CAREUNIT,FIRST_WARDID,LAST_WARDID,INTIME,OUTTIME,LOS\n"
        "1,11,1001,9001,metavision,MICU,MICU,1,1,2150-01-02,2150-01-08,6.0\n"
        "2,12,1002,9002,metavision,SICU,SICU,2,2,2150-02-01,2150-02-05,4.0\n"
        "3,13,1003,9003,metavision,CCU,CCU,3,3,2150-03-01,2150-03-07,6.0\n"
    )
    with gzip.open(mimic_dir / "ICUSTAYS.csv.gz", "wt") as f:
        f.write(icustays)

    # PATIENTS
    patients = (
        "ROW_ID,SUBJECT_ID,GENDER,DOB,DOD,DOD_HOSP,DOD_SSN,EXPIRE_FLAG\n"
        # Normal age: DOB 2100, admitted 2150 -> age 50
        "1,11,M,2100-01-01 00:00:00,,,,0\n"
        # Normal age: DOB 2110, admitted 2150 -> age 40
        "2,12,F,2110-01-01 00:00:00,,,,0\n"
        # Shifted DOB: DOB 1850, admitted 2150 -> raw age 300, capped to 90
        "3,13,M,1850-01-01 00:00:00,,,,0\n"
        # Normal age: DOB 2105, admitted 2150 -> age 45
        "4,14,F,2105-01-01 00:00:00,,,,0\n"
    )
    with gzip.open(mimic_dir / "PATIENTS.csv.gz", "wt") as f:
        f.write(patients)

    # ADMISSIONS
    admissions = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ADMITTIME,DISCHTIME,DEATHTIME,"
        "ADMISSION_TYPE,ADMISSION_LOCATION,DISCHARGE_LOCATION,"
        "INSURANCE,LANGUAGE,RELIGION,MARITAL_STATUS,ETHNICITY,"
        "EDREGTIME,EDOUTTIME,DIAGNOSIS,HOSPITAL_EXPIRE_FLAG,HAS_CHARTEVENTS_DATA\n"
        # 1001: admitted Jan 2, discharged Jan 8 -> LOS = 6 days
        "1,11,1001,2150-01-02 08:00:00,2150-01-08 14:00:00,,"
        "EMERGENCY,ER,HOME,Private,,,,WHITE,,,,0,1\n"
        # 1002: admitted Feb 1, discharged Feb 5 -> LOS = 4 days
        "2,12,1002,2150-02-01 10:00:00,2150-02-05 12:00:00,,"
        "EMERGENCY,ER,HOME,Private,,,,WHITE,,,,0,1\n"
        # 1003: admitted Mar 1, discharged Mar 7 -> LOS = 6 days
        "3,13,1003,2150-03-01 06:00:00,2150-03-07 10:00:00,,"
        "EMERGENCY,ER,HOME,Private,,,,WHITE,,,,0,1\n"
        # 1004: admitted Apr 1, discharged Apr 4 -> LOS = 3 days
        "4,14,1004,2150-04-01 08:00:00,2150-04-04 14:00:00,,"
        "EMERGENCY,ER,HOME,Private,,,,WHITE,,,,0,1\n"
    )
    with gzip.open(mimic_dir / "ADMISSIONS.csv.gz", "wt") as f:
        f.write(admissions)

    # NOTEEVENTS
    # 1001: 5 notes total
    #   - row 1: normal nursing note on Jan 3 (within admission)
    #   - row 2: error note on Jan 4 (ISERROR='1', within admission)
    #   - row 3: pre-admission note on Jan 1 (before ADMITTIME Jan 2)
    #   - row 4: discharge summary on Jan 8
    #   - row 9: note with NULL CHARTDATE (should sort last via NULLS LAST)
    # 1002: 1 note
    #   - row 5: nursing note on Feb 2
    # 1003: 3 notes on different dates (ordering test)
    #   - row 6: note on Mar 5 (should sort third)
    #   - row 7: note on Mar 1 (should sort first)
    #   - row 8: note on Mar 3 (should sort second)
    # 1004: 2 notes sharing CHARTDATE Apr 2 and CATEGORY Physician, written in
    #   reverse ROW_ID order (row 11 before row 10) so the tiebreaker is testable
    noteevents = (
        "ROW_ID,SUBJECT_ID,HADM_ID,CHARTDATE,CHARTTIME,STORETIME,"
        "CATEGORY,DESCRIPTION,CGID,ISERROR,TEXT\n"
        '1,11,1001,2150-01-03,2150-01-03 09:00:00,,"Nursing","Note",1,,'
        '"Day 2 nursing note for patient 1001."\n'
        '2,11,1001,2150-01-04,,,"Physician","Note",2,1,"This note was entered in error."\n'
        '3,11,1001,2150-01-01,,,"Nursing","Note",1,,"Pre-admission note before Jan 2."\n'
        '4,11,1001,2150-01-08,,,"Discharge summary","Report",3,,"Brief Hospital Course: etc."\n'
        '9,11,1001,,,,"Nursing","Note",1,,"Note with no chart date."\n'
        '5,12,1002,2150-02-02,,,"Nursing","Note",1,,"Only note for patient 1002."\n'
        '6,13,1003,2150-03-05,,,"Physician","Note",2,,"Third chronologically."\n'
        '7,13,1003,2150-03-01,,,"Nursing","Note",1,,"First chronologically."\n'
        '8,13,1003,2150-03-03,,,"Physician","Note",2,,"Second chronologically."\n'
        '11,14,1004,2150-04-02,,,"Physician","Note",2,,"Tie note, second by ROW_ID."\n'
        '10,14,1004,2150-04-02,,,"Physician","Note",2,,"Tie note, first by ROW_ID."\n'
    )
    with gzip.open(mimic_dir / "NOTEEVENTS.csv.gz", "wt") as f:
        f.write(noteevents)

    # D_LABITEMS: maps ITEMID to human-readable lab test name.
    d_labitems = (
        "ROW_ID,ITEMID,LABEL,FLUID,CATEGORY,LOINC_CODE\n"
        "1,50811,Hemoglobin,Blood,Hematology,718-7\n"
        "2,50912,Creatinine,Blood,Chemistry,2160-0\n"
    )
    with gzip.open(mimic_dir / "D_LABITEMS.csv.gz", "wt") as f:
        f.write(d_labitems)

    # LABEVENTS
    # 1001: 2 labs (one abnormal, one normal)
    # 1002: no labs (empty list test)
    # 1003: 3 labs at different times (ordering test, inserted out of order)
    # 1004: 2 labs sharing CHARTTIME Apr 2, written in reverse ROW_ID order
    #   (row 11 before row 10) so the ROW_ID tiebreaker is testable.
    labevents = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ITEMID,CHARTTIME,VALUE,VALUENUM,VALUEUOM,FLAG\n"
        "1,11,1001,50811,2150-01-03 08:00:00,12.0,12.0,g/dL,\n"
        "2,11,1001,50912,2150-01-04 10:00:00,1.2,1.2,mg/dL,abnormal\n"
        "3,13,1003,50811,2150-03-05 14:00:00,11.0,11.0,g/dL,\n"
        "4,13,1003,50811,2150-03-01 08:00:00,13.0,13.0,g/dL,\n"
        "5,13,1003,50912,2150-03-03 10:00:00,0.9,0.9,mg/dL,abnormal\n"
        "11,14,1004,50811,2150-04-02 09:00:00,10.0,10.0,g/dL,abnormal\n"
        "10,14,1004,50912,2150-04-02 09:00:00,1.0,1.0,mg/dL,abnormal\n"
    )
    with gzip.open(mimic_dir / "LABEVENTS.csv.gz", "wt") as f:
        f.write(labevents)

    # PRESCRIPTIONS
    # 1001: 1 prescription
    # 1002: no prescriptions (empty list test)
    # 1003: 2 prescriptions (ordering test, inserted out of order)
    # 1004: 2 prescriptions sharing STARTDATE Apr 2, written in reverse ROW_ID
    #   order (row 11 before row 10) so the ROW_ID tiebreaker is testable.
    prescriptions = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,STARTDATE,ENDDATE,"
        "DRUG_TYPE,DRUG,DRUG_NAME_POE,DRUG_NAME_GENERIC,"
        "FORMULARY_DRUG_CD,GSN,NDC,PROD_STRENGTH,"
        "DOSE_VAL_RX,DOSE_UNIT_RX,FORM_VAL_DISP,FORM_UNIT_DISP,ROUTE\n"
        # STARTDATE/ENDDATE have a midnight time component, matching real
        # MIMIC-III (e.g. "2150-01-03 00:00:00") down to the time component.
        "1,11,1001,9001,2150-01-03 00:00:00,2150-01-07 00:00:00,"
        "MAIN,Amoxicillin,Amoxicillin,Amoxicillin,"
        "AMOX500,,,500mg,"
        "500,mg,1,TAB,PO\n"
        "2,13,1003,9003,2150-03-04 00:00:00,2150-03-06 00:00:00,"
        "MAIN,Metoprolol,Metoprolol,Metoprolol Tartrate,"
        "METO25,,,25mg,"
        "25,mg,1,TAB,PO\n"
        "3,13,1003,9003,2150-03-01 00:00:00,2150-03-05 00:00:00,"
        "MAIN,Lisinopril,Lisinopril,Lisinopril,"
        "LISI10,,,10mg,"
        "10,mg,1,TAB,PO\n"
        "11,14,1004,9004,2150-04-02 00:00:00,2150-04-03 00:00:00,"
        "MAIN,Aspirin,Aspirin,Aspirin,"
        "ASA81,,,81mg,"
        "81,mg,1,TAB,PO\n"
        "10,14,1004,9004,2150-04-02 00:00:00,2150-04-03 00:00:00,"
        "MAIN,Warfarin,Warfarin,Warfarin,"
        "WARF5,,,5mg,"
        "5,mg,1,TAB,PO\n"
    )
    with gzip.open(mimic_dir / "PRESCRIPTIONS.csv.gz", "wt") as f:
        f.write(prescriptions)

    # BHC JSONL
    records = [
        {
            "hadm_id": 1001,
            "subject_id": 11,
            "bhc_text": "Patient was admitted for pneumonia and treated with antibiotics.",
            "bhc_char_count": 62,
            "bhc_word_count": 9,
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "narrative",
                "numbering_convention": None,
                "modifiers": [],
                "nicu_domain_count": 0,
                "labeled_section_count": 0,
            },
        },
        {
            "hadm_id": 1002,
            "subject_id": 12,
            "bhc_text": "Brief stay for observation after chest pain workup.",
            "bhc_char_count": 51,
            "bhc_word_count": 8,
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "domain_organized",
                "numbering_convention": "#",
                "modifiers": ["acute_chronic"],
                "nicu_domain_count": 0,
                "labeled_section_count": 3,
            },
        },
        {
            "hadm_id": 1003,
            "subject_id": 13,
            "bhc_text": "Patient admitted with hypertensive emergency requiring IV medications.",
            "bhc_char_count": 69,
            "bhc_word_count": 8,
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "domain_organized",
                "numbering_convention": "N.",
                "modifiers": [],
                "nicu_domain_count": 0,
                "labeled_section_count": 5,
            },
        },
        {
            "hadm_id": 1004,
            "subject_id": 14,
            "bhc_text": "Patient admitted for observation, discharged in stable condition.",
            "bhc_char_count": 65,
            "bhc_word_count": 9,
            "extraction_success": True,
            "quality_flags": [],
            "bhc_structure": {
                "category": "narrative",
                "numbering_convention": None,
                "modifiers": [],
                "nicu_domain_count": 0,
                "labeled_section_count": 0,
            },
        },
    ]
    with bhc_path.open("w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)


@pytest.fixture
def _patch_loader_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point mimic module at synthetic data for loader tests."""
    bhc_path = tmp_path / "bhc.jsonl"
    monkeypatch.setattr(mimic, "MIMIC_DIR", tmp_path)
    monkeypatch.setattr(mimic, "BHC_JSONL", bhc_path)
    _write_loader_fixtures(tmp_path, bhc_path)


@pytest.mark.usefixtures("_patch_loader_paths")
class TestLoadReferenceTexts:
    """Tests for load_reference_texts (bulk reference-BHC loading by hadm_id)."""

    def test_maps_each_hadm_id_to_its_reference_bhc(self) -> None:
        texts = load_reference_texts([1001, 1003])
        assert texts == {
            1001: "Patient was admitted for pneumonia and treated with antibiotics.",
            1003: "Patient admitted with hypertensive emergency requiring IV medications.",
        }


@pytest.mark.usefixtures("_patch_loader_paths")
class TestPrepareCohort:
    """Tests for prepare_cohort."""

    def test_filters_to_requested_hadm_ids(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001, 1003])
        hadm_ids = conn.execute(
            "SELECT DISTINCT HADM_ID FROM cohort_admissions ORDER BY HADM_ID"
        ).fetchall()
        assert [r[0] for r in hadm_ids] == [1001, 1003]


@pytest.mark.usefixtures("_patch_loader_paths")
class TestLoadBundle:
    """Tests for load_admission_bundle."""

    def test_admission_metadata(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        bundle = load_admission_bundle(conn, 1001)
        assert bundle.admission.HADM_ID == 1001
        assert bundle.admission.SUBJECT_ID == 11
        assert bundle.admission.GENDER == "M"
        assert bundle.admission.age == 50

    def test_ground_truth_bhc(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        bundle = load_admission_bundle(conn, 1001)
        assert "pneumonia" in bundle.ground_truth_bhc
        assert bundle.bhc_category == "narrative"

    def test_noteevents_flags_and_inclusion(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        bundle = load_admission_bundle(conn, 1001)

        # All 5 notes included (normal, error, pre-admission, DS, NULL date)
        assert len(bundle.noteevents) == 5

        # Error note: ISERROR arrives as the string '1'
        error_notes = [n for n in bundle.noteevents if n.ISERROR is not None]
        assert len(error_notes) == 1
        assert error_notes[0].ISERROR == "1"
        assert isinstance(error_notes[0].ISERROR, str)
        assert "entered in error" in error_notes[0].TEXT

        # Pre-admission note: CHARTDATE before ADMITTIME
        pre_adm = [n for n in bundle.noteevents if n.is_pre_admission]
        assert len(pre_adm) == 1
        assert pre_adm[0].CHARTDATE == "2150-01-01"

        # Discharge summary
        ds_notes = [n for n in bundle.noteevents if n.is_discharge_summary]
        assert len(ds_notes) == 1
        assert ds_notes[0].CATEGORY == "Discharge summary"

        # NULL CHARTDATE: present and sorts last
        null_date_notes = [n for n in bundle.noteevents if n.CHARTDATE is None]
        assert len(null_date_notes) == 1
        assert null_date_notes[0].TEXT == "Note with no chart date."
        assert bundle.noteevents[-1].CHARTDATE is None

    def test_noteevents_chronological_order(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1003])
        bundle = load_admission_bundle(conn, 1003)
        dates = [n.CHARTDATE for n in bundle.noteevents]
        assert dates == sorted(dates)

    def test_noteevents_tiebreak_by_row_id(self) -> None:
        """Notes sharing CHARTDATE and CATEGORY come back in ascending ROW_ID order.

        Without a ROW_ID tiebreaker the per-admission query orders only by
        CHARTDATE and CATEGORY, so same-date, same-category notes keep whatever
        physical order the scan produced (DuckDB's sort is stable). On the real
        multi-million-row NOTEEVENTS that scan order is not reproducible, so the
        note order could differ from run to run. ROW_ID pins it. To reproduce a
        non-ROW_ID scan order without a flaky scan, the test creates the
        note temp table in reverse ROW_ID order, then asserts the loader still
        returns the two Apr 2 notes ROW_ID-first.
        """
        conn = duckdb.connect()
        prepare_cohort(conn, [1004])
        # Force the tied notes into the opposite of ROW_ID order.
        conn.execute(
            "CREATE OR REPLACE TEMP TABLE cohort_noteevents AS "
            "SELECT * FROM cohort_noteevents ORDER BY ROW_ID DESC"
        )
        bundle = load_admission_bundle(conn, 1004)
        tied = [n for n in bundle.noteevents if n.CHARTDATE == "2150-04-02"]
        assert [n.ROW_ID for n in tied] == [10, 11]

    def test_noteevents_carry_charttime_when_populated(self) -> None:
        """Notes keep CHARTTIME when MIMIC populates it."""
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        bundle = load_admission_bundle(conn, 1001)
        timed = [n for n in bundle.noteevents if n.CHARTTIME is not None]
        assert timed, "expected at least one note with non-null CHARTTIME in the fixture"
        assert timed[0].CHARTTIME == "2150-01-03 09:00:00"

    def test_noteevents_carry_description(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        bundle = load_admission_bundle(conn, 1001)
        ds = [n for n in bundle.noteevents if n.is_discharge_summary]
        assert ds[0].DESCRIPTION == "Report"
        others = [n for n in bundle.noteevents if not n.is_discharge_summary]
        assert all(n.DESCRIPTION == "Note" for n in others)

    def test_labevents_values(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        bundle = load_admission_bundle(conn, 1001)
        assert len(bundle.labevents) == 2
        assert all(isinstance(lab, LabEvent) for lab in bundle.labevents)
        assert {lab.ROW_ID for lab in bundle.labevents} == {1, 2}
        abnormal = [lab for lab in bundle.labevents if lab.FLAG == "abnormal"]
        assert len(abnormal) == 1

    def test_labevents_have_label(self) -> None:
        """Lab events include human-readable test name from D_LABITEMS."""
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        bundle = load_admission_bundle(conn, 1001)
        labels = {lab.LABEL for lab in bundle.labevents}
        assert labels == {"Hemoglobin", "Creatinine"}

    def test_labevents_chronological_order(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1003])
        bundle = load_admission_bundle(conn, 1003)
        times = [lab.CHARTTIME for lab in bundle.labevents]
        assert times == sorted(times)

    def test_labevents_tiebreak_by_row_id(self) -> None:
        """Labs sharing a CHARTTIME come back in ascending ROW_ID order.

        Without a ROW_ID tiebreaker the per-admission query orders only by
        CHARTTIME, so same-time labs keep whatever physical order the scan
        produced (DuckDB's sort is stable). On the real multi-million-row
        LABEVENTS that scan order is not reproducible, so the lab order could
        differ from run to run. ROW_ID pins it. To reproduce a non-ROW_ID scan
        order without a flaky scan, the test rebuilds the lab temp table in
        reverse ROW_ID order, then asserts the loader still returns the two
        Apr 2 labs ROW_ID-first.
        """
        conn = duckdb.connect()
        prepare_cohort(conn, [1004])
        conn.execute(
            "CREATE OR REPLACE TEMP TABLE cohort_labevents AS "
            "SELECT * FROM cohort_labevents ORDER BY ROW_ID DESC"
        )
        bundle = load_admission_bundle(conn, 1004)
        tied = [lab for lab in bundle.labevents if lab.CHARTTIME == "2150-04-02 09:00:00"]
        assert [lab.ROW_ID for lab in tied] == [10, 11]

    def test_prescriptions_values(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        bundle = load_admission_bundle(conn, 1001)
        assert len(bundle.prescriptions) == 1
        assert isinstance(bundle.prescriptions[0], Prescription)
        assert bundle.prescriptions[0].DRUG == "Amoxicillin"

    def test_prescriptions_dates_are_date_only(self) -> None:
        """STARTDATE and ENDDATE load as date-only strings.

        MIMIC-III stores them as midnight datetimes ("2150-01-03 00:00:00").
        The loader normalizes both to "2150-01-03".
        """
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        bundle = load_admission_bundle(conn, 1001)
        rx = bundle.prescriptions[0]
        assert rx.STARTDATE == "2150-01-03"
        assert rx.ENDDATE == "2150-01-07"

    def test_prescriptions_chronological_order(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1003])
        bundle = load_admission_bundle(conn, 1003)
        dates = [p.STARTDATE for p in bundle.prescriptions]
        assert dates == sorted(dates)

    def test_prescriptions_tiebreak_by_row_id(self) -> None:
        """Prescriptions sharing a STARTDATE come back in ascending ROW_ID order.

        Same rationale as the labs tiebreaker: ordering by STARTDATE alone
        leaves same-day prescriptions in a non-reproducible scan order on the
        real PRESCRIPTIONS table. ROW_ID pins it. The test rebuilds the temp
        table in reverse ROW_ID order, then asserts the loader still returns the
        two Apr 2 prescriptions ROW_ID-first.
        """
        conn = duckdb.connect()
        prepare_cohort(conn, [1004])
        conn.execute(
            "CREATE OR REPLACE TEMP TABLE cohort_prescriptions AS "
            "SELECT * FROM cohort_prescriptions ORDER BY ROW_ID DESC"
        )
        bundle = load_admission_bundle(conn, 1004)
        tied = [p for p in bundle.prescriptions if p.STARTDATE == "2150-04-02"]
        assert [p.ROW_ID for p in tied] == [10, 11]

    def test_age_capped_at_90(self) -> None:
        """Patient with shifted DOB has age capped at 90."""
        conn = duckdb.connect()
        prepare_cohort(conn, [1003])
        bundle = load_admission_bundle(conn, 1003)
        assert bundle.admission.age == 90

    def test_multiple_admissions(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001, 1002, 1003])
        bundles = [load_admission_bundle(conn, hid) for hid in [1001, 1002, 1003]]
        assert len(bundles) == 3
        assert bundles[0].admission.HADM_ID == 1001
        assert bundles[1].admission.HADM_ID == 1002
        assert bundles[2].admission.HADM_ID == 1003
        # 1002 has no labs/prescriptions, 1001 and 1003 do
        assert bundles[1].labevents == []
        assert len(bundles[0].labevents) == 2
        assert len(bundles[2].labevents) == 3

    def test_bhc_structure_narrative(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        bundle = load_admission_bundle(conn, 1001)
        assert bundle.bhc_structure.category == "narrative"
        assert bundle.bhc_structure.numbering_convention is None
        assert bundle.bhc_structure.modifiers == []
        # bhc_category still works for backward compatibility
        assert bundle.bhc_category == "narrative"

    def test_bhc_structure_domain_organized_with_modifiers(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1002])
        bundle = load_admission_bundle(conn, 1002)
        assert bundle.bhc_structure.category == "domain_organized"
        assert bundle.bhc_structure.numbering_convention == "#"
        assert bundle.bhc_structure.modifiers == ["acute_chronic"]

    def test_bhc_structure_domain_organized_numbered(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1003])
        bundle = load_admission_bundle(conn, 1003)
        assert bundle.bhc_structure.category == "domain_organized"
        assert bundle.bhc_structure.numbering_convention == "N."
        assert bundle.bhc_structure.modifiers == []

    def test_duplicate_admission_raises(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        # Inject a duplicate row to simulate a prepare_cohort bug.
        conn.execute(
            "INSERT INTO cohort_admissions SELECT * FROM cohort_admissions WHERE HADM_ID = 1001"
        )
        with pytest.raises(ValueError, match="expected 1 row, got 2"):
            load_admission_bundle(conn, 1001)

    def test_duplicate_bhc_raises(self) -> None:
        conn = duckdb.connect()
        prepare_cohort(conn, [1001])
        conn.execute("INSERT INTO cohort_bhc SELECT * FROM cohort_bhc WHERE hadm_id = 1001")
        with pytest.raises(ValueError, match="expected 1 row, got 2"):
            load_admission_bundle(conn, 1001)
