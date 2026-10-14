"""Which admissions a run covers, and the bundle each one feeds the model.

Generation and scoring both call ``run_inputs``, so the input scoring measures
the ceiling over is the input the model saw. A full-input run covers the whole
cohort with every prepared record. An oracle run covers the long-strata
admissions that have at least one anchor, each reduced to its anchors and their
neighbors.
"""

from collections.abc import Callable
from pathlib import Path

from bhc_experiment.concept_spaces import concept_params
from bhc_experiment.config import MAIN_CONCEPT_SPACE, ORACLE_STRATA, CliParameters
from bhc_experiment.loader import AdmissionBundle
from bhc_experiment.oracle import (
    AdmissionRecordConcepts,
    load_record_concepts,
    record_concepts_path,
    reduce_bundle,
    selected_ids_for_admission,
)
from bhc_experiment.prepared_store import PreparedStore, prepared_path
from bhc_experiment.prompt_builder import build_prompt
from bhc_experiment.results import iter_result_records
from mimic_data_prep.cohort import load_cohort
from mimic_data_prep.mimic import COHORTS_DIR, PREPARED_DIR


def oracle_selections(
    strata_by_hadm: dict[int, str],
    record_concepts: dict[int, AdmissionRecordConcepts],
    neighbor_expansion: int,
) -> dict[int, set[tuple[str, int]]]:
    """The record ids each long-stratum admission with at least one anchor feeds the model."""
    selections: dict[int, set[tuple[str, int]]] = {}
    for hadm_id, stratum in strata_by_hadm.items():
        if stratum not in ORACLE_STRATA:
            continue
        selected = selected_ids_for_admission(record_concepts[hadm_id], neighbor_expansion)
        # An admission with no anchors would feed the model an empty prompt.
        if selected:
            selections[hadm_id] = selected
    return selections


def _record_concepts_path(cli_parameters: CliParameters) -> Path:
    """The record concepts the oracle selects from, in the concept space scoring reads."""
    return record_concepts_path(
        PREPARED_DIR,
        cli_parameters.cohort_file,
        cli_parameters.deduplication,
        concept_params(MAIN_CONCEPT_SPACE),
    )


def describe_inputs(cli_parameters: CliParameters) -> dict:
    """The files a run reads and, for an oracle run, the strata it selects from."""
    inputs = {
        "cohort_file": cli_parameters.cohort_file,
        "prepared_file": prepared_path(
            PREPARED_DIR, cli_parameters.cohort_file, cli_parameters.deduplication
        ).name,
    }
    if cli_parameters.oracle_n is not None:
        inputs["oracle_strata"] = list(ORACLE_STRATA)
        inputs["record_concepts_file"] = _record_concepts_path(cli_parameters).name
    return inputs


def mismatched_prompts(
    model_dir: Path, bundles: dict[int, AdmissionBundle], variant: str
) -> list[int]:
    """The generated admissions whose stored prompt differs from the one ``bundles`` builds.

    The stored prompt is what the model saw. An admission missing from
    ``bundles`` counts as a mismatch. A failed admission is not scored, so it is
    not compared.
    """
    mismatched = []
    for record in iter_result_records(model_dir):
        if record.error is not None:
            continue
        bundle = bundles.get(record.hadm_id)
        if bundle is None:
            mismatched.append(record.hadm_id)
            continue
        prompt = build_prompt(bundle, variant)
        if record.prompt != {"system": prompt.system, "user": prompt.user}:
            mismatched.append(record.hadm_id)
    return mismatched


def run_inputs(cli_parameters: CliParameters) -> tuple[list[int], Callable[[int], AdmissionBundle]]:
    """The run's admissions, in cohort order, and a loader for each one's bundle."""
    cohort = load_cohort(COHORTS_DIR / cli_parameters.cohort_file)
    store = PreparedStore(PREPARED_DIR, cli_parameters.cohort_file, cli_parameters.deduplication)
    if cli_parameters.oracle_n is None:
        return [a.hadm_id for a in cohort.admissions], store.bundle

    selections = oracle_selections(
        {a.hadm_id: a.token_stratum for a in cohort.admissions},
        load_record_concepts(_record_concepts_path(cli_parameters)),
        cli_parameters.oracle_n,
    )

    def load_bundle(hadm_id: int) -> AdmissionBundle:
        return reduce_bundle(store.bundle(hadm_id), selections[hadm_id])

    return [a.hadm_id for a in cohort.admissions if a.hadm_id in selections], load_bundle
