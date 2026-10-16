"""Run parameters.

Everything every run shares is a constant here. What varies between runs is a
``CliParameters`` built from the command line. The git commit recorded with each
run pins the constants.
"""

import dataclasses
from dataclasses import dataclass

from bhc_experiment.llm import InferenceConfig
from mimic_data_prep.models import model_by_key
from mimic_data_prep.sampling import TOKEN_STRATA

# The MEDCON-style concept space every run is scored in. The TUI filter is the
# 66 semantic types of Van Veen et al. 2024 (StanfordMIMI/clin-summ,
# src/UMLSScorer.py:SEMANTICS), which inherits them from Yim et al. 2023, so
# ``tuis`` is left unset to take umls_extractor.MEDCON_DEFAULT_TUIS. The
# extractor differs: scispaCy here (Neumann et al. 2019), QuickUMLS in both
# references, so the scores are not directly comparable to QuickUMLS MEDCON
# numbers. Requires a pre-downloaded scispaCy model and the UMLS linker bundle.
SCISPACY_CLINICAL = {
    "name": "scispacy-clinical",
    "extractor": "scispacy",
    "spacy_model": "en_core_sci_md",
    "linker_name": "umls",
    "threshold": 0.7,  # the scispaCy EntityLinker default
}

# Scoring, the input-coverage ceiling, and the oracle selection all read this
# space's concept artifact.
MAIN_CONCEPT_SPACE = "scispacy"

# The metrics every run is scored with.
METRICS: tuple[dict, ...] = (
    {"rouge": {"variant": "rouge1", "name": "rouge1"}},
    {"rouge": {"variant": "rouge2", "name": "rouge2"}},
    {"rouge": {"variant": "rougeL", "name": "rougeL"}},
    # num_layers 4 matches the Discharge Me! scorer (Stanford-AIMI/discharge-me,
    # scoring/bertscore.py). bert_score defaults this model to layer 5. The rescale
    # baseline is indexed by layer (0.6014 at 4, 0.6662 at 5), so layer 5 moves the
    # rescaled score off the shared task's scale.
    {
        "bertscore": {
            "model_type": "distilbert-base-uncased",
            "num_layers": 4,
            "name": "distilbert",
        }
    },
    # Rescaled distilbert, the form the Discharge Me! BioNLP 2024 shared task
    # reports. Raw and rescaled scores are on different scales, so both are kept.
    {
        "bertscore": {
            "model_type": "distilbert-base-uncased",
            "num_layers": 4,
            "rescale_with_baseline": True,
            "name": "distilbert_rescaled",
        }
    },
    {"bertscore": {"model_type": "microsoft/deberta-xlarge-mnli", "name": "deberta"}},
    # 12 is the last layer of the PubMedBERT base, chosen for the most
    # domain-specific representation. bert_score ships no default layer or rescale
    # baseline for SapBERT.
    {
        "bertscore": {
            "model_type": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            "num_layers": 12,
            "name": "sapbert",
        }
    },
    {"umls_concept": {"space": MAIN_CONCEPT_SPACE}},
    # MEDCON as the reference implementation computes it: Yim et al. 2023's
    # umls_score_individual, unmodified, over its own QuickUMLS matcher. It puts
    # one figure in the units the published MEDCON values use, and returns F1
    # alone. Requires the QuickUMLS setup in docs/quickumls-setup.md.
    {"reference_medcon": {"name": "reference-medcon"}},
)

# A concept space is what counts as a clinical concept: an extractor, its
# settings, and the UMLS semantic types kept. The same text yields different
# concept sets in different spaces, so concept scores, the input-coverage
# ceiling, and the oracle anchors compare only within one space. Each space's
# settings fingerprint its concept artifact, so changing them points at a
# different file.
#
# The concept spaces preparation can extract in, by the name --concept-space
# takes. Only the main one scores a run. The others answer robustness questions.
CONCEPT_SPACES: dict[str, dict] = {
    "scispacy": SCISPACY_CLINICAL,
    # The main space plus the seven Procedures types of NLM's SemGroups file, 73
    # TUIs instead of 66. Not a metric to report: MEDCON is defined by seven
    # semantic groups and this is eight, so a score here is comparable to nothing
    # in the literature. It tests whether the coverage framework's reading
    # survives a wider space. Procedures
    # makes lab-test names count as concepts on the input side, so the ceiling,
    # observed recall, and input utilization move together under it.
    "scispacy-plus-procedures": {
        **SCISPACY_CLINICAL,
        "name": "scispacy-plus-procedures",
        "extra_tuis": [
            "T060",  # Diagnostic Procedure
            "T061",  # Therapeutic or Preventive Procedure
            "T058",  # Health Care Activity
            "T059",  # Laboratory Procedure
            "T062",  # Research Activity
            "T063",  # Molecular Biology Research Technique
            "T065",  # Educational Activity
        ],
    },
    # QuickUMLS, the extractor of the MEDCON reference implementation, under this
    # work's set aggregation. Not a metric to report: a score here matches neither
    # the reported numbers nor the published ones. Every QuickUMLS setting comes
    # from Yim's code (docs/quickumls-setup.md), so only the extractor is set.
    "quickumls": {"name": "quickumls-clinical", "extractor": "quickumls"},
}

EXPERIMENT_COHORT_FILE = "experiment_cohort.json"
SMOKE_COHORT_FILE = "smoke_cohort.json"
PROMPT_TEST_COHORT_FILE = "prompt_test_cohort.json"
# The eligible admissions the cohort is sampled from.
FILTERED_POOL_FILE = "filtered_pool.json"

# "brief" and "thorough" each differ from "unformatted" in one paragraph: the
# format instruction and the thoroughness instruction.
PROMPT_VARIANTS = ("brief", "unformatted", "thorough")
DEDUPLICATIONS = ("none", "sentence")

INFERENCE = InferenceConfig(temperature=0.0, max_completion_tokens=4096, frequency_penalty=0.5)

# llama.cpp's DRY sampler penalizes extending a token sequence that already
# appears in the last dry_penalty_last_n tokens, more steeply the longer the
# repeat. The frequency penalty sees only the last 64 tokens, and the loops that
# reach the output cap repeat every 200 to 700. An allowed length of 5 let loops
# through. See "DRY sampler" in docs/specs/prompt-builder-design.md.
DRY_SAMPLER = {
    "dry_multiplier": 0.8,
    "dry_base": 1.75,
    "dry_allowed_length": 3,
    "dry_penalty_last_n": 2048,
}


def inference_for(model_key: str) -> InferenceConfig:
    """The inference settings one model generates with."""
    if model_by_key(model_key).dry_sampler:
        return dataclasses.replace(INFERENCE, dry_sampler=DRY_SAMPLER)
    return INFERENCE


# The oracle selects only on the long strata. Short admissions already fit in
# memory and finish in seconds, so selection has nothing to save there.
ORACLE_STRATA: tuple[str, ...] = TOKEN_STRATA[TOKEN_STRATA.index("32K") :]


def cohort_file(*, smoke: bool, prompt_test: bool = False) -> str:
    """The cohort a run or a preparation reads."""
    if smoke:
        return SMOKE_COHORT_FILE
    if prompt_test:
        return PROMPT_TEST_COHORT_FILE
    return EXPERIMENT_COHORT_FILE


@dataclass(frozen=True, kw_only=True)
class CliParameters:
    """The parameters a command was invoked with.

    ``model`` is None for a command that covers every model of a run.
    ``oracle_n`` None means the model sees every record.
    """

    model: str | None = None
    prompt: str
    deduplication: str = "sentence"
    oracle_n: int | None = None
    smoke: bool = False
    # Defaults to False so run records written before the prompt test cohort still load.
    prompt_test: bool = False

    def __post_init__(self) -> None:
        # The directory is named for the selection, so it can name only one.
        if sum([self.oracle_n is not None, self.smoke, self.prompt_test]) > 1:
            msg = "set at most one of oracle_n, smoke, prompt_test"
            raise ValueError(msg)

    @property
    def cohort_file(self) -> str:
        return cohort_file(smoke=self.smoke, prompt_test=self.prompt_test)

    @property
    def selection_name(self) -> str:
        """The top-level results directory. Sentence dedup is the default, so only none shows."""
        if self.smoke:
            name = "smoke"
        elif self.prompt_test:
            name = "prompt-test"
        elif self.oracle_n is None:
            name = "all-records"
        else:
            name = f"oracle-n-{self.oracle_n}"
        return name if self.deduplication == "sentence" else f"{name}-no-dedup"

    @property
    def run_name(self) -> str:
        """The run's directory under the results root, which every model shares."""
        return f"{self.selection_name}/{self.prompt}-prompt"
