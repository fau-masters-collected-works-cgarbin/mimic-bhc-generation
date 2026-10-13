"""Evaluation pipeline for scoring generated BHC text against reference.

All metrics share the same MetricFn interface, so new metrics are added by
writing a function (or factory) and registering it, with no changes to the
scoring loop. ``config.METRICS`` lists the metrics every run is scored with.

See docs/specs/evaluation-design.md for the design.
"""

import logging
import time
from collections.abc import Callable

from rouge_score import rouge_scorer

from bhc_experiment.concept_spaces import concept_extractor
from bhc_experiment.config import CONCEPT_SPACES
from bhc_experiment.coverage import concept_prf
from bhc_experiment.quickumls_extractor import run_medcon

logger = logging.getLogger(__name__)

# The uniform interface for all metrics. Every scorer is a batch function:
# takes parallel lists of candidate and reference texts, returns one score
# dict per pair in the same order.
MetricFn = Callable[[list[str], list[str]], list[dict[str, float]]]

# The per-pair output: a flat dict mapping metric names (e.g., "rougeL/f1",
# "sapbert/f1") to their scores. Keys take the form "{name}/{submetric}"
# where `name` comes from the metric's entry and `submetric` is fixed
# by the metric implementation.
EvalScores = dict[str, float]

# --- ROUGE metrics ---


def _make_rouge(variant: str, name: str) -> tuple[str, MetricFn]:
    """Create a ROUGE metric function for a rouge_score variant, keyed "{name}/f1"."""
    # Stemming maps inflectional variants to the same root ("prescribed" /
    # "prescribing") so they count as matches. Standard for clinical NLG.
    scorer = rouge_scorer.RougeScorer([variant], use_stemmer=True)

    def _score_pair(generated: str, reference: str) -> dict[str, float]:
        # Library expects (reference, hypothesis) argument order.
        result = scorer.score(reference, generated)
        return {f"{name}/f1": result[variant].fmeasure}

    def fn(cands: list[str], refs: list[str]) -> list[dict[str, float]]:
        # ROUGE scores one document pair at a time, on CPU.
        return [_score_pair(c, r) for c, r in zip(cands, refs, strict=True)]

    return name, fn


# --- BERTScore metric ---


def _make_bertscore(
    model_type: str,
    name: str,
    num_layers: int | None = None,
    *,
    rescale_with_baseline: bool = False,
) -> tuple[str, MetricFn]:
    """Create a BERTScore metric function.

    Returns three scores per pair (F1, precision, recall), each keyed by
    the author-chosen `name` so multiple BERTScore models can coexist in
    the same run.

    rescale_with_baseline subtracts a precomputed CommonCrawl baseline
    from raw scores so random pairs score near 0 and exact matches near
    1, instead of the compressed 0.85-0.95 raw band. Required when
    matching shared tasks that report rescaled numbers (e.g., Discharge
    Me! BioNLP 2024). Only available for models with a precomputed
    baseline shipped by bert_score (distilbert, roberta, deberta, etc.).
    Domain models like SapBERT have no baseline and must keep this False.
    """
    # Suppress the "LOAD REPORT" table and progress bar that transformers
    # emits when loading a model into a different architecture (e.g.
    # classifier head weights are UNEXPECTED when loading into a base
    # encoder). These are expected mismatches for BERTScore's use case.
    # Uses HF's own logging API because transformers uses a custom logger
    # that isn't controlled by Python's stdlib logging module.
    import torch
    import transformers as _transformers
    from bert_score import BERTScorer

    _prev_verbosity = _transformers.logging.get_verbosity()
    _transformers.logging.set_verbosity_error()
    _transformers.logging.disable_progress_bar()

    # bert_score auto-picks CUDA or CPU when device is None but does not
    # check MPS. Pick MPS explicitly on Apple Silicon, otherwise let
    # bert_score handle CUDA/CPU detection.
    device = "mps" if torch.backends.mps.is_available() else None

    logger.info("Loading BERTScore model %s on %s...", model_type, device or "auto")
    try:
        scorer = BERTScorer(
            model_type=model_type,
            num_layers=num_layers,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
            # bert_score builds the baseline file path from the language
            # (rescale_baseline/{lang}/{model_type}.tsv) and refuses to
            # construct a rescaled scorer without one. All notes here are
            # English, so "en" selects the shipped English baseline.
            lang="en",
        )
    finally:
        _transformers.logging.set_verbosity(_prev_verbosity)
        _transformers.logging.enable_progress_bar()

    # Bio_ClinicalBERT and SapBERT ship without model_max_length, which
    # HuggingFace fills in as ~1e30 and the Rust tokenizer then overflows.
    # SapBERT also has 512 position-embedding rows, so a longer sequence
    # raises a size mismatch. deberta-xlarge-mnli needs neither, and is
    # capped anyway so all three score the same window. bert_score does not
    # warn about the truncation (see issues #96 and #117).
    # https://github.com/Tiiiger/bert_score/issues/96
    # https://github.com/Tiiiger/bert_score/issues/117
    scorer._tokenizer.model_max_length = min(scorer._tokenizer.model_max_length, 512)  # noqa: SLF001

    def fn(cands: list[str], refs: list[str]) -> list[dict[str, float]]:
        if not cands:
            return []
        # 32 is safe on MPS. Raise it on CUDA.
        p, r, f1 = scorer.score(cands, refs, batch_size=32)
        return [
            {
                f"{name}/f1": f1[i].item(),
                f"{name}/precision": p[i].item(),
                f"{name}/recall": r[i].item(),
            }
            for i in range(len(cands))
        ]

    return name, fn


# --- UMLS concept overlap (MEDCON-style) ---


def _make_umls_concept(space: str) -> tuple[str, MetricFn]:
    """Create a UMLS concept-overlap metric (MEDCON-style) in a named concept space.

    Returns three scores per pair: F1, precision, recall. F1 is the headline
    metric in the MEDCON literature. P/R are retained for diagnostics.

    Adapts the MEDCON metric (Yim et al. 2023, ACI-Bench) to reward the
    right clinical entities regardless of surface wording, something ROUGE
    and BERTScore miss.

    The extractor is closed over in the returned fn so it loads once per
    scorer and caches per-text results across calls. Scores are keyed by the
    space's ``name``.
    """
    name = CONCEPT_SPACES[space]["name"]
    extractor = concept_extractor(space)

    def _pair_scores(gen_c: frozenset[str], ref_c: frozenset[str]) -> dict[str, float]:
        scores = concept_prf(gen_c, ref_c)
        return {
            f"{name}/f1": scores.f1,
            f"{name}/precision": scores.precision,
            f"{name}/recall": scores.recall,
        }

    def fn(cands: list[str], refs: list[str]) -> list[dict[str, float]]:
        cuis_by_text = extractor.extract_cuis(cands + refs)
        return [
            _pair_scores(cuis_by_text[c], cuis_by_text[r]) for c, r in zip(cands, refs, strict=True)
        ]

    return name, fn


def _make_reference_medcon(name: str) -> tuple[str, MetricFn]:
    """Create the MEDCON metric as the reference implementation computes it.

    Yim et al.'s ``umls_score_individual`` over its own QuickUMLS matcher,
    unmodified, so the score is in the units the MEDCON literature reports.
    It returns F1 alone.
    """

    def fn(cands: list[str], refs: list[str]) -> list[dict[str, float]]:
        pairs = list(zip(refs, cands, strict=True))
        return [{f"{name}/f1": f1} for f1 in run_medcon(pairs)]

    return name, fn


# --- Factory dispatch for parameterized metrics ---

_FACTORY_DISPATCH: dict[str, Callable[..., tuple[str, MetricFn]]] = {
    "rouge": _make_rouge,
    "bertscore": _make_bertscore,
    "umls_concept": _make_umls_concept,
    "reference_medcon": _make_reference_medcon,
}


def build_scorers(metric_configs: list[dict]) -> dict[str, MetricFn]:
    """Build one scorer per entry of a metric list shaped like ``config.METRICS``."""
    scorers: dict[str, MetricFn] = {}
    for entry in metric_configs:
        metric_type = next(iter(entry))  # entry is a single-key {metric_type: params} dict
        key, metric_fn = _FACTORY_DISPATCH[metric_type](**entry[metric_type])
        scorers[key] = metric_fn
    return scorers


def evaluate_batch(
    cands: list[str],
    refs: list[str],
    scorers: dict[str, MetricFn],
) -> list[EvalScores]:
    """Score a batch of (cand, ref) pairs with every scorer.

    Returns one dict per input pair, in the same order, merging the submetric keys
    of every scorer.
    """
    per_pair: list[EvalScores] = [{} for _ in cands]
    for name, metric_fn in scorers.items():
        logger.info("Metric %r: scoring %d pairs...", name, len(cands))
        t0 = time.perf_counter()
        for pair_scores, scores in zip(per_pair, metric_fn(cands, refs), strict=True):
            pair_scores.update(scores)
        logger.info(
            "Metric %r: scored %d pairs in %.1fs", name, len(cands), time.perf_counter() - t0
        )
    return per_pair
