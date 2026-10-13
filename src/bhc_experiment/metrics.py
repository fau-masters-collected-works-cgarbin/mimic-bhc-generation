"""Single home for metric-key strings so a typo cannot silently produce an all-NaN column.

Metric keys are `{name}/{submetric}` strings (e.g. "sapbert/f1"), with ``name``
taken from an entry in ``config.METRICS``. A misspelled literal does not raise:
it selects nothing, so a column silently comes back all-NaN. Import the named
constants here instead of retyping the literal.
"""

# ROUGE family
ROUGE1_F1 = "rouge1/f1"
ROUGE2_F1 = "rouge2/f1"
ROUGEL_F1 = "rougeL/f1"

# BERTScore family (one entry per scorer variant).
# The precision and recall submetrics show the coverage asymmetry: a metric
# whose recall trails its precision is reporting missing content.
DISTILBERT_F1 = "distilbert/f1"
DISTILBERT_PRECISION = "distilbert/precision"
DISTILBERT_RECALL = "distilbert/recall"
DISTILBERT_RESCALED_F1 = "distilbert_rescaled/f1"
DEBERTA_F1 = "deberta/f1"
DEBERTA_PRECISION = "deberta/precision"
DEBERTA_RECALL = "deberta/recall"
SAPBERT_F1 = "sapbert/f1"
SAPBERT_PRECISION = "sapbert/precision"
SAPBERT_RECALL = "sapbert/recall"

# scispaCy clinical concept family (MEDCON): precision, recall, F1
SCISPACY_CLINICAL_PRECISION = "scispacy-clinical/precision"
SCISPACY_CLINICAL_RECALL = "scispacy-clinical/recall"
SCISPACY_CLINICAL_F1 = "scispacy-clinical/f1"

# Reference MEDCON: Yim et al.'s scorer over QuickUMLS, run unmodified as the
# literature-units figure. Only F1 is reported (the reference scorer returns F1
# alone).
REFERENCE_MEDCON_F1 = "reference-medcon/f1"

# The project-wide headline quality metric: scispaCy UMLS-concept F1 (the
# MEDCON-style metric).
PRIMARY_METRIC = SCISPACY_CLINICAL_F1
