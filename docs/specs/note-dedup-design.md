# Note Deduplication Design

The motivation is prefill cost on long admissions. Removing duplicate content shrinks the prompt the on-device model must process, where prefill dominates latency and the KV cache dominates memory.

Deduplication runs once in `prepare_inputs.py` and is frozen in the prepared-input artifact, so `--dedup {none,sentence}` selects which artifact to run over. Generation applies no dedup of its own. See `preparation-design.md` for that placement. Only `sentence` is in the runnable recipe. Whole-note is not adopted, per the `Deduplication baseline decision` doc, and the mentions of it below cover the one-time decision studies.

How each method works and its tuning, the detection rules, and excision mechanics are documented at their definitions in `note_redundancy.py`, `sentence_redundancy.py`, and `note_dedup.py`. This spec covers the experiment those enable and the claims that depend on it.

## Design decisions

**The surviving copy is the chronologically earliest one.** A deduplicated fact keeps the timestamp of the note that first recorded it. Keeping a later copy would move the event forward in the patient's history, which concept-overlap metrics cannot detect, so the error would pass every automated check.

**Stratum is a frozen covariate, never recomputed.** Dedup can move an admission across a token boundary. The stratum is assigned once at cohort construction from the pre-dedup `max_total_tokens` and read by every later step, so the comparison stays paired and by-stratum reporting is stable. Token reduction is reported as an outcome. The bins stay fixed.

**Every admission is generated again under each dedup setting**, including the ones dedup did not change. Copying the baseline's output for an unchanged admission would assume the model returns the same text for the same input. It does not. The report finds which admissions actually changed from the `input_tokens` delta.

**Sentence dedup cannot lose a concept.** A removed sentence is byte-identical to a surviving copy. Whole-note dedup has no such guarantee, since a discarded near-duplicate can have a concept the kept note lacks. That is one reason it is not in the recipe.

**Framed as a feasibility and cost measure.** Both methods remove duplicate content only. The bulk of the prompt is distinct notes re-mentioning the same concepts, which they leave in place, so neither solves the token budget. Reach is further bounded by the note share of the prompt, since dedup operates on notes only. Labs and prescriptions pass through untouched, so a lab-dominated admission barely shrinks even when its notes are fully deduplicated. The dup-versus-dedup quality comparison fills a literature gap: no prior BHC study isolates the effect of source-note dedup on output quality (Landi 2023 reports dedup methods and Liu 2022 measures note-bloat harm for clinical NLP, neither for BHC generation).

**The redundancy-versus-length confound is named and left uncontrolled.** A quality change under deduplication cannot be fully attributed to removed redundancy versus simply shorter input, because dedup conflates the two. A length-matched control (dropping random non-duplicate content to the same length) would separate them but is a second experiment, out of scope here.

## Oracle integration

Deduplication does not move the input-coverage ceiling, since sentence dedup is lossless on coverage. An oracle run uses the same dedup method as the all-records run it is compared with, so its generated BHC comes from the same deduplicated input, which lets the retrieval-ceiling gate's recall delta isolate selection from the dedup effect.

## Reporting

Two rules apply to every dedup figure, in this experiment and in the cohort-level reports.

First, state the effect as the ceiling bounded against the labs it cannot touch. Because dedup operates on notes only, a saving is reported as a fraction of the full record (notes, labs, prescriptions, and demographics) with the note share shown, so a small percentage reads as a labs bound.

Second, report per stratum, never pooled, since the same percentage buys disproportionately more latency at the long-context tail.

Because the stratum does not equal the processed token count after deduplication, every table shows both the fixed stratum and the run's actual `input_tokens`, so a reader never mistakes the label for the processed size.

## Not doing

- No within-only or between-only sentence conditions. `sentence` removes both passes, and splitting them is deferred until the combined result warrants it.
- No peak-memory instrumentation. Per-run memory footprint is a separate measurement task.

## References

Method anchors (Gabriel 2018, Lee 2022, Landi 2023) are listed in the Deduplication methods doc. The references below support this spec's literature-gap framing.

- Landi et al. 2023. Clinical text deduplication practices for efficient pretraining and improved clinical tasks. arXiv:2312.09469.
- Liu et al. 2022. "Note Bloat" impacts deep learning-based NLP models for clinical prediction tasks. Journal of Biomedical Informatics 133:104149.
- Aali et al. 2024. MIMIC-IV-BHC, the closest BHC dataset and benchmark, whose preprocessing normalizes text and filters by length only, with no source-note dedup.
