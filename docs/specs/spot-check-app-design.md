# Spot-Check Review App Design

`app/spot_check.py` is the reviewer-facing app and `src/mimic_data_prep/spot_check_queries.py` has its queries and case selection. `select_review_cases` documents which admissions get picked and why those strata. This spec covers what the check is for.

## Goal

Verify that the experiment cohort is usable before the experiment pipeline runs against it. The spot check confirms four things per admission:

- The BHC text is coherent (no garbled characters, duplicated paragraphs, or empty template fields).
- The extracted BHC boundaries match the discharge note.
- Source records cover the events the BHC describes.
- The recorded `max_total_tokens` matches the visible content closely enough that the admission's token stratum holds up under inspection.

This is a quality gate. It complements the cohort audit, which reads aggregate distributions per stratum. The spot check reads one admission at a time.

## Why the review covers both BHC formats

The extraction pipeline parses narrative and domain-organized BHCs through different code paths. The review picks both formats at every stratum, so a reviewer exercises both code paths whatever the strata happen to contain.

## Scope decisions

Token-count anomalies go in the free-text notes. Each added checkbox changes what a saved review means, so the checkbox set stays scoped to ground-truth quality and source-record sparsity, per the dissertation's audit procedure.

The grid shows every note, including error notes and pre-admission notes, each one flagged, because the reviewer needs the full database state to debug an oddity. The coverage summary line and the checkboxes exclude those notes from their counts.

Reviews record HADM_IDs, so they are written under `RESULTS_DIR` in the sibling experiment-results repo and cannot be stored here.

## Overlap with the admission explorer

The demographics, note-timeline, and BHC-loading queries overlap with equivalents in the admission explorer (`patient_explorer.py`, `bhc_review.py`, `db.py` in that repo). They are not shared at runtime, which avoids a cross-repo dependency, but they should produce the same results for the same HADM_ID. Cross-check against the explorer when changing one.

## What this app does not do

- It does not verify content alignment between individual notes and BHC claims. That requires reading each source note, which is a different task.
- It does not modify the cohort file. A flagged admission is recorded in the review file for a separate decision.
- It does not display medications, procedures, or other structured data beyond the note, lab, and prescription counts the cohort file records. The coverage check focuses on clinical notes, the primary source for BHC generation.
