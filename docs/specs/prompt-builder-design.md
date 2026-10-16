# Prompt Builder Design

`src/bhc_experiment/prompt_builder.py` takes an `AdmissionBundle` and a prompt variant and produces the prompt. The resolution order, the filter checks, what a changed prompt does to saved runs, and the chronological sort key are documented at their definitions in that module. This spec covers the experiment-level reasons behind them.

Template files live in `prompts/`. See `prompts/README.md` for the instruction-order conventions inside them.

## Design decisions

**The template resolves per admission.** Ground-truth BHCs use different organizational formats. Matching the prompt's structural instruction to the ground truth keeps format mismatch from inflating ROUGE and BERTScore differences when the clinical content is equivalent.

**The variant and the BHC format are chosen separately.** The prompt variant (`unformatted`, `brief`, `thorough`) is the experimental variable and stays fixed for a run. Under `brief`, the BHC format is picked per admission. They are independent because they test different things: the variant asks whether format guidance or an explicit thoroughness instruction changes generation, while matching the format controls for format-driven evaluation noise.

**Each variant differs from `unformatted` in one paragraph.** `brief` is `unformatted` plus the format paragraph, so comparing the two measures the format instruction alone. `thorough` is `unformatted` with the brevity paragraph replaced by a thoroughness instruction, so comparing the two measures what the model is asked to include. Every other paragraph is identical across all four templates, including the coverage and disposition instructions, which earlier differed between the narrative and domain templates. `TestPromptVariantDifferences` in `tests/test_prompt_builder.py` enforces this.

**The brevity paragraph names a length.** The earlier sentence "Prioritize brevity over completeness of detail" cut Gemma's output under the unformatted prompt to a median 0.33 of the reference length, with 78 of 116 admissions under half. `scripts/analysis/length.py` computes both figures from the archived run `archive/obsolete/case0_minimal/case0_minimal-2026-07-14-dedup-sentence`. The earlier sentence also pulled against the coverage paragraph's "Do not omit documented problems". The paragraph now asks for 250 to 600 words, the rounded interquartile range (242 to 617 words) of the 51,015 successfully extracted BHCs outside the experiment cohort, reported at the top of the same script's output. Leaving out the cohort keeps the scored references out of the prompt. The thoroughness paragraph replaces the length sentence along with the rest of the brevity paragraph.

**The prompts do not ask for abbreviations.** The abbreviation instruction existed to shorten output, and output was already far shorter than the references. The concept extractor does not expand abbreviations, so an abbreviated term can link to a different UMLS concept than the reference's spelled-out term.

**The role is the physician who writes the discharge summary.** A clinical documentation specialist reviews and codes records. The physician writes the hospital course, in the register the references use.

**Format routing stops at narrative vs. domain-organized.** Earlier versions had a separate acute/chronic template and a neonatal standardized template. The acute/chronic template, like the narrative-specific coverage wording, fitted the prompt to finer ground-truth structure and was nearing overfitting. The standardized template never ran because the pool has no standardized admissions. Both are gone. Modifiers stay on `BhcStructure` for cohort profiling.

**Every template is self-contained.** Each template file is a complete prompt. This duplicates instruction text across the five files, but it makes the exact prompt sent to the model directly readable and citable in the dissertation. Auditability wins over DRY here.

**Records are interleaved chronologically.** It mirrors how clinicians experience the record and is supported by Kruse et al. 2025, where combined modalities sorted chronologically outperformed separated modalities.

**One domain template for all numbering conventions.** The delimiter ablation found no measurable score effect from matching the prompt's delimiter to the ground truth. One template serves both conventions. `numbering_convention` stays on `BhcStructure` for cohort profiling and ground-truth provenance. See `docs/archive/prompt-format-delimiter-investigation.md` for the evidence.

## DRY sampler

Llama 3.2 3B generates with llama.cpp's DRY sampler on top of the shared settings. Gemma and Qwen do not. The parameters are `DRY_SAMPLER` in `src/bhc_experiment/config.py`. The per-model switch is `dry_sampler` in `src/mimic_data_prep/models.py`.

**What it fixes.** Generations that reach the 4,096-token output cap are repetition loops. They start early, at a median of about 150 tokens, and repeat a two-paragraph cycle or the whole problem list every 200 to 700 tokens. The frequency penalty counts single tokens over the last 64 tokens only, so it cannot see the previous cycle. DRY penalizes extending a token sequence that already appears in its window. The penalty grows exponentially with the length of the repeat. With multiplier 0.8, base 1.75 and allowed length 3, a 4-token repeat costs 1.4 logits and an 8-token repeat costs 13.1. The 2,048-token window covers the longest loop period.

**Evidence.** A test on Llama compared the current settings with and without DRY on 17 admissions where Llama had truncated before and on the 14 prompt test admissions:

| Set | Condition | Truncated | Median repeated 8-gram rate | Median words |
| --- | --- | ---: | ---: | ---: |
| Loop-prone (17) | without DRY | 4 | 0.225 | 562 |
| Loop-prone (17) | with DRY | 0 | 0.000 | 434 |
| Prompt test (14) | without DRY | 2 | 0.132 | 440 |
| Prompt test (14) | with DRY | 0 | 0.000 | 310 |

DRY ended all 6 loops that reproduced and caused no new truncation. Concept recall did not fall: the paired difference was +0.010 (SD 0.054) on the prompt test set and +0.018 (SD 0.053) on the loop-prone set. An allowed length of 5 let 2 of 4 loops through.

**Faithfulness checks.** Because the DRY window includes the end of the prompt, DRY could push the model away from reproducing record text exactly. Two checks on 25 paired admissions found no harm beyond noise:

- **Drug names and numbers.** Near-miss spellings plus synonym swaps of drug mentions changed by -0.023 (SD 0.103). Numbers absent from the input changed by -0.024 (SD 0.110).
- **End of the stay.** The share of output concepts drawn from the last 2,048 tokens of the records changed by -0.003 (SD 0.095).

With about 25 pairs, the checks rule out only shifts larger than about 0.05. A rare single error, such as one wrong dose in 30 outputs, would go unseen. The six admissions where DRY changed the output most, the baseline loops, have no paired comparison.

**Costs.** DRY also trims outputs that did not repeat, by a median of about 10%. It shortens multi-word prescription names, keeping the drug word and dropping the rest, in 12 of 19 admissions with such names. Concepts echoed from the prompt text rose slightly, from 1.50 to 1.86 per output.

**Why Llama only.** Every loop and every check was on Llama. Gemma never truncated. Qwen truncated 0 to 3 times per 116 admissions. For both, DRY would mostly add its costs. The price is a confound: Llama's completion rate and output length come from a different sampler than the other two models', and cross-model statements about either name it. Qwen's rare loops stay and are reported.

**At 116 admissions.** DRY did not reduce Llama's truncations. Llama truncated 21 times over the three variants, as in July (2 against 5 under `unformatted`, 7 against 4 under `brief`, 12 against 12 under `thorough`). Generation with DRY is also less reproducible. Rerun with the same prompts, Llama's concept F1 moved by a standard deviation of 0.017 to 0.024 per admission, where Gemma and Qwen barely moved. Whether to keep DRY is open.

**Verification.** llama-server accepts a misspelled sampler key without error. After every generation, `verify_sampler_applied` reads the slot's parameters back from `/slots` and stops the run if a DRY setting was not applied.

## Prompt change log

Each entry records one change to the prompt wording, its prompt test result, and the decision. Newest first. Each entry's decision covers that wording change. Which prompt the experiments use is decided in `Prompt selection funnel.md` in the writing vault's experiment design folder. The commit of each tested change is in the prompt test run records, under `git_commit`.

### Version comparison

The versions, each as a change from the one before (-/+ for removed/added text):

- **v0, July** (runs of record, archived under `archive/obsolete/`). The baseline.
- **v1, attempt 1** (commit `2a1783e`). Tested on Gemma `brief` only, stopped after 7 admissions.
  - Role: - "You are a clinical documentation specialist." + "You are the physician writing the discharge summary for this hospital admission."
  - Task boundary: - "Do not re-summarize the history of present illness." + "…summarizes the clinical trajectory from presentation to discharge… Leave the history before presentation to the other sections of the discharge summary."
  - Brevity (`brief`, `unformatted`): - "Prioritize brevity over completeness of detail." + "A typical Brief Hospital Course runs 250 to 600 words. Scale the length to the length and complexity of the stay."
  - Abbreviations: - "Use standard clinical abbreviations (e.g., s/p, b/l, HD, POD, CXR, ACS, TTE, NGT, BID, prn)."
  - Coverage: "Do not omit documented problems or management decisions." moves here from the grounding paragraph. Narrative templates + "including chronic conditions maintained on home medications".
  - Disposition: - "woven into the narrative". Narrative templates + "and any follow-up plans".
  - Title: + its own paragraph, "Output only the Brief Hospital Course text. Do not begin with a title or heading that restates the section name." It replaces the title clauses inside the format paragraphs.
  - Order: role, boundary, content, coverage, disposition, grounding, output, format.
  - Header: - "Service:", - "Length of stay:", + "Medical records:" before the records.
  - Routing: the acute/chronic domain template is gone. Two of the 14 prompt test admissions used it in July.
- **v2, attempt 2** (commit `62e1d60`). Brevity: - "A typical Brief Hospital Course runs 250 to 600 words." + "Write 250 to 600 words."
- **v3, attempt 3** (commit `c6175d2`).
  - Records: - "Medical records:" + "Start of medical records." before the records, + "End of medical records." after them.
  - Closing reminder: + "Write the Brief Hospital Course for this admission now, following the instructions above. Carry the course through the final hospital days. Output only the Brief Hospital Course text, with no title."
  - Grounding: + "Do not label a lab abnormality as a diagnosis unless the notes name it or the records show it was treated. Never write that information is missing from the records. Leave it out."
  - Run setting: Llama generates with the DRY sampler. The wording is unchanged by it.

`scripts/analysis/prompt_versions.py` in the `phd-dissertation-data-prep` repository produces the tables. They give means over the three models, on the prompt test admissions scored in v0, v2 and v3 for that variant and model (truncated outputs have no scores). Each cell is precision / recall / F1. Concept is scispaCy concepts in the MEDCON semantic types. MEDCON is the reference implementation, which stores only F1. Words is the median.

`unformatted` (40 admission and model pairs):

| Version | ROUGE-1 | ROUGE-L | SapBERT | Concept | MEDCON F1 | Words |
| --- | --- | --- | --- | --- | ---: | ---: |
| v0 | 0.428 / 0.209 / 0.266 | 0.212 / 0.103 / 0.131 | 0.727 / 0.715 / 0.721 | 0.165 / 0.116 / 0.126 | 0.170 | 164 |
| v2 | 0.368 / 0.295 / 0.310 | 0.167 / 0.136 / 0.142 | 0.731 / 0.726 / 0.728 | 0.149 / 0.140 / 0.135 | 0.171 | 293 |
| v3 | 0.365 / 0.304 / 0.312 | 0.161 / 0.137 / 0.139 | 0.734 / 0.731 / 0.732 | 0.155 / 0.157 / 0.143 | 0.180 | 306 |

`brief` (41 pairs):

| Version | ROUGE-1 | ROUGE-L | SapBERT | Concept | MEDCON F1 | Words |
| --- | --- | --- | --- | --- | ---: | ---: |
| v0 | 0.394 / 0.280 / 0.313 | 0.185 / 0.130 / 0.146 | 0.744 / 0.738 / 0.741 | 0.190 / 0.156 / 0.157 | 0.187 | 271 |
| v2 | 0.342 / 0.336 / 0.323 | 0.151 / 0.149 / 0.143 | 0.733 / 0.732 / 0.733 | 0.154 / 0.180 / 0.154 | 0.174 | 369 |
| v3 | 0.349 / 0.326 / 0.320 | 0.156 / 0.147 / 0.143 | 0.734 / 0.735 / 0.734 | 0.150 / 0.155 / 0.142 | 0.176 | 368 |

`thorough` (38 pairs):

| Version | ROUGE-1 | ROUGE-L | SapBERT | Concept | MEDCON F1 | Words |
| --- | --- | --- | --- | --- | ---: | ---: |
| v0 | 0.314 / 0.330 / 0.305 | 0.142 / 0.152 / 0.139 | 0.718 / 0.725 / 0.721 | 0.149 / 0.176 / 0.149 | 0.170 | 350 |
| v2 | 0.298 / 0.328 / 0.296 | 0.136 / 0.151 / 0.135 | 0.705 / 0.716 / 0.711 | 0.148 / 0.178 / 0.149 | 0.167 | 378 |
| v3 | 0.292 / 0.346 / 0.302 | 0.132 / 0.158 / 0.137 | 0.713 / 0.722 / 0.718 | 0.134 / 0.179 / 0.142 | 0.165 | 424 |

v1 was never scored. On its 7 Gemma `brief` admissions, ROUGE-1 was 0.384 / 0.223 / 0.266 under v0, 0.338 / 0.281 / 0.296 under v1, 0.315 / 0.301 / 0.295 under v2 and 0.313 / 0.288 / 0.292 under v3. Median words were 178, 273, 306 and 279.

What the tables show:

- **`unformatted` gained the most.** From v0 to v3, recall rose on every metric: ROUGE-1 0.209 to 0.304, concept 0.116 to 0.157, SapBERT 0.715 to 0.731. ROUGE precision fell (0.428 to 0.365) as outputs nearly doubled in length. Concept precision fell less (0.165 to 0.155). F1 rose on every metric.
- **`brief` traded precision for recall, and F1 did not move.** Recall rose at v2 and fell back at v3, all the way on concepts (0.156, 0.180, 0.155). Concept F1 ended below v0 (0.142 against 0.157), and MEDCON F1 too (0.176 against 0.187).
- **`thorough` barely moved.** It was already long under v0. Concept precision fell at v3 (0.148 to 0.134) as outputs grew to a median of 424 words.
- **v3's own effect is small next to v1 and v2's.** Most of the change from v0 came with the length target. v3 moved F1 by at most 0.012 on any variant and metric, the size of the noise at this sample size.

These are 14 admissions per model. A change of about 0.02 in a mean is within noise for a single model, so the pattern across models and metrics carries more weight than any one cell. Every step bundles several changes, so no single instruction's effect can be read from the tables.

### 2026-10-16: thoroughness without the format instruction

Change, in `thorough`:

- **No format instruction.** `thorough` uses one template for every admission, like `unformatted`. The narrative and domain templates are gone. `thorough` is now `unformatted` with the brevity paragraph replaced by the thoroughness paragraph.
- **Specificity in the task's terms.** "Name each distinct clinical concept that the records document rather than summarizing at a high level" became "Name specific diagnoses, procedures, and medications rather than summarizing at a high level".

Reason: stage 1 of the prompt selection funnel kept `unformatted`, so stage 2 tests thoroughness against it, one paragraph apart. `unformatted` guides the model toward the usual content and length of a Brief Hospital Course. `thorough` tells it to list the documented details, with no length limit. The old sentence used the concept metric's unit ("distinct clinical concept") as its vocabulary. The new one asks for the same specificity in the words a physician would use.

The October `thorough` run used the format-routed templates. It is archived, so a new run does not resume against it.

Expected effects, against `unformatted` on the same admissions: longer outputs, higher recall and lower precision on every metric.

Prompt test: pending.

### 2026-10-15: records markers, closing reminder, lab-label guard, filler ban

Changes, identical in every template that has the paragraph:

- **Records markers.** The records block opens with "Start of medical records.", which replaces the "Medical records:" label, and closes with "End of medical records.". A note's free text runs across several paragraphs and often ends with directions, so without an end marker the closing reminder reads like another paragraph of the last note. A matching pair marks one block, where a label reads as a heading. The markers are plain text, which keeps the plain-prose layout the vendor-guidance appendix chose over tags.
- **Closing reminder** (new paragraph after the end marker). "Write the Brief Hospital Course for this admission now, following the instructions above. Carry the course through the final hospital days. Output only the Brief Hospital Course text, with no title." It restates neither the length target nor the format, so each variant still differs from `brief` in one paragraph.
- **Grounding paragraph.** Adds "Do not label a lab abnormality as a diagnosis unless the notes name it or the records show it was treated. Never write that information is missing from the records. Leave it out." The filler ban names missing information, which leaves documented uses such as "etiology unknown" alone.

Reason: an aggregate inspection of the previous change's 126 generations, with scripts that never print text, found these failures:

- **Instructions lose force on long inputs.** 17 of the 18 title echoes came at 64K and above. Llama echoed on 13 of 18 long generations against 1 of 24 short ones. Across the 1,044 July generations, the echo rate rose from 2% at 4K to 53% at 112K. The instructions sit in the system message, ahead of up to about 100K tokens of records.
- **Llama drops the end of long stays.** It recalled 2% of reference concepts that first appear in the last fifth of a long record, against 25% for Gemma and 40% for Qwen.
- **The models turn lab values into diagnosis labels** that appear in neither the input nor the reference (anemia in 12 admissions, thrombocytopenia in 9, hypomagnesemia in 7).
- **"Not documented" phrasing moved inside sentences.** The standalone lines the disposition paragraph bans were nearly gone, but the phrasing appeared inside sentences in 3 of 42 `thorough` outputs (7%), under the measure defined below. An earlier version of this entry gave 24%, which counted phrases per output over a broader phrase list that included "unknown".

Each candidate change was screened for overfitting, by where its evidence came from. A change that answers a model failure, or that follows from the task definition, was kept. A change motivated by overlap with the references would fit the prompt to how MIMIC authors write, and was dropped:

- **Dropped: an opening sentence with age, sex and comorbidities.** It is reference style. The comorbidities the references mention are mostly absent from the input, so asking for them invites fabrication.
- **Dropped: separate length targets for narrative and domain-organized output.** They would add a second reference-derived parameter on top of the format routing, which already uses the ground-truth format.
- **Dropped: medication instructions.** Medications were the weakest concept type on all three models: recall of drug concepts present in the input was 0.16 to 0.41, against 0.41 to 0.63 for diseases. Both versions considered, "name the chronic medications continued" and "state significant medication changes", rest on that overlap with the references. A hospital course describes the treatment for each problem, which the coverage paragraph already asks for as management decisions. Medication changes as such belong to the discharge medication list. The gap is reported as a finding about how small models handle medications.
- **Dropped: "end with the discharge disposition".** The discharge summary is excluded from the input, and disposition terms appear in the input notes of only 38 of the 116 cohort admissions, against 63 references. Gemma stating disposition less often than in July (57% against 93% under `brief`) is consistent with the July outputs inventing it.
- **Narrowed: the lab-label guard.** 78 of the 116 admissions have imaging-only notes, where a managed problem shows up only as labs plus prescriptions. The guard keeps a label when the records show the abnormality was treated.

Repetition loops are left to the sampler. The frequency penalty sees only the last 64 tokens, while the loops repeat every 200 to 700 tokens. Llama now generates with the DRY sampler (see [DRY sampler](#dry-sampler)), so this change's prompt test runs with it.

Expected effects, against the previous prompt test (commit `62e1d60`, 126 generations):

- **Title echoes fall** from 18, all but one at 64K and above. The closing reminder targets them.
- **Llama truncations fall** from 4. DRY targets them.
- **Filler phrasing inside sentences falls.** The general filler ban targets it. The measure counts phrases saying information is missing from the records ("not documented", "not specified", "no information available", "not mentioned") inside a sentence of at least 12 words. "Unknown" is left out of the measure, as it is out of the instruction.

A directional pass is a screen. With 14 admissions, a count can fall by one or two through noise, so a pass says the change shows no sign of being wrong. The full 116-admission run confirms an effect. This is also the third round on the same 14 admissions. The inspection behind this change read their generations. The title-echo evidence comes mostly from the 1,044 July generations, but a drop on these 14 could partly be regression to the mean.

Prompt test: passed every gate (commit `c6175d2`, 126 generations, no failures). Llama ran with DRY, and the `/slots` check confirmed the settings on every Llama generation. Median generated words:

| Variant | Gemma | Llama | Qwen | Strata outside 250 to 600 |
| --- | ---: | ---: | ---: | --- |
| `brief` | 324.5 | 340.5 | 427 | Gemma 4K. Llama 4K, 112K |
| `unformatted` | 269 | 298 | 334 | Gemma 4K, 16K. Llama 8K, 32K, 96K |
| `thorough` (no target) | 372.5 | 467 | 489 | Qwen 8K |

Counts against the previous test, totals over the nine model and variant cells: format breaks under `brief` 2 against 1, truncations 1 against 5, title echoes 1 against 18, standalone missing-information lines 1 against 3. No cell rose by more than one. Against July on the same admissions, the totals are 2, 1, 37 and 3.

Expected effects:

- **Title echoes fell from 18 to 1.** Llama went from 14 to 0. Qwen's remaining echo is the 112K domain-organized admission that also used bold headers in the previous test.
- **Llama truncations fell from 4 to 0.** Llama outputs with any repeated 8-gram fell from 20 to 7. The drop did not hold at 116 admissions (see the full run below).
- **Inline missing-information phrases did not fall.** Outputs with them fell from 3 to 1, but phrases rose from 7 to 14, all in one Llama `thorough` output. Five of the previous test's 7 came from one Qwen repetition loop. With the phrasing this rare, 14 admissions cannot show the ban's effect either way.

Two findings beyond the gates:

- **Grounded precision recovered.** The share of generated concepts found in the input fell in every cell of the previous test as outputs grew. It rose again in 8 of 9 cells, by up to 0.051 (Llama `brief`, 0.500 to 0.551).
- **The format instruction changed no metric beyond noise.** On the 7 domain-organized references, `brief` against `unformatted` differed by at most 0.017 on ROUGE-1, ROUGE-L and SapBERT F1. Gemma and Qwen traded ROUGE precision for recall as `brief` outputs ran 56 to 176 words longer. Llama lost on both ROUGE recall and concept recall under `brief`. Against July, `unformatted` rose on concept F1, ROUGE-1 and SapBERT for all three models, while `brief` and `thorough` stayed level.

Decision: keep the tested prompts for the full run. The filler ban stays, against the rule that drops a change whose expected effect does not move. That rule is for text that does nothing. The ban was tested, and the gates show no harm. What is missing is evidence that it helps, because the behavior it targets is too rare to measure on 14 admissions. Its justification also changes: the 24% figure was wrong, so the ban rests on extending the disposition paragraph's rule against noting absent information to any missing information. The full run measures the inline phrasing at 116 admissions.

Full run: the current prompts ran on all 116 admissions for every variant and model (commit `989d7f5`, 1,044 generations, no failures). `scripts/analysis/run_comparison.py` in `phd-dissertation-data-prep` produces the numbers below. Each comparison pairs a model's untruncated outputs on the admissions both runs scored. Cells read Gemma / Llama / Qwen.

Against July:

| Variant | Concept R | Concept F1 | MEDCON F1 | Grounded P | Median words |
| --- | --- | --- | --- | --- | --- |
| `unformatted` | +0.042 / +0.025 / +0.044 | +0.014 / +0.013 / +0.002 | +0.022 / +0.012 / +0.002 | -0.019 / +0.022 / -0.054 | 138 to 270 / 161 to 270 / 183 to 344 |
| `brief` | +0.025 / +0.018 / +0.024 | +0.008 / +0.013 / -0.006 | +0.006 / +0.016 / -0.007 | -0.011 / +0.004 / -0.041 | 222 to 309 / 232 to 298 / 306 to 434 |
| `thorough` | -0.003 / +0.011 / +0.008 | -0.001 / +0.009 / +0.002 | -0.005 / +0.012 / -0.003 | +0.024 / -0.013 / +0.007 | 355 to 372 / 302 to 333 / 450 to 505 |

- **`unformatted` gained the most, and less than the prompt test showed.** Recall rose on every metric and model as outputs nearly doubled in length. ROUGE and concept precision fell. SapBERT precision held. Concept F1 rose by 0.002 (Qwen) to 0.014 (Gemma), against paired standard deviations of 0.054 to 0.073. On the 14 prompt test admissions, the pooled gain was 0.017.
- **Format matching does not raise the concept scores by itself.** Both October variants have the same length target. `brief` against `unformatted` moved concept F1 by -0.002 to +0.006 and MEDCON F1 by -0.008 to +0.005. On the 67 domain-organized references, only Qwen gained (concept F1 +0.016, recall +0.050), and its `brief` outputs ran a median 152 words longer. In July, where `brief` outputs ran 77 to 196 words longer on those references, `brief` gained on Gemma and Qwen. The July templates differed in more than format. On narrative references, `unformatted` also asked for chronic conditions on home medications and for follow-up plans, and `brief` did not. On domain-organized references, the two differed only in format and in "woven into the narrative". The July format effect went with length, and on narrative references with that wording too.
- **Llama's domain-organized output is poorly grounded.** On domain-organized references, Llama's grounded precision fell from 0.640 under `unformatted` to 0.412 under `brief` (July 0.632 to 0.438). Gemma and Qwen moved by less than 0.03.
- **Truncations.** Llama truncated 2, 7 and 12 times under `unformatted`, `brief` and `thorough`, 21 in all, the same total as July (5, 4 and 12). Qwen truncated 2 times, against 4 in July. Gemma never truncated.
- **Re-run variation.** The 14 prompt test admissions ran again with the same prompts and settings. Qwen's scores were unchanged on every metric. Gemma's changed on a few `brief` and `thorough` admissions, by at most 0.004 in a mean. Llama with DRY changed more: concept F1 moved by a standard deviation of 0.017 to 0.024 per admission. Its means moved by at most 0.006. Llama's re-run variation is a quarter to two fifths of its paired variation against July (0.060 to 0.073).

### 2026-10-14: length target as an instruction

Change, in `brief` and `unformatted`: "A typical Brief Hospital Course runs 250 to 600 words" became "Write 250 to 600 words".

Reason: under the descriptive wording, Gemma's median output was 247.5 words over the first 6 prompt test admissions, just under the target. Three outputs fell below 250 words, two of them single-paragraph narratives at 4K and 8K.

Prompt test: passed the length gate on every variant and model (commit `62e1d60`, 126 generations, no failures). Median generated words:

| Variant | Gemma | Llama | Qwen | Strata under 250 |
| --- | ---: | ---: | ---: | --- |
| `brief` | 328 | 332 | 418 | Llama 16K |
| `unformatted` | 263.5 | 266 | 344.5 | Gemma 4K, 16K, 32K. Llama 4K, 16K |
| `thorough` (no target) | 358.5 | 378 | 480 | none |

Under `brief`, every narrative output had no headers and every domain-organized output used `#` headers, except one Qwen output at 112K that used bold lines (2 of 7 in July). Counts on the same 14 admissions, new against July:

| Variant | Model | Truncated | Title echo | "Not documented" |
| --- | --- | --- | --- | --- |
| `brief` | Gemma | 0 / 0 | 0 / 6 | 0 / 3 |
| `brief` | Llama | 1 / 0 | 4 / 8 | 0 / 0 |
| `brief` | Qwen | 0 / 0 | 1 / 4 | 0 / 0 |
| `unformatted` | Gemma | 0 / 0 | 0 / 0 | 0 / 0 |
| `unformatted` | Llama | 2 / 0 | 5 / 6 | 1 / 0 |
| `unformatted` | Qwen | 0 / 0 | 3 / 0 | 1 / 1 |
| `thorough` | Gemma | 0 / 0 | 0 / 3 | 0 / 4 |
| `thorough` | Llama | 1 / 1 | 5 / 7 | 2 / 0 |
| `thorough` | Qwen | 1 / 0 | 0 / 3 | 2 / 0 |

The "not documented" count matches the phrase anywhere. Most new hits sit inside longer sentences. Every truncation is a repetition loop that hit the 4096-token cap. Llama truncated 4 to 12 of 116 admissions under the July prompts, so 1 or 2 of 14 cannot be told apart from its base rate. Qwen's 3 title echoes under `unformatted` are new: July had none in 116, and all 3 fall at 96K and 112K.

Concept scores moved within noise. scispaCy concept F1 changed by -0.021 to +0.022 against July on every variant and model, with paired standard deviations of 0.046 to 0.079. Under `brief`, recall rose on all three models (Gemma 0.144 to 0.187, Llama 0.120 to 0.138, Qwen 0.203 to 0.212) while precision fell, so F1 stayed flat. Only Gemma's recall gain stands out from noise (paired mean +0.044, SD 0.061, 9 of 14 admissions higher). Llama's and Qwen's gains sit within it. The brevity wording held recall down somewhat, but removing it bought recall at a precision cost and left F1 unchanged.

As written, the regression criterion fails: it allows no truncation and no title echo, and Llama truncates and echoes under every prompt. Read against each model's July counts, title echoes and "not documented" lines fell under `brief` and `thorough`.

Decision: keep the length target. Address the failures the run exposed in the next change. Counts on 14 admissions cannot resolve a model's base rate for truncations, so the full run measures those rates.

### 2026-10-14: length target, physician role, no abbreviation instruction

Changes, identical in every template that has the paragraph:

- **Brevity paragraph** (`brief`, `unformatted`). "Prioritize brevity over completeness of detail" became the 250 to 600 word target.
- **Abbreviation paragraph.** Removed from all five templates.
- **Role.** "A clinical documentation specialist" became the physician writing the discharge summary.
- **Task boundary.** "Do not re-summarize the history of present illness" became a positive statement. The course runs from presentation to discharge. The history before presentation belongs to other sections.

Reason: under the earlier wording, the July runs produced output far shorter than the references. Gemma's median ratio under the unformatted prompt was 0.33. The design decisions above give the evidence for each change.

Prompt test: failed at fail-fast checkpoint 1 (Gemma, `brief`, commit `2a1783e`). Over the 6 admissions from 4K to 16K, the median ratio of generated to reference words was 0.67, below the 0.7 floor. By stratum, with 2 admissions each: 4K 0.81, 8K 0.54, 16K 0.92. No output fell under half the reference length. Format adherence was clean. All 3 narrative outputs had no headers. All 3 domain-organized outputs used `#` headers (6 to 7 each). No output was truncated, restated the title, or wrote a "not documented" line. The run was stopped after 7 of 14 admissions.

Decision: revise the length wording and test again. Two findings also changed the gate for the next attempt. Gemma's output length barely followed reference length (176 to 324 words on references of 197 to 571), so the ratio mostly showed how long each reference was. The three admissions under 0.7 had the three longest references (401 to 571 words). One of them had the best concept F1 of the seven. The gate now checks generated words against the 250 to 600 target the prompt asks for. By that measure attempt 1 also ran short, with a median of 247.5 words over the 6 admissions. The gate changed after this attempt failed, on the length data above. Concept scores did not decide it. On the 7 completed admissions, scispaCy concept F1 was 0.138 against 0.107 for July `brief`, a gain within noise at n=7.

## Appendix: testing prompt changes

A change to the prompt wording runs on the prompt test cohort before the full experiment. The prompt test takes about 4.5 hours for the three variants across the three models, where the full run takes about 27.

The prompt test cohort has 14 admissions, one narrative and one domain-organized reference from each token stratum, drawn by `scripts/pipeline/select_prompt_test_cohort.py`. Wording that meets the length target on a 4K record can miss it on a 112K record, where the instructions sit far from where the model starts writing. The two formats get different format paragraphs under `brief`.

The admissions come from the scored experiment cohort, because only those admissions have earlier generations to compare against. The prompt test therefore gates on output length and format adherence only. Revising wording to raise the prompt test's metric scores would tune the prompt on scored admissions.

Commit the change before running. A run records its commit and resumes only at a clean one. Run every variant. The variants share all but one or two paragraphs, so a change to the wording almost always reaches all three:

```bash
uv run python scripts/pipeline/select_prompt_test_cohort.py
uv run python scripts/pipeline/prepare_inputs.py --prompt-test

for prompt in brief unformatted thorough; do
  scripts/pipeline/run_all_models.sh --prompt $prompt --prompt-test
done

RESULTS=../phd-dissertation-experiment-results/results
for prompt in brief unformatted thorough; do
  uv run python scripts/pipeline/score_results.py $RESULTS/prompt-test/$prompt-prompt
  uv run python scripts/analysis/length.py $RESULTS/prompt-test/$prompt-prompt
  uv run python scripts/analysis/completion.py $RESULTS/prompt-test/$prompt-prompt
done
```

Each count below is compared with the previous prompt test, on the same 14 admissions. A fixed zero is a rate no model meets: Llama truncated and echoed the title under every prompt tested. The previous test differs from the new one only in the change under test, where the July runs differ in much more. At 14 admissions, one admission is the noise floor. Before the run, the change log entry names each count the change is expected to move, and in which direction.

The change passes when all of these hold on the worst model:

- **Length under `brief` and `unformatted`.** The median generated words (`median_words` in `length.md`) fall within the 250 to 600 target. The per-stratum rows show whether the long strata keep the target. The ratio of generated to reference words is context only, because output length barely follows reference length. `thorough` has no length target, so its lengths are reported without a gate.
- **Format adherence under `brief`.** Narrative outputs have no headers. Domain-organized outputs use `#` problem headers. No more outputs break this than in the previous test, plus one.
- **No regressions.** For each model and variant, the count of generations that hit the output cap (`completion.md`), restate the section title, or write a standalone line saying information is not documented is no more than the previous test's count plus one.
- **Expected effects.** Each count the change log entry names moves in the named direction, pooled over the three models.

The report also gives each count's total over the nine model and variant cells, so a small rise spread over many cells shows, and each count against the July runs on the same admissions, so drift over several rounds shows. Both are context. Neither is a gate.

Concept recall and precision show direction only. Fourteen admissions are too few to read the size of a difference.

### Failing fast

Admissions run in cohort-file order, which runs from 4K to 112K, and the 96K and 112K admissions take about 80% of each model's time. `length.py` reads a partial run, because the run record is written before generation starts. Two checks on one model catch a failed change before the full 4.5 hours:

1. **Run `brief` on Gemma alone.** Gemma fell furthest below the reference length in July.

   ```bash
   scripts/setup/serve_local_llm.sh gemma_4_e4b
   uv run python scripts/pipeline/run_experiment.py --model gemma_4_e4b --prompt brief --prompt-test
   ```

2. **After about 2 minutes, check the short strata.** Once 4K through 16K finish, run `length.py` on the run folder. If the median generated words over those admissions fall outside 250 to 600, the change fails. Stop the run and revise the wording.
3. **After about 28 minutes, check every stratum.** Gemma's run is complete. The 96K and 112K rows show whether the target holds when the instructions sit far from where the model starts writing. Score the run with `score_results.py` for the direction of the concept metrics. If the target holds, run the full loop above. It skips Gemma's completed `brief` run.

The run log does not flag a generation that hits the output cap. `length.md` counts those generations, so the same partial check on Gemma's `thorough` run shows the risk early.

When the prompt test is done, move its runs to the results repo's `archive/one-off/`.

## Appendix: vendor guidance on in-prompt formatting

Checked 2026-10-12 for the three models in the experiment. The question was whether the vendors recommend structuring prompt content with markdown headers (`# Role`), XML-style tags (`<role>`), or bold labels (`**Role**:`), how to delimit long input, and whether instructions go before or after it. Chat template control tokens are a separate layer and were out of scope.

None of the three vendors gives guidance on content formatting, long-input delimiting, or instruction placement for these model versions. Their model-specific docs cover the chat template, sampling settings, and context length. Their example prompts are plain prose.

| Model | Sources checked | Content formatting | Long input |
|-------|-----------------|--------------------|------------|
| Llama 3.2 3B Instruct | Meta prompting guide, Llama 3.2 prompt-format page, llama-models prompt format and model card, HF model card, deprecated llama-cookbook prompt engineering notebook | No guidance. The prompting guide covers output format only ("Use bullet points", "Return as a JSON object"). | No guidance. The cookbook RAG example puts the data before the question without stating a rule. |
| Gemma 4 E4B-it | Gemma 4 model card, Gemma 4 prompt formatting page, text chat page, HF model card | No guidance. | No guidance. |
| Qwen3.5-4B | HF model card Best Practices, Qwen3.5 GitHub README, Qwen readthedocs (covers Qwen3 and Qwen3-2507 only) | No guidance. Best Practices covers sampling and task-specific output formats (boxed math answers, a JSON field for multiple choice). | No guidance. YaRN applies past the 262K native context, so 112K inputs do not need it. |

Gemma 4 adds a native `system` role. The older Gemma prompt-structure page still says to fold system instructions into the first user turn. That page predates Gemma 4.

Two general guides give concrete advice, but neither names these models:

- Google's Gemini prompting guide recommends XML-style tags or markdown headings used consistently, long context first, and instructions at the end. The Gemma docs do not reference it.
- Alibaba Cloud's Model Studio prompt engineering guide recommends fencing long reference text with rare delimiters such as `###` or `===`. It does not mention Qwen3.5.

Prompt format can move scores. He et al. 2024 report up to 40% variation across plain text, Markdown, JSON, and YAML on GPT-3.5, with GPT-4 more robust. No study we found measures these three small models or compares markdown headers to XML tags on them.

The templates stay plain prose, one instruction per paragraph, with the instructions in the system message ahead of the records. That layout matches the vendors' own examples. Tags, headers, or instructions after the records would be a new prompt variable backed only by the Gemini guide, so they belong in a separate ablation on top of the current variants.

## References

- Alibaba Cloud. Model Studio prompt engineering guide. https://www.alibabacloud.com/help/en/model-studio/prompt-engineering-guide
- Google. Gemma 4 model card. https://ai.google.dev/gemma/docs/core/model_card_4
- Google. Prompt formatting for Gemma 4. https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
- Google. Gemini API prompting strategies. https://ai.google.dev/gemini-api/docs/prompting-strategies
- He et al. 2024. Does Prompt Formatting Have Any Impact on LLM Performance? arXiv:2411.10541. https://arxiv.org/abs/2411.10541
- Kruse et al. 2025. Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction. Findings of the Association for Computational Linguistics: EMNLP 2025. Reports that combined modalities sorted chronologically outperformed separated modalities.
- Meta. Llama prompting guide. https://dev.meta.ai/llama/docs/how-to-guides/prompting/
- Meta. Llama 3.2 model cards and prompt formats. https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/llama3_2
- Qwen. Qwen3.5-4B model card. https://huggingface.co/Qwen/Qwen3.5-4B
