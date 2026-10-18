# Prompt Builder Design

## Purpose

The prompt is an experiment variable. If a prompt instruction changes the scores, the result reflects the instruction as well as the model, which confounds the model comparison. We tested three prompt variants to measure the effect of the instructions on the scores.

## The selected prompt

The experiments selected the `prompts/unformatted.txt`.

The prompt was selected in two stages, each testing one question:

- **Stage 1, format.** Does asking for the reference's format change the scores? `format-matched` against `unformatted`.
- **Stage 2, detail.** Should the prompt guide the model toward a typical Brief Hospital Course, or tell it to list the documented details? `detailed` against `unformatted`.

Each stage compares the current winner with one challenger on all 116 admissions for the three models, paired by admission. The stage reads the direction of F1 on ROUGE-L, SapBERT, scispaCy concept and MEDCON. For each model and metric, the F1 moved up when the challenger's mean was higher and more admissions went up than down, and down in the reverse case. The challenger advances when more of the 12 results moved up than down. A tie keeps the incumbent. `unformatted` won both stages. `scripts/analysis/run_comparison.py` in the `phd-dissertation-data-prep` repository computes the stages.

## What each variant tests

### `format-matched`: does the reference's format affect the scores?

`format-matched` is `unformatted` plus a format paragraph, so comparing the two measures the format instruction alone. The hypothesis was that a format mismatch inflates ROUGE and BERTScore differences when the clinical content is equivalent.

- **The template resolves per admission.** The builder picks `prompts/format-matched_narrative.txt` or `prompts/format-matched_domain.txt` from the reference's format.
- **Routing stops at narrative vs. domain-organized.** Finer routing fitted the prompt to the ground-truth structure and was nearing overfitting.
- **One domain template serves every numbering convention.** Matching the prompt's delimiter to the ground truth had no measurable score effect. `docs/archive/prompt-format-delimiter-investigation.md` in the `phd-dissertation-data-prep` repository has the evidence.

Result: of the 12 F1 results, 3 moved up, 6 down and 3 not at all, every one by less than 0.01. ROUGE drops markup in tokenization and SapBERT ignores order, so format matching did not raise the scores.

### `detailed`: guide the model toward a typical BHC, or tell it to list the documented details?

`unformatted` guides the model toward the usual content and length of a Brief Hospital Course. `detailed` (`prompts/detailed.txt`) rewrites three of its paragraphs (scope, detail and coverage) to ask for a complete account that lists every documented problem, diagnosis and medication, with no length target. Every paragraph asks for the same thing, so no sentence asking for a condensed summary pulls against the request.

Result: all 12 F1 results moved down, so stage 2 kept `unformatted`. Outputs grew by a median of 48 to 78 words while concept recall moved by -0.010 to +0.008.

## How every prompt is built

These decisions apply to every template.

- **The role is the physician who writes the discharge summary.** The physician writes the hospital course, in the register the references use. A clinical documentation specialist reviews and codes records.
- **The scope runs from presentation to discharge.** The history before presentation belongs to other sections of the discharge summary, and a positive statement of the boundary tells the model what to cover.
- **The length target comes from references outside the cohort.** A brevity instruction without a number produced output far shorter than the references (Gemma at a median 0.33 of the reference length). The 250 to 600 word target is the rounded interquartile range (242 to 617 words) of the 51,015 successfully extracted BHCs outside the experiment cohort, computed by `scripts/analysis/length.py`. Leaving out the cohort keeps the scored references out of the prompt. `detailed` has no target.
- **The prompts do not ask for abbreviations.** Output already runs shorter than the references. The concept extractor does not expand abbreviations, so an abbreviated term can link to a different UMLS concept than the reference's spelled-out term.
- **Records are interleaved chronologically.** It mirrors how clinicians experience the record. Kruse et al. 2025 found that combined modalities sorted chronologically outperformed separated modalities.
- **Markers enclose the records. A closing reminder follows them.** A note often ends with directions, so without an end marker the closing reminder reads like another paragraph of the last note. The instructions sit ahead of up to about 100K tokens of records and lose force on long inputs, so the reminder repeats the output instructions and asks the model to carry the course through the final hospital days. The reminder restates neither the length target nor the format, so it adds no difference between the variants. The markers are plain text, which keeps the plain-prose layout of the [vendor guidance appendix](#appendix-vendor-guidance-on-in-prompt-formatting).
- **The grounding paragraph guards against lab labels and filler.** The models turned lab values into diagnosis labels that appear in neither the input nor the reference. The guard keeps a label when the records show the abnormality was treated, because in many admissions a managed problem shows up only as labs plus prescriptions. The filler ban extends the disposition paragraph's rule against noting absent information to any missing information.
- **Every template is self-contained.** Each template file is a complete prompt. This duplicates instruction text across the files, but it makes the exact prompt sent to the model directly readable and citable in the dissertation.

## Testing a prompt change

Commit the change first. A run records its commit and resumes only at a clean one. Run each variant whose template the change touches, first on the prompt test cohort, then on the full cohort. The prompt test cohort has 14 admissions, one narrative and one domain-organized reference from each token stratum.

```bash
uv run python scripts/pipeline/select_prompt_test_cohort.py
uv run python scripts/pipeline/prepare_inputs.py --prompt-test
scripts/pipeline/run_all_models.sh --prompt unformatted --prompt-test

RUN=../phd-dissertation-experiment-results/results/prompt-test/unformatted-prompt
uv run python scripts/pipeline/score_results.py $RUN
uv run python scripts/analysis/length.py $RUN
uv run python scripts/analysis/completion.py $RUN
```

Watch length and truncations while the prompt test runs, and stop on a failed gate. The change passes when these hold on the worst model, except expected effects, which are judged pooled over the three models:

- **Length.** The median generated words fall within the 250 to 600 target, where the variant has one.
- **Format adherence under `format-matched`.** Narrative outputs have no headers and domain-organized outputs use `#` problem headers.
- **No regressions.** Truncations, title echoes and standalone "not documented" lines are no more than in the previous prompt test plus one.
- **Expected effects.** Each effect the change log entry names moves in the named direction.

The prompt test admissions come from the scored cohort, so their metric scores show direction only. Revising wording to raise them would tune the prompt on scored admissions. When the prompt test is done, move its runs to the results repo's `archive/one-off/`.

## Generation settings: the DRY sampler

Llama 3.2 3B generates with llama.cpp's DRY sampler on top of the shared settings. DRY penalizes extending a token sequence that already appears in its window, more steeply for longer repeats. The parameters are `DRY_SAMPLER` in `src/bhc_experiment/config.py`. The per-model switch is `dry_sampler` in `src/mimic_data_prep/models.py`.

- **Why Llama only.** Llama's truncations are repetition loops that repeat every 200 to 700 tokens, which the frequency penalty's 64-token window cannot see. Gemma never truncated and Qwen rarely did, so for them DRY would mostly add its costs. The price is a confound: Llama's completion rate and output length come from a different sampler, and cross-model statements about either name it.
- **Verification.** llama-server accepts a misspelled sampler key without error. After every generation, `verify_sampler_applied` reads the slot's parameters back from `/slots` and stops the run if a DRY setting was not applied.
- **Decision.** DRY stays on for Llama, for comparability with the runs already done. At 116 admissions it showed no measured benefit, with truncations matching July's, and it adds variation between re-runs of the same prompt.

## Prompt change log

Each entry records one change to the prompt wording, or a set tested together, with its outcome, in the order tested. The commit of each tested change is in the prompt test run records, under `git_commit`. `scripts/analysis/prompt_versions.py` in the `phd-dissertation-data-prep` repository scores tests 1 to 3 on the prompt test admissions.

`brief` was renamed `format-matched` on 2026-10-16. `thorough`, a format-routed thoroughness variant, was retired the same day.

### Test 1: length target, physician role, no abbreviation instruction

Replaced the brevity sentence with the 250 to 600 word target, removed the abbreviation instruction, made the physician the role and stated the scope positively (commit `2a1783e`), because the July output ran far shorter than the references. The test failed at the short-strata length check on Gemma under `format-matched`. Output length barely followed reference length, so the length gate changed from a ratio to the reference to generated words against the target. Decision: revise the length wording and test again.

### Test 2: length target as an instruction

Changed "A typical Brief Hospital Course runs 250 to 600 words" to "Write 250 to 600 words" (commit `62e1d60`), because Gemma's median output fell just under the target. The prompt test passed the length gate on every variant and model. The regression criterion failed as written, because Llama truncates and echoes the title under every prompt. Decision: keep the target and address the failures in the next change.

### Test 3: records markers, closing reminder, lab-label guard, filler ban

Added the start and end markers, the closing reminder and the grounding guards to every template (commit `c6175d2`). An aggregate inspection of the previous test found title echoes on long inputs, Llama dropping the end of long stays, and lab values turned into diagnosis labels. Candidate changes motivated by overlap with the references were dropped, to keep from fitting the prompt to how MIMIC authors write. The prompt test passed every gate except the expected fall in inline "not documented" phrasing, which is too rare to measure on 14 admissions. On the prompt test admissions, title echoes fell from 18 to 1 and Llama truncations from 4 to 0. The truncation drop did not hold on the full cohort. The filler ban stays because the gates show no harm. In the full run (commit `989d7f5`), `unformatted` gained the most against July on Gemma and Qwen, and less than the prompt test showed.

### Test 4: `detailed`, a request to list the documented details

Added `detailed` (commit `3d87cad`) for stage 2 of the selection, after stage 1 kept `unformatted`. The prompt test passed every gate. The full run moved all 12 selection F1s down, so `unformatted` stays the selected prompt and `detailed` stays runnable as the stage 2 record.

## Appendix: vendor guidance on in-prompt formatting

Checked 2026-10-12 for the three models in the experiment: whether the vendors recommend markdown headers, XML-style tags or bold labels, how to delimit long input, and where instructions go.

- **Model vendors.** The Meta, Google and Qwen docs for these model versions give no guidance on content formatting, long-input delimiting or instruction placement. Their example prompts are plain prose.
- **General guides.** Google's Gemini prompting guide recommends tags or headings, long context first and instructions at the end. Alibaba Cloud's Model Studio guide recommends fencing long text with rare delimiters. Neither names these models.
- **Evidence that format matters.** He et al. 2024 report up to 40% variation across prompt formats on GPT-3.5. No study we found measures these three small models.
- **Conclusion.** The templates stay plain prose, one instruction per paragraph, with the instructions in the system message ahead of the records, matching the vendors' examples. Tags, headers or instructions after the records would be a new prompt variable for a separate ablation.

## References

- Alibaba Cloud. Model Studio prompt engineering guide. https://www.alibabacloud.com/help/en/model-studio/prompt-engineering-guide
- Google. Gemma 4 model card. https://ai.google.dev/gemma/docs/core/model_card_4
- Google. Prompt formatting for Gemma 4. https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
- Google. Gemini API prompting strategies. https://ai.google.dev/gemini-api/docs/prompting-strategies
- He et al. 2024. Does Prompt Formatting Have Any Impact on LLM Performance? arXiv:2411.10541. https://arxiv.org/abs/2411.10541
- Kruse et al. 2025. Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction. Findings of the Association for Computational Linguistics: EMNLP 2025.
- Meta. Llama prompting guide. https://dev.meta.ai/llama/docs/how-to-guides/prompting/
- Meta. Llama 3.2 model cards and prompt formats. https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/llama3_2
- Qwen. Qwen3.5-4B model card. https://huggingface.co/Qwen/Qwen3.5-4B
