# Prompt Builder Design

`src/bhc_experiment/prompt_builder.py` takes an `AdmissionBundle` and a prompt variant and produces the prompt. The resolution order, the filter checks, what a changed prompt does to saved runs, and the chronological sort key are documented at their definitions in that module. This spec covers the experiment-level reasons behind them.

Template files live in `prompts/`. See `prompts/README.md` for the instruction-order conventions inside them.

## Design decisions

**The template resolves per admission.** Ground-truth BHCs use different organizational formats. Matching the prompt's structural instruction to the ground truth keeps format mismatch from inflating ROUGE and BERTScore differences when the clinical content is equivalent.

**The variant and the BHC format are chosen separately.** The prompt variant (`unformatted`, `brief`, `thorough`) is the experimental variable and stays fixed for a run. The BHC format is picked per admission. They are independent because they test different things: the variant asks whether format guidance or an explicit thoroughness instruction changes generation, while matching the format controls for format-driven evaluation noise.

**Each variant differs from `brief` in one paragraph.** `unformatted` is `brief` without the format paragraph, so comparing the two measures the format instruction alone. `thorough` is `brief` with the brevity sentences replaced by a thoroughness instruction, so comparing the two measures what the model is asked to include. Every other paragraph is identical across all five templates, including the coverage and disposition instructions, which earlier differed between the narrative and domain templates. `TestPromptVariantDifferences` in `tests/test_prompt_builder.py` enforces this.

**The brevity paragraph names a length.** The earlier sentence "Prioritize brevity over completeness of detail" cut Gemma's output under the unformatted prompt to a median 0.33 of the reference length, with 78 of 116 admissions under half. `scripts/analysis/length.py` computes both figures from the archived run `archive/obsolete/case0_minimal/case0_minimal-2026-07-14-dedup-sentence`. The earlier sentence also pulled against the coverage paragraph's "Do not omit documented problems". The paragraph now gives a typical length of 250 to 600 words, the rounded interquartile range (242 to 617 words) of the 51,015 successfully extracted BHCs outside the experiment cohort, reported at the top of the same script's output. Leaving out the cohort keeps the scored references out of the prompt. The thoroughness paragraph replaces the length sentence along with the rest of the brevity paragraph.

**The prompts do not ask for abbreviations.** The abbreviation instruction existed to shorten output, and output was already far shorter than the references. The concept extractor does not expand abbreviations, so an abbreviated term can link to a different UMLS concept than the reference's spelled-out term.

**The role is the physician who writes the discharge summary.** A clinical documentation specialist reviews and codes records. The physician writes the hospital course, in the register the references use.

**Format routing stops at narrative vs. domain-organized.** Earlier versions had a separate acute/chronic template and a neonatal standardized template. The acute/chronic template, like the narrative-specific coverage wording, fitted the prompt to finer ground-truth structure and was nearing overfitting. The standardized template never ran because the pool has no standardized admissions. Both are gone. Modifiers stay on `BhcStructure` for cohort profiling.

**Every template is self-contained.** Each template file is a complete prompt. This duplicates instruction text across the five files, but it makes the exact prompt sent to the model directly readable and citable in the dissertation. Auditability wins over DRY here.

**Records are interleaved chronologically.** It mirrors how clinicians experience the record and is supported by Kruse et al. 2025, where combined modalities sorted chronologically outperformed separated modalities.

**One domain template for all numbering conventions.** The delimiter ablation found no measurable score effect from matching the prompt's delimiter to the ground truth. One template serves both conventions. `numbering_convention` stays on `BhcStructure` for cohort profiling and ground-truth provenance. See `docs/archive/prompt-format-delimiter-investigation.md` for the evidence.

## Prompt change log

Each entry records one change to the prompt wording, its prompt test result, and the decision. Newest first. The commit of each tested change is in the prompt test run records, under `git_commit`.

### 2026-10-14: length target, physician role, no abbreviation instruction

Changes, identical in every template that has the paragraph:

- **Brevity paragraph** (`brief`, `unformatted`). "Prioritize brevity over completeness of detail" became the 250 to 600 word target.
- **Abbreviation paragraph.** Removed from all five templates.
- **Role.** "A clinical documentation specialist" became the physician writing the discharge summary.
- **Task boundary.** "Do not re-summarize the history of present illness" became a positive statement. The course runs from presentation to discharge. The history before presentation belongs to other sections.

Reason: under the earlier wording, the July runs produced output far shorter than the references. Gemma's median ratio under the unformatted prompt was 0.33. The design decisions above give the evidence for each change.

Prompt test: pending.

Decision: pending.

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

The change passes when all three criteria hold on the worst model:

- **Length under `brief` and `unformatted`.** The median ratio of generated to reference words is between 0.7 and 1.3 in `length.md`. The per-stratum rows show whether the long strata keep the target. `thorough` has no length target, so its ratio is reported without a gate.
- **Format adherence under `brief`.** Narrative outputs have no headers. Domain-organized outputs use `#` problem headers. Check in the results viewer.
- **No regressions.** No generation hits the output cap (`completion.md`), restates the section title, or writes a line saying information is not documented.

Concept recall and precision show direction only. Fourteen admissions are too few to read the size of a difference.

### Failing fast

Admissions run in cohort-file order, which runs from 4K to 112K, and the 96K and 112K admissions take about 80% of each model's time. `length.py` reads a partial run, because the run record is written before generation starts. Two checks on one model catch a failed change before the full 4.5 hours:

1. **Run `brief` on Gemma alone.** Gemma fell furthest below the reference length in July.

   ```bash
   scripts/setup/serve_local_llm.sh gemma_4_e4b
   uv run python scripts/pipeline/run_experiment.py --model gemma_4_e4b --prompt brief --prompt-test
   ```

2. **After about 2 minutes, check the short strata.** Once 4K through 16K finish, run `length.py` on the run folder. If those rows miss the length target, the change fails. Stop the run and revise the wording.
3. **After about 28 minutes, check every stratum.** Gemma's run is complete. The 96K and 112K rows show whether the target holds when the instructions sit far from where the model starts writing. If they hold, run the full loop above. It skips Gemma's completed `brief` run.

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
