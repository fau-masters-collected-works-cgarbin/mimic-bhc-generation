# Prompt Design Principles

This directory contains the prompt templates used to generate Brief Hospital Course (BHC) sections from MIMIC-III clinical records. Each template is a text file split by `---` into a system message (clinical documentation instructions) and a user message (patient demographics + medical records with placeholders).

The prompts are evaluated against ground-truth BHCs extracted from MIMIC-III discharge summaries, using ROUGE, four BERTScore backbones (distilbert, distilbert rescaled against its baseline for comparison with the Discharge Me! shared task, deberta, and SapBERT), and a MEDCON-style UMLS concept overlap. Scoring runs all of them in one pass. The design principles come from clinical documentation conventions, validated against generated output.

See also:

- `src/bhc_experiment/prompt_builder.py` for the assembly mechanics (event sources and sort key).
- `docs/specs/prompt-assembly-choices.md` for the measured serialization overhead against the 112K filter. The full rationale is documented in the dissertation.
- The dissertation for a methodology-level discussion of how corpus-driven design choices show up in the metrics, worked through on one example experiment.

## Prompt variants

A score from one prompt mixes what the model can do with what the prompt told it to do. Two instructions could move the scores on their own:

- **Format.** The format instruction sets the layout of the text. Layout alone can move ROUGE and BERTScore.
- **Brevity.** The brevity instruction tells the model to leave detail out, so low concept recall could be the instruction's doing.

Running the same admissions under more than one prompt separates the effect of each instruction from the model's ability.

Each prompt variant differs from `brief` in one instruction, so a score difference between `brief` and another variant comes from that instruction.

- `brief` is the reference point. It asks for brevity, to bring output close to the length of a ground-truth BHC, and tells the model to use the admission's ground-truth format (narrative or domain-organized).
- `unformatted` is `brief` without the format instruction. Comparing the two shows whether knowing the output format changes what the model writes. The format comes from the ground-truth BHC, so this comparison measures what the model gains from being told the reference's format.
- `thorough` is `brief` with the brevity instruction replaced by an instruction to be thorough. Comparing the two shows whether asking for more raises concept recall. Both use the same format instruction.

Every other instruction is identical in all five templates. `TestPromptVariantDifferences` in `tests/test_prompt_builder.py` checks both differences. The table at the end of **Instruction order** shows the instructions that vary, verbatim.

## Template inventory

| Template | Target structure | When selected |
|----------|-----------------|---------------|
| `unformatted.txt` | Format-agnostic | `variant=unformatted` (all admissions) |
| `brief_narrative.txt`, `thorough_narrative.txt` | Continuous temporal narrative | `variant=brief` or `thorough`, `category=narrative` |
| `brief_domain.txt`, `thorough_domain.txt` | Problem-organized | `variant=brief` or `thorough`, every other category |

Template selection logic is in `src/bhc_experiment/prompt_builder.py`. The `brief` and `thorough` variants use the ground-truth `bhc_structure` category to select the matching template. Routing stops at narrative vs. domain-organized. Modifiers such as `acute_chronic` do not select a template, because fitting the prompt to finer ground-truth structure was nearing overfitting. The pool has no `standardized_template` admissions. If any appear, they get the domain template, the closest format.

The domain-organized templates use `#` headers. Numbered ground-truth admissions route to the same templates. An ablation found no measurable score effect from matching the prompt delimiter to the ground truth, so one template per variant serves both.

## Instruction order

Every template follows the same instruction sequence in the system message, one instruction per paragraph. The paragraphs are grouped by purpose. The task definition comes first so the model has the right frame. What to include (content guidance, coverage, disposition) comes next, then what not to add (grounding), then how to present the text (output only, format, abbreviations).

```text
Shared by all five templates (system message)
│  1  Role and task       "You are a clinical documentation specialist... write the BHC"
│  2  Task boundary (P5)  trajectory only, do not re-summarize the HPI
│  4  Coverage            all active problems, including chronic conditions on home meds,
│                         do not omit documented problems, breadth over depth
│  5  Disposition         disposition and follow-up only if documented, never "not documented"
│  6  Grounding (P3)      only documented facts, no synthesized reasoning or undocumented doses
│  7  Output only         no title restating the section name
│  9  Abbreviations (P2)  s/p, b/l, HD, POD, ...
│  --- user message: demographics + chronological records (identical everywhere)
│
├── Paragraph 3: content guidance        WHY: does asking for more raise concept recall?
│   │
│   ├── Brevity
│   │   │
│   │   └── Paragraph 8: format          WHY: does knowing the output format change generation?
│   │       │
│   │       ├── (absent) ............................................ unformatted.txt
│   │       │
│   │       └── routed by ground-truth bhc_structure.category
│   │           ├── narrative: continuous temporal narrative ....... brief_narrative.txt
│   │           └── anything else: "# Problem:" headers ............ brief_domain.txt
│   │
│   └── Thoroughness
│       │
│       └── Paragraph 8: format, same routing and same text as brief
│           ├── narrative ............................................ thorough_narrative.txt
│           └── anything else ........................................ thorough_domain.txt
```

Notes on individual paragraphs:

- **Coverage.** The coverage wording is identical in every template. Ground-truth narrative BHCs mention chronic conditions and follow-up plans less often than domain-organized ones, but fitting the wording to that difference would mean that switching format also changes what the model is asked to include. The shared wording keeps both clauses. The cohort's references mention chronic conditions in 62% of admissions and follow-up plans in 69% (by heuristics that undercount), so one rule for every admission includes both. Only chronic conditions in narrative references fall below half, at 43%.
- **Disposition.** The prompts request discharge disposition and follow-up plans conditionally: include them only when the records document them, omit them otherwise, and never add a line stating they are absent. This applies P4 and removes the empty-filler lines (e.g., "Follow-up: not documented") that instruction-tuned models produce when told to include content the records lack.
- **Output only.** The model outputs only the BHC text and does not restate "Brief Hospital Course" as a title, because instruction-tuned models otherwise echo the section name as a heading.
- **Format.** The domain-organized templates open with a summary of the admission reason and trajectory before the problem list. That opening matches ~84% of domain-organized ground-truth BHCs by heuristic.

The paragraphs that vary between prompt variants, verbatim:

| Paragraph | `unformatted` | `brief` | `thorough` |
|-----------|-----------|--------------|------------|
| Content | same as `brief` | Do not enumerate individual lab values or imaging findings unless they directly changed management. Prioritize brevity over completeness of detail. | Enumerate every documented problem, diagnosis, and active medication, along with the clinically significant findings that characterize each problem's course. Name each distinct clinical concept that the records document rather than summarizing at a high level. |
| Format, narrative | (none) | Write the Brief Hospital Course as a continuous narrative following the patient's clinical trajectory in temporal order. Do not use labeled sections, numbered entries, or headers. Use dates or hospital day references to mark key transitions. | same as `brief` |
| Format, domain-organized | (none) | Write the Brief Hospital Course organized by clinical problem. Use # prefixed headers for each problem (e.g., "# Sepsis:", "# Acute Kidney Injury:", "# Atrial Fibrillation:"). Within each problem entry, describe the clinical course for that issue. Start with a brief summary of the admission reason and overall trajectory before the problem-by-problem entries. The # problem headers are the only headings. | same as `brief` |

## Design principles

### P1: Clinical brevity

A BHC summarizes the clinical trajectory. It is not a retelling of the full medical record. MIMIC BHCs vary in length (median 397 words, p75 617 words across the 51,131 successfully extracted BHCs in `bhc.jsonl`), but they are concise relative to the volume of source data. Detail that belongs in other discharge summary sections (HPI, physical exam, imaging reports) is generally not repeated in the BHC.

How concise? By heuristic, only ~2% of BHCs include numeric values with explicit clinical units (e.g., "K 3.8 mEq/L", "SBP 90 mmHg"). Imaging results described with anatomical detail or multiple findings (e.g., "CT showed peripancreatic fluid tracking into the left paracolic gutter with associated fat stranding") appear in ~14% of BHCs. The remaining imaging references are brief narrative mentions (e.g., "CXR negative for infiltrate").

The prompt instructions reflect this. Telling a model to "be concise" is too vague. Telling it "do not enumerate individual lab values unless they directly changed management" is actionable.

Example of typical ground-truth brevity (synthetic, based on observed style):

> Pt was started on IV abx and fluid resuscitated. On HD3 developed worsening abdominal distention and tachycardia, transferred to the ICU for closer monitoring.

Two sentences cover initial management, clinical deterioration, and an ICU transfer. No lab values, no specific vital signs, no imaging details.

### P2: Clinical abbreviation style

By heuristic, about 61% of the 51,131 successfully extracted BHCs in `bhc.jsonl` use at least one standard clinical abbreviation ("s/p", "b/l", "HD", "POD", "CXR", "TTE", "prn", etc.). The remaining 39% spell terms out. The prompts unconditionally instruct the model to use abbreviations, accepting the mismatch on the other 39%.

**Effect on metrics.** BERTScore matches tokens by meaning. "s/p CABG" and "status post coronary artery bypass grafting" mean the same thing in context, so the abbreviation mismatch has minimal direct effect on BERTScore. The concept metric extracts UMLS concepts. It does not expand abbreviations: `resolve_abbreviations` is set on the linker and is inert without an abbreviation detector, which this work does not add. Where UMLS carries the abbreviation as one of the concept's own strings, both forms reach the same CUI anyway and the concept sets agree. Where it carries the expansion only as an alias, they can disagree. ROUGE is pure n-gram overlap and would be hurt by the mismatch, but ROUGE is not a priority metric for this work.

**Why we instruct abbreviations anyway.** The primary benefit is indirect: shorter text means less unmatched content, which improves BERTScore precision. Without brevity or abbreviation instructions, models spell out terms and produce output substantially longer than the ground-truth BHCs. The abbreviation instruction contributes to the overall length reduction strategy alongside P1 (clinical brevity) and P5 (task boundary). For the 39% of BHCs that don't use abbreviations, the semantic metrics absorb the text mismatch without meaningful penalty.

### P3: Grounded reporting only

The strongest anti-hallucination guardrail is specificity. "Do not infer diagnoses" is necessary but not sufficient. The prompts go further:

- No synthesized clinical reasoning beyond what the notes state
- No differential diagnoses unless explicitly documented
- No causal interpretations the clinician didn't write
- No medication doses unless documented in the records

Example of the problem: a model generating interpretive headings like "concern for interval ischemia" or adding clinical reasoning (e.g., attributing TTE findings to a specific vascular distribution) that the source notes do not state.

### P4: Conditional content

Instructions should not force content that the input records don't support. If a prompt required a "Transitional Issues" section and the notes contained no follow-up instructions, the model would either fabricate content or generate empty filler. Both hurt scores and correctness.

The templates express this rule as conditional language. It governs discharge disposition and follow-up plans in every template: include them when the records document them, omit them otherwise, and never add a line noting their absence.

### P5: Task boundary definition

A BHC is one section of a discharge summary. It does not duplicate the HPI, the medication list, the physical exam, or the imaging reports. The prompts define what the BHC covers (clinical trajectory, management decisions, disposition) and what it does not (admission history, detailed lab trends, full radiology report text).

This boundary follows clinical documentation standards. The HPI and the hospital course are governed as distinct parts of the record. The HPI is the chronological account of the illness up to presentation (defined by the CMS 1997 Documentation Guidelines for E/M Services), whereas the hospital course narrates the in-hospital trajectory. Kind and Smith (2008) tabulate the Joint Commission's mandated discharge-summary components and place the HPI under "reason for hospitalization" and the hospital course under "procedures and treatment provided". They are separate components with separate purposes. MIMIC discharge summaries reproduce the HPI as its own section inside the same note as the BHC, so "do not re-summarize the history of present illness" enforces a standard boundary that MIMIC's note layout blurs and that instruction-tuned models otherwise cross.

The task boundary instruction is the main control on output length that does not inject ground-truth information into the prompt.

## Scoring considerations

Ground-truth BHCs contain MIMIC de-identification placeholders (`[**Known lastname XXXXX**]`, `[**Hospital1 XX**]`, `[**YYYY-M-DD**]`). Generated text does not and should not attempt to reproduce these. This creates a hard ceiling on token-overlap metrics (ROUGE especially). BERTScore and MEDCON are less affected because they operate at the semantic or concept level.

This is an artifact of the de-identification. No prompt change addresses it, but ROUGE scores should be interpreted with this ceiling in mind.

## Known limitation: problem selectivity

The prompts instruct the model to "include all active clinical problems" and "do not omit documented problems or management decisions." This is a correctness guardrail. Ground-truth authors vary in how selectively they cover problems: some address nearly every documented issue, others aggressively prune to 3-4 major problems even for complex admissions.

In the 116-admission experiment cohort, 9.5% of admissions (11/116) have short BHCs (under 250 words) despite above-median note counts. For these admissions, the attending chose to be highly selective about which problems to cover, while the "include all problems" instruction pushes the model to report every problem the records document. That costs precision whenever the reference author pruned documented problems.

We accept this tradeoff because the alternative is worse. Instructing the model to select which problems matter most would require clinical judgment that the grounding paragraph (P3) explicitly prohibits. The coverage instruction serves the remaining 90.5% of admissions where the ground-truth author covered most documented problems. Input features correlate only weakly with ground-truth BHC length: Pearson r = 0.387 for LOS, r = 0.038 for note count in the 116-admission experiment cohort. A predicted "brevity signal" from input features would add noise as often as signal.

## Record ordering and reproducibility

The same admission has to produce the same prompt bytes on every run, or a score delta cannot be read as a model effect. Record ordering is what decides it:

- Events sort by (effective_time, ROW_ID).
- A record with a date and no time sits at noon. Noon is a guess that no ablation supports.
- ROW_ID breaks ties within each record type, so the same records always come out in the same order.
- Test fixtures for the sort key must use the full datetime MIMIC stores. `PRESCRIPTIONS.STARTDATE` is a midnight datetime (`"YYYY-MM-DD 00:00:00"`), so a date-only fixture hides a malformed key.

## References

- CMS (1997). Documentation Guidelines for Evaluation and Management Services. Centers for Medicare & Medicaid Services. Defines the HPI as "a chronological description of the development of the patient's present illness from the first sign and/or symptom or from the previous encounter to the present," with eight elements (location, quality, severity, duration, timing, context, modifying factors, associated signs and symptoms). Supports the P5 boundary: the HPI ends at presentation and precedes the in-hospital course.
- Kind AJH, Smith MA (2008). Documentation of Mandated Discharge Summary Components in Transitions from Acute to Subacute Care. In Advances in Patient Safety: New Directions and Alternative Approaches (Vol. 2: Culture and Redesign). Agency for Healthcare Research and Quality. Tabulates the Joint Commission-mandated discharge-summary components (standard RC.02.04.01) and places the HPI under "reason for hospitalization" and the hospital course under "procedures and treatment provided". The HPI and the BHC are separate mandated components.
