# Agent Guidelines

This file contains project conventions and guidelines. Do not add function names, signatures, return types, or module-level descriptions here. That information lives in the code (docstrings, type hints). Duplicating it creates staleness and multi-file rename burdens.

## Project overview

See [README.md](README.md) for what this repo does, the codebase structure, and the cohort pipeline summary.

## MIMIC data safety

This repository is code-only. No MIMIC data may be stored here, down to patient IDs. DUA compliance requires this. Aggregate statistics (counts, distributions) are allowed because they are not traceable to individual patients.

MIMIC-derived data (cohort files, experiment results, spot-check reviews) lives in the sibling repo `phd-dissertation-experiment-results` at `~/projects/phd-dissertation-experiment-results/`. The path constants `COHORTS_DIR` and `RESULTS_DIR` in `src/mimic_data_prep/mimic.py` point there.

Do not include PHI or patient-identifiable content (patient IDs, clinical text snippets, dates of service) in commit messages, PR descriptions, console output, or any other metadata visible outside DUA-authorized systems.

Test fixtures must use obviously synthetic values for patient identifiers (HADM_ID, SUBJECT_ID, ICUSTAY_ID). Use small values outside the real MIMIC-III ranges (e.g., HADM_ID=100, SUBJECT_ID=10) or values verified not to exist in MIMIC-III (e.g., HADM_ID=99999). Never use a value that "looks realistic" without checking it against the actual data. Real MIMIC ITEMIDs (lab test codes) are acceptable in test fixtures because they are publicly documented dictionary values that identify no patient.

When adding new output paths or file types, update `.gitignore` before generating the files.

## Experiment design

The README's [Usage](README.md#usage) section has the runs and the results layout, and the notes under [Understanding the experiments](#understanding-the-experiments) have the design and the strata. What neither says, and what matters when planning code changes:

- Cloud models are out of scope. All generation is local, via llama.cpp.
- No retriever is built. Treat any doc that schedules a retrieval build as stale.
- The contribution list was restated on 2026-07-22 and lives in section 1.4 of `dissertation paper.md` in the writing vault. Read it there. Do not copy it here. A copy goes stale.

## Current task

Project status lives in `dissertation execution plan.md` in the writing vault. Read its "Status" section before planning work.

## Understanding the experiments

Before planning code changes to the experiment pipeline, read these in order. Each anchor points at the section that owns the topic. All paths are relative to `~/projects/`.

1. `phd-dissertation-writing/PhD dissertation - temporal EHR summary/Supporting material/Experiment design/Three-case experiment design.md`. The full spec: what each case answers, the feasibility working budget, the per-case mechanics, the data to collect, and the result-table schema. The cases test whether a feasibility-bounded slice of the record lets a small on-device model produce trustworthy Brief Hospital Courses from records too large to run in full on the device.
2. `phd-dissertation-writing/PhD dissertation - temporal EHR summary/dissertation execution plan.md#Working budget framework`. Extended rationale for the feasibility budget (memory, latency, completion) and the deployment tiers, and why the no-degradation finding means quality does not set the budget. The operational definition lives in the full spec above.

## Decision documents

These explain specific design choices. Paths relative to `~/projects/phd-dissertation-writing/PhD dissertation - temporal EHR summary/Supporting material/Experiment design/`:

- `CareVue MetaVision decision.md`: the MetaVision-only restriction.
- `Cohort definition.md`: eligibility criteria for the admission pool.
- `Sampling stratification.md`: the token strata, the tokenizer used for binning, and the per-stratum allocation targeting 132, which drew 116.
- `Prompt variant evaluation.md`: the prompt variant comparison. The variants were renamed on 2026-10-12: `structured` is now `brief`, `minimal` is `unformatted`, and `complete` is `thorough`. `brief` was renamed `format-matched` on 2026-10-16. `thorough` was retired the same day.
- `Prompt selection funnel.md`: the staged test of each prompt instruction, and the one prompt all three models use in the remaining experiments.
- `Deduplication baseline decision.md`: sentence-level dedup adopted as the baseline.
- `Oracle retriever design.md`: the best-possible-selection retriever that drives the retrieval gates.
- `Retrieval decision protocol.md`: the retrieval gate protocol.
- `Retrieval-ceiling gate result.md`: the retrieval-ceiling gate outcome.

## Architecture

### Sibling repositories

See [Prerequisites](docs/setup.md#prerequisites) for sibling repo paths and [Related repositories](README.md#related-repositories) for what each repo provides. Read their source code and docstrings for current APIs. Snapshots here go stale.

Guidelines for working with sibling repos:

- **mimic-bhc-extraction**: Do not reimplement extraction logic. If the extraction API is insufficient, propose changes to that repo first. The reference for DuckDB test fixtures: `tests/test_batch.py`.
- **mimic-admission-explorer**: Copy and adapt query patterns. This repo must not depend on the explorer at runtime. Large tables (CHARTEVENTS, LABEVENTS, INPUTEVENTS, NOTEEVENTS): skip row counts, use targeted queries.
- **mimic-explorer**: Do not re-derive statistics that are already pre-computed there.
- **phd-dissertation-experiment-results**: Read and write it through the path constants in `mimic.py`. Do not duplicate data between repos.

### MIMIC schema reference

**Read [`mimic-admission-explorer/docs/mimic_schema_reference.md`](../mimic-admission-explorer/docs/mimic_schema_reference.md) before writing any version-aware SQL.** It has verified column names, version differences, and join patterns. Do not derive schema at runtime when the information is already in the reference.

### MIMIC-III data location

Reach the CSV.gz files through `table_ref()`, never by building paths. MIMIC-III filenames and column names are both UPPERCASE.

Three pipeline-specific notes the schema reference does not cover:

- `NOTEEVENTS.csv.gz`: filter with `CATEGORY` and `ISERROR` (mostly NULL, non-NULL is '1').
- `CHARTEVENTS.csv.gz`: very large, and not loaded by the experiment pipeline. CareVue and MetaVision use different item IDs.
- `D_LABITEMS.csv.gz`: joined against LABEVENTS to get human-readable test names.

### Data access pattern

Every repo reads MIMIC the same way: an in-memory DuckDB connection over `read_csv_auto`, with no persistent database. Follow what `mimic.py` already does. One pattern for the whole repo. This experiment is MIMIC-III-only, so hardcode the UPPERCASE column names.

### Source packages

The README's Codebase structure section maps the two packages under `src/`. Read module docstrings and function signatures for current APIs.

Design specs in `docs/specs/` document design decisions, contracts, and rationale behind each module. They do not contain code-level detail (dataclass definitions, function signatures, SQL). The code is the source of truth for implementation. Specs explain the "why" that isn't obvious from reading it.

### Run settings

Settings every run shares are constants in `src/bhc_experiment/config.py`, including the metric set and the concept spaces. What varies between runs is a command-line flag. The models are in the registry at `src/mimic_data_prep/models.py`. Do not add YAML config files. Code and results do not use the Case 0/1/2 names. The glossary below has the naming.

### Running experiments

Full invocations for every scenario (smoke test, running across the three models, oracle selection, scoring) are in the README's [Usage](README.md#usage) section, along with the results layout and the resume behavior. The one thing it does not spell out: model directory names are deterministic, which is what makes a re-run resumable.

### Glossary: run, model, series, and cell

- **Run** = one `(selection, prompt)` pair. The directory `results/<selection>/<prompt>-prompt/`. Produced by running one set of settings against the three models. The natural unit of analysis.
- **Model** = one model's generations and scores inside a run. The directory `<run>/<model key>/`, with `run_record.yaml`, `run_results.jsonl`, `eval_results.jsonl`, and `eval_record.yaml`. Within a run the model is the only thing that differs.
- **Series** = one line in a figure or one row in a table. Identifying it takes both the run and the model, so it is neither a run nor a model. Derived at read time and never persisted.
- **Cell** = one cell of a cross-tabulation grid, and only that. The dimensions vary with the grid: `(token_stratum, bhc_category)` for the cohort sampling grid, `(shingle_size, jaccard_threshold)` for a parameter sweep, `(stratum, covariate, level)` for a covariate balance table, `(run, model)` for a comparison table. The word also keeps its ordinary rendering sense, one box in a markdown table. The test is whether the thing sits at the intersection of two or more dimensions. A model's directory does not, so it is never a cell.

The filenames `run_record.yaml` and `run_results.jsonl` are model-level artifacts. Each records that model's generation run. In code, `run_dir` is always a run and `model_dir` is always a model.

**Rule for new code:** "run" for the directory that groups models, "model" for a model's directory inside it, "series" for a display label spanning both, "cell" only for a grid or table cell. Do not use "batch" for a run.

## Dev loop

Setup is in [docs/setup.md](docs/setup.md). Its [Base install](docs/setup.md#base-install) section is enough to run the tests.

Pre-commit hooks run lint, format, and the fast tests on commit. Pre-push runs the full suite including the slow tests, so a push takes longer than a commit. There is no need to run checks manually as a commit gate.

## Git workflow

- NEVER commit or push without explicit user approval. Always show a summary of staged changes and wait for confirmation before running `git commit`.
- Once asked to commit, use the commit skill to do the work.
- Always rebase, never merge. Keep history linear.
- Push directly to `main` when done. No PRs for single-author work.
- Split commits by logical unit (e.g. new module, config change, AGENTS.md update). Don't go back into edit mode just to split changes that got intertwined during development. If they arrived together, commit them together.

## Simplicity first

- Do NOT over-engineer solutions. Start with the simplest approach that works (e.g., rolling median over per-key dicts, single scripts over multi-script splits).
  - Do not add Protocols, helper extractions, extra abstraction layers, or defensive tests beyond what the task explicitly requires. Prefer the leanest change that satisfies the task.
  - Do not write complex, nested list comprehensions. Prefer multi-step code, with clearly-named variables for intermediate steps.
- Do NOT add inferential statistics, dual-field schemas, or parallelism layers unless explicitly requested.
- When refactoring, preserve valuable 'why' comments. Do not delete context-providing comments.

## Search thoroughness

- When asked to verify a change across files, use Grep/Glob exhaustively across the entire repo (code, tests, configs, docs, vault) BEFORE reporting results.
- Never answer 'all instances found' without running a broad search.

## Code style

- Python 3.12+ features are fine.
- Simple for loops over complex comprehensions. Use comprehensions only for simple, single-line transformations.
- Type hints on function signatures.
- SQL injection lint rules are suppressed for `table_ref()` interpolation (see `pyproject.toml`). These SQL fragments are built from local file paths only.
- Use DuckDB parameterized queries (`$1`, `$2`, ...) for any value that originates from user input or variable data (e.g., HADM_ID). Table references from `table_ref()` cannot be parameterized and are interpolated directly.
- Write SQL for explainability. Every query must be explainable in the dissertation, however long that makes it.
  - Use sequential temp tables instead of CTE chains. Each step gets a named table, a step number, and a focused comment. This makes each step independently queryable and verifiable.
  - Add an inline comment on every SQL decision point. WHERE: why this filter. JOIN vs LEFT JOIN: what would be lost. CASE: why these buckets. COALESCE: what NULL means here. GROUP BY: what is being deduplicated. CROSS JOIN: what pattern this implements.
  - Extract repeated SQL patterns (ICUSTAYS dedup, eligible pool) into the shared module (`src/mimic_data_prep/mimic.py`). Inline copies drift.

## Documentation conventions

- Use the christian-writing-style skill when writing and the text-review skill when reviewing and editing.
- Write for a non-expert audience: avoid jargon, explain SQL logic in plain English.
- Use bullet format for lists (do not convert bullets to paragraphs).
- No semicolons, em-dashes, or double-dashes in prose.
- Cite data provenance explicitly.
- **A claim that ranges over models or strata is verified against its worst member.** The aggregate written as if it held everywhere is the recurring failure. A result can hold on the mean and on two models while the third reverses it, or hold per model while one stratum breaks it. Before writing "every", "all three", "dominant", "about half", or a single range, read the per-model and per-stratum columns and quote the member closest to breaking the claim. If the claim survives only on the mean, state the exception in the same sentence. This is the reporting counterpart of "Test what would be believed if it were wrong" under Testing philosophy.
- Lead with questions. Analysis scripts should open their docstring with a "Questions answered:" block listing what the script determines for the decision. Decision document sections should open with a "Questions:" block before presenting results. This creates a traceable thread: decision doc question → script question → query → result → interpretation.
- Do not duplicate SQL in decision documents. The scripts are the authoritative source for queries and are well-commented with inline explanations. Decision documents should reference the script and step number (e.g., "See `eligible_pool_profiling.py`, step 2") and leave the SQL where it runs. This avoids drift between the doc and the script.
- Decision documents follow this structure: Goal (why this matters, what it gates), Parameters and constraints (what to consider), Data to collect (numbered sections, each with a Questions block, then results, interpretation, inline SQL), Decision framework (explicit criteria for each option), Decision, Rationale. See the CareVue/MetaVision decision doc as the reference example.

## Naming conventions

- **Name things for the domain concept.** Use the right abstraction level: an `AdmissionBundle` names what the data represents, where a `PatientBundle` would name who it relates to. Another example: `samples_per_group` counts what is being counted, where `cells_per_stratum` names the statistical framework. Names should read naturally in a sentence describing the pipeline. If a name only makes sense to someone reading the code, it belongs in a local variable. Config keys, class names, and doc headings need names a reader understands without the code.
- **Match the name the output uses.** If the cohort file has `hadm_id`, the log says `HADM_ID`, and the experiment iterates over admissions, then code and docs should say "admission" too. When you notice a mismatch between what the code calls something and what the output calls it, fix it before it propagates.
- **No generic variable names outside a short block.** `entry`, `meta`, `data`, `result`, `config`, `key` are placeholders. If you have to read the assignment to know what the variable contains, rename it: `run_config`, `run_record`, `reviews_data`, `filtered_notes`, `match_key`. Short-lived loop variables (`for key, value in dict.items()`) are fine.
- **Filenames as constants.** When a filename appears in multiple Python files, define it as a constant (e.g., `RUN_RECORD_FILENAME`) and import it everywhere. Docstrings name the concept ("the results file"), so a rename leaves them accurate.
- **Domain language in docs, precise names in code.** README and prose docs use domain terms ("cohort file," "run record," "metric set"). Design specs reference class names in prose but do not duplicate definitions or signatures.
- **Parallel naming for related artifacts.** Files that belong together share a prefix (e.g., `run_record.yaml` and `run_results.jsonl` both start with `run_`).
- **Labels match what is measured.** A name is a claim about what the variable contains. "Data richness" implies a graded quality. The actual variable was a binary presence/absence (any Nursing or Physician* note vs. none). A mismatched label costs a multi-file rename once someone catches it. Before introducing a new label, ask: does this name describe what we measure, or what we wish we measured? If they differ, pick a name for what we measure (e.g., `note_category_availability` for the binary).

## Reproducibility

Every data-producing step must be reproducible from a script checked into the repository. This applies to BHC extraction, cohort sampling, spot-checks, and experiment runs.

- **Scripts over terminal commands.** Any step that produces data or artifacts (extraction, sampling, export) must be a versioned script in `scripts/`. Do not run one-off commands directly in the terminal. The script is the reproducibility record: it captures the exact command, input paths, parameters, and any version dependencies.
- **Pin external dependencies.** When a script depends on output from a sibling repo (e.g., `mimic-bhc-extraction`), the script should log or assert the sibling repo's git commit hash so the result can be traced to a specific code version.
- **Deterministic sampling.** Any random sampling must use an explicit seed. The seed and sample size are parameters in the script, visible at the top.
- **Running a script twice changes nothing the second time.** Running a script twice with the same inputs and parameters must produce identical output. If a script has side effects (e.g., writing files), it should overwrite cleanly.

## Testing philosophy

- Test logic modules with synthetic CSV.gz fixtures (same pattern as `mimic-bhc-extraction/tests/test_batch.py`, the reference for DuckDB test fixtures).
- Mock nothing. DuckDB reads temp files directly in tests.
- **Test what would be believed if it were wrong.** A wrong mean, a wrong `n`, or a row under the wrong stratum reads as a normal result and gets cited. Pin those, with golden values where the arithmetic matters.
- **Don't test what running the script and reading the output already shows.** A mistyped run folder raising `FileNotFoundError`, a figure that never saved, a scatter drawn in the wrong color, `nan` in a table cell: all obvious at a glance, and pinning them buys nothing. The line is whether spotting the defect needs a second source of truth. A wrong color shows itself. A point under the wrong stratum shows up only if you check the figure against the table, so that one gets pinned. If you would have to know the right answer to notice, the test is what supplies it.
- **Don't test what cannot break.** Library behavior you did not write is a given: `write_text` creating a file needs no test. Neither do branches unreachable on real data. `assign_token_stratum` raises above 112K, so no report ever sees a stratum outside `TOKEN_STRATA`, and its fallback needs no test. Verify an edge case against real MIMIC before testing it, the same way you would before guarding it.
- **A test must fail against the bug it names.** Prove it by breaking the code and watching the test fail. `assert "0.123" in table` also passes on `0.123456`, so it never checked the formatting it was named for. Pin whole values.
- Keep tests isolated across API boundaries. Don't cover two functions with one test just to reduce count. If one function breaks, the other's tests should still confirm whether it works.
