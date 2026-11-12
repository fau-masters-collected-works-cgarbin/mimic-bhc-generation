# Results Storage Design

`src/bhc_experiment/results.py` documents the JSONL and YAML split, the append-and-flush rule, and how counts are derived. `CliParameters` in `config.py` names the directories. This spec covers the layout as a whole and the decisions a single function's docstring cannot state.

## Directory layout

```
results/
  all-records/                           # input selection: all-records, oracle-n-<N>, smoke, prompt-test, plus -no-dedup
    format-matched-prompt/                   # prompt variant
      outlier_review_notes.json          # optional, run-level
      gemma_4_e4b/                       # model registry key
        run_record.yaml                  # provenance, settings, and status
        run_results.jsonl                # one line per admission (prompt, LLM result, error)
        eval_results.jsonl               # one line per admission (scores, coverage)
        eval_record.yaml                 # what scored it
```

The order of the levels follows the questions the results answer: which records the model saw, then which prompt, then which model. The prompt directory is a run, the unit every analysis script takes, because it groups the three models run under the same settings.

`outlier_review_notes.json` sits at the run level, one directory above each model, because the outlier review notes apply to all the models in the run.

To rerun a model, delete its directory. The results repo's git history keeps replaced runs.

## Design decisions

**Full prompt storage per admission.** The assembled system and user messages are the largest fields in the JSONL but are essential for debugging prompt construction. Without them, diagnosing an odd generation requires re-running the pipeline against the exact same data state. Disk cost is acceptable at experiment-cohort scale.

**The git commit is the provenance.** Settings every run shares are code constants, so the commit recorded in `run_record.yaml` pins them, and generation refuses to start on a dirty tree. The run record also stores the run's CLI parameters, the generation settings, the files it read, and what the server reported about itself, so analysis scripts read what a run used without importing the code it ran at.

**Scoring replaces the eval file.** Every run is scored with the same metrics, so a scoring pass owns the whole `eval_results.jsonl`. What scored it goes once into `eval_record.yaml`.

**A run records the commit its inputs were prepared at.** Preparation writes its commit to the prepared-input `.meta.json`, and generation copies it into the run record as `prepared_commit`. Scoring refuses a run whose inputs were prepared at another commit, because its generations and the ceiling would then describe different records. For the same reason, scoring rebuilds each successfully generated admission's prompt and refuses a run whose stored prompt no longer matches.

## Resume semantics

These rules are not obvious from reading the functions that implement them:

- **A partial run resumes only at the clean commit it started under.** A run in `running` or `interrupted` status continues from the JSONL when HEAD equals the recorded commit and the run did not start on a dirty tree (`BHC_SKIP_GIT_CHECK=1` records `git_dirty: true`). Otherwise it refuses, so one run never mixes generations from two versions of the code. A hard crash never gets to write `interrupted`, so `running` is treated the same way.
- **Failed admissions are not retried.** They count as done, so a run whose every admission failed has nothing left to resume. Resume refuses it, and deleting the directory is how to rerun it.
- **A completed run is skipped.** Deleting its model directory is how to rerun it.

## Not owned by the storage module

- **Which admissions a run covers.** `run_inputs.py` decides that, for generation and scoring alike, from the prepared-input and record-concepts artifacts.
- **Concurrency coordination.** Each model directory is owned by a single process. Runs are sequential, with no multi-runner orchestration.
