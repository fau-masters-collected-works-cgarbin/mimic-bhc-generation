# Experiment Cohort Sampling Design

The stratification variable, the seven strata and their bin convention, the max-of-three binning rule, the asymmetric per-stratum allocation, the seed, the covariate list, and the audit's floor and verdict logic are all documented at their definitions in `sample_cohort.py`, `sampling.py`, and `audit_cohort.py`. This spec records only what those files cannot say from where they sit.

## Where 112K comes from

128K is the binding context window, set by Llama 3.2 3B and Gemma 4 E4B. Qwen3.5-4B is 262K nominal and is not the binding constraint. Every model is served at 131,072 tokens (`CTX_SIZE` in `scripts/setup/serve_local_llm.sh`). The pool filter sits at 112,000, 87.5% of a nominal 128,000, leaving about 19K of the served window for the prompt template, the demographics prefix, and the generation budget.

That reserve is not a fixed cost. Serialization wraps every record in decoration the filter never counted. The overhead scales with the record count, mostly labs. `prompt-assembly-choices.md` has the measured gap on a lab-dense admission. Treat the fit as a soft guarantee, and read that file before relaxing the filter.

## Decisions

**The sampler is consume-only.** After loading the pool it runs no MIMIC queries. Every per-admission field comes from the pool, so a new field is added to the pool first. Otherwise the cohort could have a field the pool never recorded. The pool file would then stop describing what was sampled.

**The audit runs outside the sampler.** Running it inside invites redrawing until the audit passes, which leaves the manifest recording the lucky seed and hides pool problems no seed can fix.

**The sampler has no standardized-template exclusion.** The MetaVision filter already removes every standardized template, so a second exclusion would be redundant.

## Rebuild triggers

The cohort needs to be rebuilt when:

- A model with a smaller binding context window joins the pivot set, since the 112K headroom is tied to 128K.
- Per-model tokenizer versions move and the count differences exceed the headroom.
- The eligibility filters in `mimic.py` change.
- The seven-stratum definition or the binning rule changes.

A rebuild draws a fresh stratified sample from the new pool. It does not patch the previous cohort.
