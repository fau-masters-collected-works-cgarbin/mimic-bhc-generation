# Prompt assembly: the context-window overflow margin

What the assembler does (event sources and sort key) is in `src/bhc_experiment/prompt_builder.py` and `src/bhc_experiment/loader.py`. This file records how much headroom the 112K pool filter actually leaves.

## Context-window overflow

The assembler does not cap or pre-budget the prompt. Two layers handle overflow instead. The cohort filter in `src/mimic_data_prep/cohort.py` excludes admissions above 112K tokens (87.5% of the 128K nominal window), so admissions are expected to fit. If one still exceeds the model's window at run time, the inference call fails on that admission, which makes a bad cohort filter loud. This relies on the local inference path erroring on overflow instead of truncating the context silently, a property of the llama.cpp wrapper configuration and not of the assembler.

The 112K filter counts raw record content under each tokenizer with special tokens excluded, so it does not bound the served prompt. Serialization wraps every record in a `[timestamp] TYPE: ...` form. That decoration is outside the filter's count. The overhead grows with the record count, dominated by labs.

In one observed admission the 2,745 lab lines added about 15K tokens of bracket, keyword, and parenthesis decoration, and the served prompt ran about 18K above the raw-content figure the filter checked. A lab-dense admission near the 112K cap can therefore reach or exceed the 128K window before the generation budget, so the fit is a soft guarantee.

Relaxing the filter beyond 112K is bounded by this overhead as well as by content. The ~16K headroom quoted in `experiment-cohort-sampling-design.md` is a nominal reserve that shrinks with the record count.
