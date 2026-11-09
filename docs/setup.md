# Setup

One-time setup for the repository. Run it once, top to bottom, before the Usage steps in the [README](../README.md). Model weights, tokenizers, and the scispaCy bundle are cached after the first download, so reruns are cheap.

## Prerequisites

- [uv](https://docs.astral.sh/uv/).
- Python 3.12 (pinned because of the [UMLS scoring dependency](#umls-concept-overlap)). After cloning, run `uv python pin 3.12`.
- Local copy of MIMIC-III v1.4 (CSV.gz files from PhysioNet) at `../mimic-iii/physionet.org/files/mimiciii/1.4/`. If your data lives elsewhere, change `MIMIC_DIR` in `src/mimic_data_prep/mimic.py`.
- BHC extraction output from [mimic-bhc-extraction](https://github.com/fau-masters-collected-works-cgarbin/mimic-bhc-extraction) at `../mimic-bhc-extraction/bhc.jsonl`. Run that repo's extraction pipeline first to produce the JSONL file.
- Directory to store experiment data: `mkdir ../phd-dissertation-experiment-results/`. Cohort files and experiment results are stored there (DUA compliance: no MIMIC data in this code repo).
- A local llama.cpp install with weights for Gemma 4 E4B, Qwen3.5-4B, and Llama 3.2 3B. Every generation run, including the smoke test, goes through this local server. See the [Local LLM](#local-llm) section.

## Base install

Installs the Python dependencies and pre-commit hooks. Everything else builds on this.

```bash
uv sync --all-groups
uv run pre-commit install
```

## Hugging Face tokenizers

Required because the pool builder counts tokens per admission with the actual tokenizer for each of the three target models (Gemma 4 E4B, Qwen3.5-4B, Llama 3.2 3B).

Licenses:

- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct): Llama 3.2 Community License. Requires a token to download (see below).
- [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it): Apache 2.0. Can be downloaded without a token.
- [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B): Apache 2.0. Can be downloaded without a token.

### Get a Llama access token

Sign in to Hugging Face and click "Agree and access" on the [Llama 3.2 3B Instruct page](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct). Approval is automatic.

At <https://huggingface.co/settings/tokens>, create a token with the **Read** scope. The token is required for the gated Llama download.

Add the token to your `.env` file (gitignored):

```bash
HF_TOKEN=hf_your_token_here
```

`transformers` and `huggingface_hub` both pick up `HF_TOKEN` from the process environment.

### Verify

Confirm all three tokenizers download and load without errors:

```bash
uv run --env-file .env python -c "
from transformers import AutoTokenizer
for repo in ['meta-llama/Llama-3.2-3B-Instruct', 'google/gemma-4-E4B-it', 'Qwen/Qwen3.5-4B']:
    tok = AutoTokenizer.from_pretrained(repo)
    print(f'{repo}: vocab={tok.vocab_size}')
"
```

The first run downloads each tokenizer (~10-30 MB per model, cached under `~/.cache/huggingface/hub/`). Re-runs read from cache.

A `401 Unauthorized` on Llama means the license has not been accepted on your HF account, or `--env-file .env` was omitted so `HF_TOKEN` never reached the subprocess. A `403 Forbidden` means the token has the wrong scope (needs Read).

## BERTScore models

BERTScore scoring uses three HuggingFace models, configured in `METRICS` in `src/bhc_experiment/config.py`:

- `distilbert-base-uncased` (~250 MB)
- `microsoft/deberta-xlarge-mnli` (~750 MB)
- `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (~430 MB)

They are downloaded on the first scoring run, but the large download can be a surprise at that point. Pre-download all three so scoring doesn't stall:

```bash
uv run python -c "
from transformers import AutoTokenizer, AutoModel
for repo in [
    'distilbert-base-uncased',
    'microsoft/deberta-xlarge-mnli',
    'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
]:
    AutoTokenizer.from_pretrained(repo)
    AutoModel.from_pretrained(repo)
    print(f'{repo}: ok')
"
```

Cached under `~/.cache/huggingface/hub/`.

Expect a list of `UNEXPECTED` keys for DistilBERT (`vocab_*`) and DeBERTa (`pooler.*`, `classifier.*`) on every load. The original checkpoints include task-specific heads (MLM head and MNLI classifier respectively) that the base encoder class doesn't use. `AutoModel` drops them. BERTScore only needs the encoder, so the dropped heads are exactly the right ones. SapBERT's checkpoint has no extra heads and loads cleanly.

Optional: once all three are cached, add `HF_HUB_OFFLINE=1` to `.env` to skip per-run network checks against the HuggingFace Hub. Remove it when you need to download a new model.

## UMLS concept overlap

The UMLS concept overlap metric is a MEDCON-style clinical concept overlap score, adapted from Yim et al. 2023 (ACI-Bench) by substituting scispaCy's UMLS linker for QuickUMLS. See `SCISPACY_CLINICAL` in `src/bhc_experiment/config.py` for the full provenance and `src/bhc_experiment/umls_extractor.py` for the `UmlsExtractor` class used by the metric.

### Install

The [base install](#base-install) includes scispaCy and its model. Verify the imports and model load before proceeding:

```bash
uv run python -c "import spacy, scispacy; from scispacy.linking import EntityLinker; spacy.load('en_core_sci_md'); print('ok')"
```

The `en_core_sci_md` model is declared as a dependency in `pyproject.toml`. Pinning it there matters because the model wheel caps `spacy<3.8` while scispaCy 0.6.2 allows `spacy<3.9`. Declaring the model lets that cap reach the resolver, which would otherwise pick spaCy 3.8.x and break the model load.

The UMLS linker bundle (~1 GB, cached under `~/.scispacy/`) is downloaded on the first run.

The lock file pins a prebuilt Python 3.12 macOS wheel for `nmslib`, a scispaCy dependency. If `uv sync` tries to build `nmslib` from source, verify you are on Python 3.12 (run `uv python pin 3.12` if not, since scispaCy 0.6.2 rejects 3.13).

scispaCy models are hosted on AI2's S3 bucket, which `python -m spacy download` does not read. To bump the model, change the URL in the `pyproject.toml` dependencies and re-lock with `uv lock`. The model version in the URL (`v0.5.4`) does not have to match the installed scispaCy version, since scispaCy reuses model tarballs across package releases that share a spaCy base. To find the URL for a newer scispaCy release, fetch its tagged README and grep for the model name:

```bash
SCISPACY_VERSION=$(uv pip show scispacy | awk '/^Version:/ {print $2}')
curl -s "https://raw.githubusercontent.com/allenai/scispacy/v${SCISPACY_VERSION}/README.md" | grep en_core_sci_md
```

### Validate

Run the smoke script to confirm the linker produces sensible CUI sets:

```bash
uv run python scripts/setup/umls_concept_smoke.py
```

Expect F1 = 1.0 for the identical pair, 0 < F1 < 1 with precision higher than recall for the partial-overlap pair (candidate's concepts are a subset of the reference's), and ~0.0 for the disjoint pair. Values outside those ranges usually mean the threshold is too high or the TUI filter is too narrow.

Expect two warnings on every run, both harmless: a spaCy `FutureWarning` about a set union when the model loads, and one or more sklearn `InconsistentVersionWarning` entries about unpickling `TfidfTransformer` / `TfidfVectorizer` from version 1.1.2 under a newer sklearn. AI2 ships scispaCy bundles pickled at the sklearn version that existed when the model was last retrained. sklearn's compatibility layer still loads them correctly.

### QuickUMLS

The reference MEDCON implementation uses QuickUMLS instead of scispaCy. It needs Python 3.10 and a UMLS license, so it lives in a sibling repository. See [QuickUMLS setup](quickumls-setup.md).

Scoring requires it: `config.METRICS` includes the reference MEDCON metric.

## Local LLM

Experiments run inference locally via llama.cpp at q4_k_m, against the fixed three-model set (Gemma 4 E4B, Qwen3.5-4B, Llama 3.2 3B). One script downloads the pinned llama.cpp release and the pinned model GGUFs into `local-llm/`. One-time setup, reused across every run.

### Install

```bash
scripts/setup/fetch_llama_and_models.sh
```

The script writes to:

- `local-llm/llama.cpp/`: prebuilt llama.cpp binaries (pinned release tag).
- `local-llm/models/`: GGUF files for the three models, pinned by HuggingFace revision SHA.

All pins live at the top of `scripts/setup/fetch_llama_and_models.sh`. To bump a pin, edit the constant, delete the target (`rm -rf local-llm/llama.cpp` or `rm local-llm/models/<file>`), and re-run. `local-llm/` is gitignored, so the script itself is the reproducibility record.

### Verify

```bash
# `--list-devices` prints the available compute devices
local-llm/llama.cpp/llama-server --list-devices

# Should list three GGUF files, one per model
ls -lh local-llm/models/
```

### Launch the server

The pipeline talks to llama.cpp's OpenAI-compatible HTTP server. One server instance serves one model, so launch one at a time and restart when switching.

Launch the server with:

```bash
scripts/setup/serve_local_llm.sh gemma_4_e4b # or qwen_3_5_4b, llama_3_2_3b
```

The script kills any running llama-server first and sets the context window to 128K (131072 tokens). It reads the model's server alias, weights file, and flash-attention default from the model registry in `src/mimic_data_prep/models.py`. Stop with Ctrl+C, or `pkill -f llama-server`.

Verify the server is up and generating responses:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":300}' \
  | jq
```
