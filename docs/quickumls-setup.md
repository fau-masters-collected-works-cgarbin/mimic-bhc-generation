# QuickUMLS setup

One-time setup for the QuickUMLS extractor, the reference MEDCON implementation from Yim et al. 2023 (ACI-Bench).

QuickUMLS needs Python 3.10, while this repository pins 3.12 for scispaCy. The two cannot share an environment, so QuickUMLS gets its own directory.

These instructions were tested on macOS. Most of the steps should work on Linux as well, but the toolchain commands may differ.

## Prerequisites

- [uv](https://docs.astral.sh/uv/), the same one this repository uses.
- A working C++ toolchain, because QuickUMLS depends on `leveldb`, which compiles from source.
- A UMLS Metathesaurus license. Approval takes one to three business days, so start there.
- Roughly 40 GB of disk space during installation and 6 GB after installation.

## Get a UMLS license

Request an account at <https://uts.nlm.nih.gov/uts/signup-login>. Once approved, copy the API key from the UTS profile page. The download step below is the only place it is used.

## Locations

Both destinations sit alongside this repository, matching how the rest of the project locates its dependencies:

- `../phd-dissertation-quickumls/`, the QuickUMLS directory.
- `../umls/`, with the RRF files, the built index, and the CUI to TUI map. Licensed UMLS data, so it never enters git.

Create them now, from the root of this repository, so every step below can assume they exist:

```bash
mkdir -p ../umls ../phd-dissertation-quickumls
```

## Confirm the toolchain

```bash
printf '#include <string>\nint main(){ std::string s="ok"; return 0; }\n' > /tmp/t.cpp && /usr/bin/c++ /tmp/t.cpp -o /tmp/t.out && echo TOOLCHAIN OK
```

Must print `TOOLCHAIN OK`. On macOS, if it fails on a missing C++ standard library header, delete `/Library/Developer/CommandLineTools/usr/include/c++/v1` and rerun. `xcode-select --install` does not fix this case.

## Download UMLS 2022AA

Download 2022AA, the release Yim's `SETUP.md` names (~5GB). Do not use a different release to keep the scoring consistent with published work.

From the root of this repository:

```bash
export UTS_API_KEY='<your key here>'
cd ../umls
curl -L -o umls-2022AA-metathesaurus.zip \
  "https://uts-ws.nlm.nih.gov/download?url=https://download.nlm.nih.gov/umls/kss/2022AA/umls-2022AA-metathesaurus.zip&apiKey=$UTS_API_KEY"
```

Extract the two files QuickUMLS needs and verify they are there:

```bash
unzip -j umls-2022AA-metathesaurus.zip '*/META/MRCONSO.RRF' '*/META/MRSTY.RRF' -d 2022AA/
ls -lh 2022AA/
```

Those two files are everything QuickUMLS needs. Delete the archive to save space:

```bash
rm umls-2022AA-metathesaurus.zip
```

Build the CUI to TUI map while `MRSTY.RRF` is on disk:

```bash
awk -F'|' '{print $1"\t"$2}' 2022AA/MRSTY.RRF | sort -u > 2022AA/cui_to_tui.tsv
```

## Prepare the QuickUMLS directory

Use these exact dependencies. They match Yim's configuration, our reference implementation. Changing any of the components changes which concepts get matched.

```bash
cd ../phd-dissertation-quickumls
uv python pin 3.10
uv init --bare
uv add 'quickumls==1.4.2' 'spacy==3.4.4' 'nltk==3.8.1' 'numpy==1.26.4' 'setuptools==80.9.0'
uv add 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1-py3-none-any.whl'
uv add 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz'
```

QuickUMLS loads `en_core_web_sm` itself, hardcoded in `constants.py:SPACY_LANGUAGE_MAP`, to tokenize and tag behind every match. `UMLS_evaluation.py` loads `en_ner_bc5cdr_md` at module scope, so the scorer will not import without it. Yim's other pins (`scispacy`, `en_core_sci_lg`, `en_core_web_md`) serve his BART and retrieval baselines.

Example of variations: `medspacy_quickumls`, a maintained fork, collects candidates in `_get_all_matches` into a dictionary keyed by concept ID, so one ID survives at most once per span. Yim's collects them into a list and drops a repeat only when it appears consecutively, so the same ID can survive twice under different matched terms. Yim's scorer counts matched terms, so one concept mentioned twice counts twice. This difference can affect the final score.

The numpy and setuptools pins keep spaCy 3.4.4 working. It fails to import under numpy 2 or setuptools 81 and later.

Confirm the environment before building the index, which takes far longer to fail:

```bash
uv run python -c "from quickumls import QuickUMLS; print('OK')"
```

Must print `OK`. A `pkg_resources is deprecated` warning above it is expected.

## Build the index

Run the command from [Yim's `SETUP.md`](https://github.com/wyim/aci-bench/blob/main/SETUP.md), changing only the paths. It runs from the QuickUMLS directory, because `uv run` needs the project there:

```bash
mkdir -p ../umls/quickumls-2022AA
uv run python -m quickumls.install ../umls/2022AA ../umls/quickumls-2022AA
```

The build writes several gigabytes and takes a long time. Confirm it finished before moving on:

```bash
cd ../umls
ls -lh quickumls-2022AA/
du -sh quickumls-2022AA/
```

Expect four `.flag` files recording the install options, plus two database directories. An empty or near-empty directory means the build died partway. Repeat the instructions if necessary.

Create a provenance file for the index to record how the files were created and the version of the RRF index files (through their SHA256 hashes):

```bash
cat > quickumls-2022AA/install.json <<EOF
{
  "umls_release": "2022AA",
  "download_url": "https://download.nlm.nih.gov/umls/kss/2022AA/umls-2022AA-metathesaurus.zip",
  "mrconso_sha256": "$(shasum -a 256 2022AA/MRCONSO.RRF | cut -d' ' -f1)",
  "mrsty_sha256": "$(shasum -a 256 2022AA/MRSTY.RRF | cut -d' ' -f1)",
  "install_command": "python -m quickumls.install ../umls/2022AA ../umls/quickumls-2022AA"
}
EOF
cat quickumls-2022AA/install.json
```

## Copy Yim's scorer

Yim's scorer reads its data files through paths relative to its own location, so it only works inside the directory layout it was written for. Reproduce that layout exactly:

```text
phd-dissertation-quickumls/
└── aci_bench/
    ├── evaluation/
    │   ├── UMLS_evaluation.py
    │   └── semantics.py
    ├── resources/
    │   ├── semantic_types.txt
    │   └── des -> ../umls/quickumls-2022AA
    └── PROVENANCE
```

Both `evaluation/` and `resources/` have to stay as directories. The scorer looks for `resources/` one level above itself, so moving the two `.py` files up into `aci_bench/` breaks it.

`des` is a symlink to the index built above. That name is not a choice: the scorer hardcodes it. Leave the downloaded files unmodified, including that path.

The download pins a commit. aci-bench publishes no releases or tags. Its default branch still receives edits, so the commit is the only stable way to name a revision. All three files were added in commit `5d3cd4d` on 2023-07-01 and none has changed since. `PROVENANCE` records it next to the copies.

```bash
cd ../phd-dissertation-quickumls
mkdir -p aci_bench/evaluation aci_bench/resources

REPO=wyim/aci-bench
COMMIT=5d3cd4d8a25b4ebb5b2b87c3923a7b2b7150e33d
for f in evaluation/UMLS_evaluation.py evaluation/semantics.py resources/semantic_types.txt; do
  curl -fsSL -o "aci_bench/$f" \
    "https://raw.githubusercontent.com/$REPO/$COMMIT/$f"
done
printf '%s\ncommit id: %s\n' "$REPO" "$COMMIT" > aci_bench/PROVENANCE

# Absolute target, because a relative one would resolve against the symlink's
# own directory, not the QuickUMLS directory
ln -s "$(cd ../umls/quickumls-2022AA && pwd)" aci_bench/resources/des
```

Confirm the layout and that the symlink resolves to a populated index:

```bash
find aci_bench -maxdepth 2
ls aci_bench/resources/des/
```

## Verify

### The 66 semantic types

Yim's list has to match `MEDCON_DEFAULT_TUIS` in this repository element for element. Dump both and diff them.

This check uses both. Switch to this repository first:

```bash
cd ../phd-dissertation-data-prep
uv run python -c "
from bhc_experiment.umls_extractor import MEDCON_DEFAULT_TUIS
print('\n'.join(sorted(MEDCON_DEFAULT_TUIS)))
" > /tmp/tuis-ours.txt
```

Then back to the QuickUMLS directory to dump Yim's list and compare:

```bash
cd ../phd-dissertation-quickumls
uv run python -c "
import sys; sys.path.insert(0, 'aci_bench/evaluation')
from semantics import SEMANTICS
print('\n'.join(sorted(SEMANTICS)))
" > /tmp/tuis-yim.txt
diff /tmp/tuis-ours.txt /tmp/tuis-yim.txt && wc -l < /tmp/tuis-ours.txt
```

Expect no diff output and a count of 66. A mismatch means the two extractors would find different sets of concepts, which defeats the reason for running QuickUMLS.

### Smoke test under Yim's configuration

```bash
cd ../phd-dissertation-quickumls
uv run python -c "
import sys; sys.path.insert(0, 'aci_bench/evaluation')
from semantics import SEMANTICS
from quickumls import QuickUMLS
m = QuickUMLS('aci_bench/resources/des', window=5, threshold=1, accepted_semtypes=SEMANTICS)
print(m.get_info())
print(m.match('PT was elevated.', ignore_syntax=True))
print(m.match('myocardial infarction', ignore_syntax=True))
"
```

Expect:

- Threshold 1 with minimum match length 3: `'threshold': 1, 'similarity_name': 'jaccard', 'window': 5, 'ngram_length': 3, 'min_match_length': 3,`
- nothing for `PT`, dropped by the three-character minimum: `[]`
- myocardial infarction identified: `'term': 'myocardial infarction', 'cui': 'C0027051',`

Allow time for the first call, which loads the index. Later calls in the same process are fast.

## References

- Yim et al. 2023. Aci-bench: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation. Scientific Data 10:586. doi:10.1038/s41597-023-02487-3
- [`wyim/aci-bench`](https://github.com/wyim/aci-bench), [`SETUP.md`](https://github.com/wyim/aci-bench/blob/main/SETUP.md) and [`evaluation/UMLS_evaluation.py`](https://github.com/wyim/aci-bench/blob/main/evaluation/UMLS_evaluation.py).
- [`Georgetown-IR-Lab/QuickUMLS`](https://github.com/Georgetown-IR-Lab/QuickUMLS), the main package, the one to install.
- [`medspacy/QuickUMLS`](https://github.com/medspacy/QuickUMLS), a maintained fork. Matches text differently, so it is not a substitute here.
