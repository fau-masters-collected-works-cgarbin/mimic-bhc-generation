"""QuickUMLS concept extraction, reached through a subprocess bridge.

QuickUMLS is the extractor the MEDCON reference implementation runs (Yim et
al. 2023). It cannot share this environment because it requires a specific
Python version.

This code uses ``scripts/quickumls_bridge.py`` to communicate with the
QuickUMLS library.

See docs/quickumls-setup.md.

Reports every candidate of every span group, matching Yim's ``get_matches``.
"""

import json
import subprocess
from pathlib import Path

from bhc_experiment.concepts import Concept, ExtractionResult
from mimic_data_prep.mimic import PROJECTS_DIR

_EXTRACTOR_NAME = "quickumls"

# Pinned in the sibling repo's pyproject.toml, and fingerprinted below. Every
# bridge run reports its own versions and raises on any disagreement.
EXPECTED_BRIDGE_VERSIONS = {
    "quickumls_version": "1.4.2",
    "spacy_version": "3.4.4",
    "spacy_model_version": "3.4.1",
}

# Created by docs/quickumls-setup.md and read here to save the installation configuration.
INSTALL_METADATA_FILENAME = "install.json"

# Where docs/quickumls-setup.md installs the QuickUMLS environment.
BRIDGE_REPO = PROJECTS_DIR / "phd-dissertation-quickumls"
BRIDGE_PYTHON = BRIDGE_REPO / ".venv" / "bin" / "python"
BRIDGE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "quickumls_bridge.py"

# Where docs/quickumls-setup.md builds the index.
UMLS_INDEX = PROJECTS_DIR / "umls" / "quickumls-2022AA"


def _run_bridge_command(command: str, requests: list[dict]) -> tuple[dict, list[dict]]:
    """Run one bridge command over the requests, returning its handshake and responses."""
    stdin = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests)
    result = subprocess.run(  # noqa: S603
        [str(BRIDGE_PYTHON), str(BRIDGE_SCRIPT), command, str(BRIDGE_REPO)],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        # The bridge never writes note text to stderr, so the message is safe
        # to include.
        msg = f"{BRIDGE_SCRIPT.name} exited {result.returncode}: {result.stderr.strip()}"
        raise RuntimeError(msg)

    handshake, *responses = (
        json.loads(line) for line in result.stdout.splitlines() if line.strip()
    )
    _verify_bridge_versions(handshake["versions"])
    return handshake, list(responses)


def run_extract(texts: list[str]) -> tuple[dict, dict[str, list]]:
    """Match every text, returning the matcher's settings and its span groups by text."""
    requests = [{"id": i, "text": t} for i, t in enumerate(texts)]
    handshake, responses = _run_bridge_command("extract", requests)
    return handshake["settings"], {texts[r["id"]]: r["spans"] for r in responses}


def run_medcon(pairs: list[tuple[str, str]]) -> list[float]:
    """Score each (reference, prediction) pair with Yim's scorer, returning F1.

    His ``umls_score_individual`` computes precision and recall but returns
    only F1.
    """
    requests = [
        {"id": i, "reference": ref, "prediction": pred} for i, (ref, pred) in enumerate(pairs)
    ]
    _, responses = _run_bridge_command("medcon", requests)
    by_id = {r["id"]: r["f1"] for r in responses}
    return [by_id[i] for i in range(len(pairs))]


class QuickUmlsExtractor:
    """Extraction over a QuickUMLS index, satisfying ``EntryExtractor``."""

    def __init__(self) -> None:
        # Entries are cached to save the cost of re-extracting them.
        self._cache: dict[str, ExtractionResult] = {}

    def extract_concepts(self, texts: list[str]) -> dict[str, list[Concept]]:
        """Return per-text linked UMLS concepts, one per surviving candidate."""
        return {t: entry.matches for t, entry in self._get_entries(texts).items()}

    def extract_cuis(self, texts: list[str]) -> dict[str, frozenset[str]]:
        """Return per-text CUI sets, the input to set-overlap metrics."""
        entries = self._get_entries(texts)
        return {t: frozenset(c.cui for c in entry.matches) for t, entry in entries.items()}

    def extract_entries(self, texts: list[str]) -> dict[str, ExtractionResult]:
        """Return the full per-text record: matches, rejections, and counts.

        ``n_entities`` is the number of text spans QuickUMLS matched. A span can
        have several CUIs, so there are usually more matches than spans.

        QuickUMLS does not report dropped candidates, so ``rejected`` is always
        empty and ``n_unlinked`` always zero.
        """
        return self._get_entries(texts)

    def _get_entries(self, texts: list[str]) -> dict[str, ExtractionResult]:
        """Every text's extraction record, matching only the ones not yet cached."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            _settings, spans_by_text = run_extract(missing)
            for text, span_groups in spans_by_text.items():
                self._cache[text] = _result_from_span_groups(span_groups)
        return {t: self._cache[t] for t in texts}


def _result_from_span_groups(span_groups: list[list[dict]]) -> ExtractionResult:
    """One text's extraction record, from the bridge's nested span groups.

    A span group is one character interval, with every CUI QuickUMLS
    matched there. All of them become concepts.
    """
    matches = []
    for group in span_groups:
        for candidate in group:
            matches.append(
                Concept(
                    cui=candidate["cui"],
                    term=candidate["term"],
                    semtypes=tuple(candidate["semtypes"]),
                    score=candidate["similarity"],
                    start=candidate["start"],
                    end=candidate["end"],
                )
            )
    return ExtractionResult(matches=matches, rejected=[], n_entities=len(span_groups), n_unlinked=0)


def _verify_bridge_versions(reported: dict) -> None:
    mismatched = sorted(k for k, v in EXPECTED_BRIDGE_VERSIONS.items() if reported.get(k) != v)
    if mismatched:
        msg = f"mismatched keys: {mismatched}. Expected {EXPECTED_BRIDGE_VERSIONS}, got {reported}."
        raise RuntimeError(msg)


def params_from_settings(index: str | Path = UMLS_INDEX) -> dict:
    """The fingerprinted parameters, read from the matcher the bridge runs.

    The matcher settings are read from the running matcher. The index
    contributes ``install.json``'s UMLS release and source-file digests.
    """
    install = json.loads((Path(index) / INSTALL_METADATA_FILENAME).read_text())
    settings, _spans = run_extract([])
    return {
        "extractor": _EXTRACTOR_NAME,
        "extractor_params": {
            **EXPECTED_BRIDGE_VERSIONS,
            "umls_release": install["umls_release"],
            "mrconso_sha256": install["mrconso_sha256"],
            "mrsty_sha256": install["mrsty_sha256"],
            **settings,
        },
    }
