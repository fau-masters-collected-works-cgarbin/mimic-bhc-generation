"""ConceptStore: a ConceptExtractor that reads the concept artifact instead of running one.

The artifact is a content-addressed text-to-concepts map: one JSONL file per
extractor fingerprint, one line per source text, keyed by that text's
``text_sha256``. This module imports no extraction dependency (scispaCy or
otherwise), so reading concept records back stays cheap.
"""

import json
from pathlib import Path

from bhc_experiment.concepts import Concept, concept_artifact_path, text_sha256


class ConceptStore:
    """A read-only ConceptExtractor backed by one concept artifact file.

    The constructor loads the artifact ``params`` fingerprints fully into memory.
    One file covers one concept space.
    """

    def __init__(self, concepts_dir: Path, params: dict) -> None:
        path = concept_artifact_path(concepts_dir, params)
        if not path.exists():
            msg = f"no concept artifact for fingerprint {path.stem}: expected {path}"
            raise FileNotFoundError(msg)

        self._by_hash: dict[str, list[Concept]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                self._by_hash[entry["text_sha256"]] = [
                    Concept(
                        cui=match["cui"],
                        term=match["term"],
                        semtypes=tuple(match["semtypes"]),
                        score=match["score"],
                        start=match["start"],
                        end=match["end"],
                    )
                    for match in entry["matches"]
                ]

    def extract_concepts(self, texts: list[str]) -> dict[str, list[Concept]]:
        """Look up each text's stored mentions by its sha256, mention order preserved.

        A text with no entry in the artifact raises ``KeyError`` rather than
        returning an empty list, which would undercount coverage. The error
        names only the text's hash and length, never its content, so patient
        text never reaches an exception message.
        """
        result: dict[str, list[Concept]] = {}
        for text in texts:
            digest = text_sha256(text)
            if digest not in self._by_hash:
                msg = f"no concept artifact entry for text_sha256={digest} (length={len(text)})"
                raise KeyError(msg)
            result[text] = self._by_hash[digest]
        return result

    def extract_cuis(self, texts: list[str]) -> dict[str, frozenset[str]]:
        """The CUI projection of ``extract_concepts``."""
        by_text = self.extract_concepts(texts)
        return {text: frozenset(c.cui for c in concepts) for text, concepts in by_text.items()}
