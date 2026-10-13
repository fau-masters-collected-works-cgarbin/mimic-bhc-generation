"""Expose QuickUMLS to this repo's Python over a JSONL pipe.

Runs on the sibling QuickUMLS environment's interpreter, which alone can import
``quickumls``. Keep it Python 3.10 compatible and standard-library only.
See docs/quickumls-setup.md.

    <quickumls repo>/.venv/bin/python scripts/quickumls_bridge.py CMD <quickumls repo>

Each command reads JSON objects on stdin and writes JSON objects on stdout, one
per line, the first line out being a handshake.

    CMD = `extract`  {"id","text"} in, QuickUMLS's span groups out
    CMD = `medcon`   {"id","reference","prediction"} in, one MEDCON F1 out
"""

import importlib.metadata
import json
import sys
from pathlib import Path

# The QuickUMLS match fields that pass through unchanged.
# The input text itself is left out, so the artifact has no note text.
_MATCH_FIELDS = ("start", "end", "term", "cui", "similarity")


def _add_aci_bench_to_path(quickumls_repo: str) -> None:
    """Make Yim's modules importable, from the layout they hardcode.

    ``semantics.py`` and ``UMLS_evaluation.py`` both resolve their resources
    from their own ``__file__``, so they only work from inside ``evaluation/``.
    """
    sys.path.insert(0, str(Path(quickumls_repo) / "aci_bench" / "evaluation"))


def _versions() -> dict:
    """The installed versions that decide what this bridge computes.

    spaCy is here because QuickUMLS calls it inside ``match()``, and since
    ``ignore_syntax`` became inert in 1.4.1 its part-of-speech and sentence
    output decides which spans exist at all.
    """
    return {
        "quickumls_version": importlib.metadata.version("quickumls"),
        "spacy_version": importlib.metadata.version("spacy"),
        "spacy_model_version": importlib.metadata.version("en_core_web_sm"),
    }


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _requests():
    for line in sys.stdin:
        if line.strip():
            yield json.loads(line)


def _settings(matcher) -> dict:
    """Report the matcher's own description of itself.

    Sets are sorted into lists so the same matcher always serializes the same
    way, and so JSON can encode them at all.
    """
    return {
        key: sorted(value) if isinstance(value, (set, frozenset, list)) else value
        for key, value in matcher.get_info().items()
    }


def _match_record(match: dict) -> dict:
    """One QuickUMLS candidate as a JSON-encodable dict.

    ``semtypes`` arrives as a set, which JSON cannot encode, so it becomes a
    sorted list.
    """
    record = {field: match[field] for field in _MATCH_FIELDS}
    record["semtypes"] = sorted(match["semtypes"])
    return record


def extract() -> None:
    """Match every text with the reference implementation's own matcher.

    Uses ``UMLS_evaluation.py``'s module-level ``matcher``, built there at
    Yim's settings, and passes ``ignore_syntax=True`` as his ``get_matches``
    does. The handshake reports what that matcher says it is.

    Output preserves QuickUMLS's nesting: ``spans`` has one list per
    character interval, each with that interval's candidates, and the
    character offsets kept.
    """
    from UMLS_evaluation import matcher

    _emit({"versions": _versions(), "settings": _settings(matcher)})

    for request in _requests():
        groups = matcher.match(request["text"], ignore_syntax=True)
        spans = []
        for group in groups:
            spans.append([_match_record(match) for match in group])
        _emit({"id": request["id"], "spans": spans})


def medcon() -> None:
    """Score each pair with Yim's ``umls_score_individual``, unmodified.

    ``UMLS_evaluation.py`` builds its own matcher at module scope, so the
    settings and the ``resources/des`` index path both come from his file.
    """
    from UMLS_evaluation import umls_score_individual

    _emit({"versions": _versions()})
    for request in _requests():
        f1 = umls_score_individual(request["reference"], request["prediction"])
        _emit({"id": request["id"], "f1": f1})


COMMANDS = {"extract": extract, "medcon": medcon}
_EXPECTED_ARGC = 3  # script, command, repo path


def main() -> int:
    if len(sys.argv) != _EXPECTED_ARGC or sys.argv[1] not in COMMANDS:
        sys.stderr.write(f"usage: {sys.argv[0]} {{{'|'.join(COMMANDS)}}} <quickumls repo>\n")
        return 2
    _add_aci_bench_to_path(sys.argv[2])
    COMMANDS[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
