"""UMLS concept extraction for clinical text.

Wraps scispaCy's UMLS entity linker into one class with extract methods.

Importing this module does not import spaCy, which takes seconds to load.
``UmlsExtractor`` imports it on construction.
"""

import importlib.metadata
import logging

from bhc_experiment.concepts import Concept, ExtractionResult, Rejection

logger = logging.getLogger(__name__)

# Batch size for the scispaCy nlp.pipe call. Fixed at the scispaCy default.
_PIPE_BATCH_SIZE = 16

# Identifies this extractor in the fingerprint.
_EXTRACTOR_NAME = "scispacy"

# Hand-bumped when _extract_doc_result's candidate logic changes structure
# without changing a listed parameter (e.g. adding a fallback candidate),
# so a structural change fingerprints itself. git_commit is provenance only
# and never enters the identity.
SCISPACY_EXTRACTOR_VERSION = 1

# Hardcoded in the scispaCy linker config. Defined once here so the fingerprint
# tracks the real value if it ever changes.
#
# True but inert: the linker only resolves abbreviations when the pipeline has an
# AbbreviationDetector, and UmlsExtractor.__init__ adds only the linker itself.
_RESOLVE_ABBREVIATIONS = True

# UmlsExtractor.__init__'s defaults.
DEFAULT_SPACY_MODEL = "en_core_sci_md"
DEFAULT_LINKER_NAME = "umls"
DEFAULT_THRESHOLD = 0.7  # scispaCy EntityLinker default

# Default TUI set from MEDCON (Yim et al. 2023, ACI-Bench). Not a hand-picked
# list of types: it is the union of seven of the fifteen UMLS semantic groups,
# in SemGroups file order. Anatomy, Chemicals & Drugs, Devices, Disorders,
# Genes & Molecular Sequences, Phenomena, Physiology. Restricting extraction to
# these TUIs is what separates "clinical concept overlap" from "any-entity
# overlap".
#
# Procedures is one of the eight excluded groups, so a lab test name
# ("Hematocrit", T059) is invisible while its result ("Laboratory or Test
# Result", T034) is not. Add or drop whole groups, never single types, or the
# set stops matching the published metric.
MEDCON_DEFAULT_TUIS: tuple[str, ...] = (
    "T017",
    "T029",
    "T023",
    "T030",
    "T031",
    "T022",
    "T025",
    "T026",
    "T018",
    "T021",
    "T024",
    "T116",
    "T195",
    "T123",
    "T122",
    "T103",
    "T120",
    "T104",
    "T200",
    "T196",
    "T126",
    "T131",
    "T125",
    "T129",
    "T130",
    "T197",
    "T114",
    "T109",
    "T121",
    "T192",
    "T127",
    "T203",
    "T074",
    "T075",
    "T020",
    "T190",
    "T049",
    "T019",
    "T047",
    "T050",
    "T033",
    "T037",
    "T048",
    "T191",
    "T046",
    "T184",
    "T087",
    "T088",
    "T028",
    "T085",
    "T086",
    "T038",
    "T069",
    "T068",
    "T034",
    "T070",
    "T067",
    "T043",
    "T201",
    "T045",
    "T041",
    "T044",
    "T032",
    "T040",
    "T042",
    "T039",
)

# CUIs that scispaCy links but that are not real concepts in this corpus, so they
# are dropped from every extraction. scispaCy's linker matches on string similarity
# alone, however short or common the text, so it links everyday clinical
# abbreviations and words to UMLS concepts that happen to have a semantic type
# the metric counts. Each entry is a high-frequency mislink confirmed by tracing
# the matched text across the reference BHCs: the string on the left almost always
# means something else in a hospital course.
# Dropping these keeps the MEDCON reference set and the oracle ceiling from
# crediting phantom concepts.
# To re-derive the candidates behind this list, for example after a cohort change,
# run scripts/decision_analysis/medcon_linking_misfires.py.
LINKING_ARTIFACT_CUIS: frozenset[str] = frozenset(
    {
        "C0032207",  # "Pt"/"PT" (patient / prothrombin time) -> platinum, the element
        "C0168634",  # "baseline" -> a dental cement product named "BaseLine"
        "C0242350",  # "ED" (emergency department) -> erectile dysfunction
        "C0030855",  # "PCP" (primary care provider / pneumocystis) -> pentachlorophenol
        "C0059036",  # "EF" (ejection fraction) -> peptide elongation factor 1
        "C0012621",  # "discharge[d]" (hospital discharge) -> body fluid discharge
        "C0181904",  # "monitored" (observed) -> a monitor device
    }
)


class UmlsExtractor:
    """Linked UMLS concept extractor backed by scispaCy.

    Construction loads the spaCy pipeline and UMLS entity linker. This
    is expensive (the linker bundle is ~1 GB), so a single instance is
    meant to be reused across many extract calls inside one process.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        spacy_model: str = DEFAULT_SPACY_MODEL,
        linker_name: str = DEFAULT_LINKER_NAME,
        tuis: tuple[str, ...] | list[str] = MEDCON_DEFAULT_TUIS,
        extra_tuis: tuple[str, ...] | list[str] = (),
        threshold: float = DEFAULT_THRESHOLD,
        n_process: int = 1,
    ) -> None:
        import spacy  # noqa: PLC0415
        from scispacy.linking import EntityLinker  # noqa: F401, PLC0415  # registers the pipe

        logger.info("Loading scispaCy model %s with %s linker...", spacy_model, linker_name)
        self._nlp = spacy.load(spacy_model)
        self._nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": _RESOLVE_ABBREVIATIONS,
                "linker_name": linker_name,
                "threshold": threshold,
            },
        )
        # Fast membership check for the TUI filter in the extraction loop.
        # extra_tuis adds to tuis, so a wider filter names only what it adds
        # and still follows the MEDCON default.
        self._tui_set = frozenset(tuis) | frozenset(extra_tuis)
        self._linker = self._nlp.get_pipe("scispacy_linker")
        self._n_process = n_process
        self._cache: dict[str, ExtractionResult] = {}
        self._spacy_model = spacy_model
        self._linker_name = linker_name
        self._threshold = threshold

    def extract_concepts(self, texts: list[str]) -> dict[str, list[Concept]]:
        """Return per-text linked UMLS concepts.

        Every input text gets an entry, even one with no concepts found.
        """
        return {t: entry.matches for t, entry in self._get_entries(texts).items()}

    def extract_cuis(self, texts: list[str]) -> dict[str, frozenset[str]]:
        """Return per-text CUI sets.

        Mention order and repeated mentions within a text are dropped.
        """
        entries = self._get_entries(texts)
        return {t: frozenset(c.cui for c in entry.matches) for t, entry in entries.items()}

    def extract_entries(self, texts: list[str]) -> dict[str, ExtractionResult]:
        """Return the full per-text extraction record: matches, rejections, and counts.

        Use this instead of ``extract_concepts`` to see why a concept is
        absent: blocklisted, out of filter, or unlinked.
        """
        return self._get_entries(texts)

    def semantic_types(self, cui: str) -> frozenset[str]:
        """Every UMLS semantic type the knowledge base gives this CUI.

        Takes a bare CUI, for when there is no Concept in hand. An unknown
        CUI returns an empty set.
        """
        entity = self._linker.kb.cui_to_entity.get(cui)
        return frozenset(entity.types) if entity else frozenset()

    def _get_entries(self, texts: list[str]) -> dict[str, ExtractionResult]:
        """Every text's extraction record, extracting only the ones not yet cached."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for text, doc in zip(
                missing,
                self._nlp.pipe(missing, batch_size=_PIPE_BATCH_SIZE, n_process=self._n_process),
                strict=True,
            ):
                self._cache[text] = self._extract_doc_result(doc)
        return {t: self._cache[t] for t in texts}

    def _extract_doc_result(self, doc) -> ExtractionResult:  # noqa: ANN001
        # Take the entity's top linker candidate, then drop the mention when that
        # candidate's UMLS semantic types miss the TUI filter. Falling back to a
        # lower-ranked in-filter candidate turns "patient" into a body substance.
        matches: list[Concept] = []
        rejected: list[Rejection] = []
        n_unlinked = 0
        for ent in doc.ents:
            if not ent._.kb_ents:
                n_unlinked += 1
                continue
            cui, score = ent._.kb_ents[0]
            if cui in LINKING_ARTIFACT_CUIS:
                rejected.append(
                    Rejection(
                        cui=cui,
                        score=score,
                        reason="blocklisted",
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )
                continue
            entity = self._linker.kb.cui_to_entity.get(cui)
            if not entity:
                n_unlinked += 1
                continue
            matching = self._tui_set.intersection(entity.types)
            if not matching:
                rejected.append(
                    Rejection(
                        cui=cui,
                        score=score,
                        reason="out_of_filter",
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )
                continue
            matches.append(
                Concept(
                    cui=cui,
                    term=entity.canonical_name,
                    semtypes=tuple(sorted(entity.types)),
                    score=score,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )
        return ExtractionResult(
            matches=matches, rejected=rejected, n_entities=len(doc.ents), n_unlinked=n_unlinked
        )


def params_from_settings(
    spacy_model: str,
    linker_name: str,
    threshold: float,
    tuis: tuple[str, ...] | list[str] | frozenset[str],
    *,
    versions: dict[str, str] | None = None,
) -> dict:
    """The fingerprinted parameters for a given extractor configuration.

    Covers only what changes extraction output, so a machine's throughput
    settings do not invalidate the artifact.

    When ``versions`` is omitted, package versions are looked up here via
    ``importlib.metadata``.
    """
    if versions is None:
        versions = {
            spacy_model: importlib.metadata.version(spacy_model),
            "scispacy": importlib.metadata.version("scispacy"),
            "spacy": importlib.metadata.version("spacy"),
        }
    return {
        "extractor": _EXTRACTOR_NAME,
        "extractor_version": SCISPACY_EXTRACTOR_VERSION,
        "extractor_params": {
            "spacy_model": spacy_model,
            "spacy_model_version": versions[spacy_model],
            "scispacy_version": versions["scispacy"],
            "spacy_version": versions["spacy"],
            "linker_name": linker_name,
            "threshold": threshold,
            "resolve_abbreviations": _RESOLVE_ABBREVIATIONS,
            "tuis": sorted(tuis),
            "linking_artifact_cuis": sorted(LINKING_ARTIFACT_CUIS),
        },
    }


def params_from_extractor(
    extractor: UmlsExtractor, *, versions: dict[str, str] | None = None
) -> dict:
    """The fingerprinted parameters read off a constructed extractor."""
    return params_from_settings(
        extractor._spacy_model,  # noqa: SLF001
        extractor._linker_name,  # noqa: SLF001
        extractor._threshold,  # noqa: SLF001
        extractor._tui_set,  # noqa: SLF001
        versions=versions,
    )
