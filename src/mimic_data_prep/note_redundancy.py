"""Whole-note redundancy detection for MIMIC-III admissions.

MIMIC-III notes have two systematic redundancy patterns that the BHC
deduplication ablations may want to remove:

1. **Copy-forward duplicates.** Nursing and physician notes often
   reuse verbatim text from a prior shift (vital signs, medication
   lists, assessment paragraphs). Detected by Jaccard
   similarity over k-word shingles (overlapping windows of k
   consecutive words from the lowercased text) between every pair of
   notes within an admission. Pairs above the threshold form
   single-link transitive clusters: if A and B are near-duplicates
   and B and C are near-duplicates, A/B/C form one cluster of three.
   The "reduction" step keeps the most recent note in each cluster
   and treats the rest as redundant.

2. **PFI/final radiology pairs.** Radiologists file a preliminary
   findings impression (PFI) as one NOTEEVENTS row, then later file
   the final report as a separate row. The final quotes the PFI text
   verbatim and adds the full interpretation. Their Jaccard is too
   low to flag as near-duplicate (the final has 2-5x the text), but
   the PFI adds no information beyond what the final contains.
   Detected by text markers: ``FINAL REPORT`` for
   the final, and any of ``PFI REPORT`` / ``WET READ`` /
   ``PROVISIONAL FINDINGS`` for the standalone PFI.

Edge case: when a PFI and its final report are also Jaccard
near-duplicates (rare, because the final usually adds substantial
interpretation text that pushes Jaccard below the threshold), the
cluster pass discards one of them as the older cluster member. The
surviving note has no partner left, so ``pfi_final_pairs`` under-counts
that pair by one. ``discarded_indices`` stays correct because exactly
one of the two notes is discarded in either path.

Pinned parameters live in ``NOTE_REDUNDANCY_CONFIG``.
"""

from collections.abc import Callable

NOTE_REDUNDANCY_CONFIG: dict = {
    "shingle_size": 7,
    "jaccard_threshold": 0.95,
    "text_normalization": "text.lower().split()",
    # Marker matching is case-insensitive: markers and note text are lowercased.
    "pfi_markers": ["PFI REPORT", "WET READ", "PROVISIONAL FINDINGS"],
    "final_marker": "FINAL REPORT",
    "pfi_pair_key": ["HADM_ID", "CHARTDATE", "DESCRIPTION"],
    "calibration_reference": "Gabriel et al. 2018",
}

_SHINGLE_SIZE: int = NOTE_REDUNDANCY_CONFIG["shingle_size"]
_JACCARD_THRESHOLD: float = NOTE_REDUNDANCY_CONFIG["jaccard_threshold"]
_PFI_MARKERS: tuple[str, ...] = tuple(m.lower() for m in NOTE_REDUNDANCY_CONFIG["pfi_markers"])
_FINAL_MARKER: str = NOTE_REDUNDANCY_CONFIG["final_marker"].lower()


def _shingle_set(text: str, k: int) -> frozenset[int]:
    """Return the set of hashed k-word shingles for text.

    Returns an empty frozenset for notes shorter than k words, which can
    never match another note.
    """
    words = text.lower().split()
    if len(words) < k:
        return frozenset()
    shingles: set[int] = set()
    for i in range(len(words) - k + 1):
        window = " ".join(words[i : i + k])
        shingles.add(hash(window))
    return frozenset(shingles)


def _make_recency_key(notes: list[dict]) -> Callable[[int], str]:
    """Build a sort key that orders note indices by recency.

    Prefers CHARTTIME, falls back to CHARTDATE, falls back to "" so notes
    without either timestamp sort first (oldest position). MIMIC-III uses
    ISO-style date and datetime strings, which sort lexicographically in
    chronological order.
    """
    return lambda idx: notes[idx].get("charttime") or notes[idx].get("chartdate") or ""


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    return intersection / union


def _classify_radiology(text_lower: str) -> str | None:
    """Return 'final', 'pfi', or None for a radiology note's lowercased text.

    FINAL REPORT wins over any embedded PFI markers, since it makes the
    document type final regardless of the PFI text it quotes.

    The marker set and the "FINAL REPORT trumps PFI" rule come from the
    empirical analysis in ``scripts/decision_analysis/radiology_pfi_final_detection.py``.
    """
    if _FINAL_MARKER in text_lower:
        return "final"
    if any(m in text_lower for m in _PFI_MARKERS):
        return "pfi"
    return None


def _find_near_duplicate_pairs(shingles: list[frozenset[int]]) -> list[tuple[int, int]]:
    """All (i, j) pairs whose Jaccard is at or above the pinned threshold.

    An admission has about 38 notes at the median, so the pairwise loop is
    fast enough.
    """
    n = len(shingles)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(shingles[i], shingles[j]) >= _JACCARD_THRESHOLD:
                pairs.append((i, j))
    return pairs


def _build_clusters(n: int, pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Single-link transitive clustering via union-find with path compression."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        clusters.setdefault(find(idx), []).append(idx)
    return clusters


def _detect_pfi_final(
    notes: list[dict], texts: list[str], discarded: set[int]
) -> tuple[list[dict], set[int]]:
    """PFI/final radiology pairs among the cluster-pass survivors.

    A standalone PFI is dropped when a final report exists on the same
    (CHARTDATE, DESCRIPTION). Pair count per key is ``min(n_pfi, n_final)``,
    and the oldest PFIs are the ones dropped, symmetric with the cluster
    pass keeping the most recent note. Notes without a marker are skipped, since
    guessing risks discarding a real note. Returns the pair records (dropped PFI
    paired with the kept, most-recent final) and the set of dropped PFI indices.
    """
    pfi_indices_by_key: dict[tuple, list[int]] = {}
    final_indices_by_key: dict[tuple, list[int]] = {}
    for idx, note in enumerate(notes):
        if idx in discarded or note["category"] != "Radiology":
            continue
        kind = _classify_radiology(texts[idx].lower())
        if kind is None:
            continue
        key = (note["chartdate"], note["description"])
        bucket = final_indices_by_key if kind == "final" else pfi_indices_by_key
        bucket.setdefault(key, []).append(idx)

    recency = _make_recency_key(notes)
    pairs: list[dict] = []
    dropped: set[int] = set()
    for key, pfi_indices in pfi_indices_by_key.items():
        finals = final_indices_by_key.get(key, [])
        n_pair = min(len(pfi_indices), len(finals))
        if n_pair == 0:
            continue
        kept_final = max(finals, key=recency)
        chartdate, description = key
        for idx in sorted(pfi_indices, key=recency)[:n_pair]:
            dropped.add(idx)
            pairs.append(
                {
                    "pfi_index": idx,
                    "final_index": kept_final,
                    "chartdate": chartdate,
                    "description": description,
                }
            )
    return pairs, dropped


def detect_note_redundancy(notes: list[dict]) -> dict:
    """Find the duplicated notes in one admission and pick which copy to keep.

    Returns the annotated detection structure. Keys:

    - ``near_duplicate_pairs``: note-index pairs at or above the Jaccard
      threshold.
    - ``clusters``: one entry per near-duplicate cluster (size > 1) with its
      ``members``, the ``keep`` index (most recent), and ``drop`` entries
      (each ``index`` plus its ``jaccard`` against the kept note).
    - ``in_cluster_count``: notes participating in any cluster.
    - ``pfi_final_pairs``: dropped PFI paired with the kept final, computed
      on the cluster-pass survivors.
    - ``discarded_indices``: union of cluster drops and PFI drops, the notes
      the combined rule removes.

    Every index is a position in ``notes``. Notes that tie on recency keep
    the one that comes first in ``notes``.
    """
    texts = [note["text"] or "" for note in notes]
    shingles = [_shingle_set(t, _SHINGLE_SIZE) for t in texts]
    pairs = _find_near_duplicate_pairs(shingles)
    clusters_map = _build_clusters(len(notes), pairs)
    recency = _make_recency_key(notes)

    clusters: list[dict] = []
    cluster_discarded: set[int] = set()
    in_cluster_count = 0
    for members in clusters_map.values():
        if len(members) <= 1:
            continue
        in_cluster_count += len(members)
        ranked = sorted(members, key=recency, reverse=True)
        keep = ranked[0]
        drop = []
        for idx in ranked[1:]:
            similarity = round(_jaccard(shingles[idx], shingles[keep]), 4)
            drop.append({"index": idx, "jaccard": similarity})
        cluster_discarded.update(d["index"] for d in drop)
        clusters.append({"members": ranked, "keep": keep, "drop": drop})

    pfi_final_pairs, pfi_discarded = _detect_pfi_final(notes, texts, cluster_discarded)
    return {
        "near_duplicate_pairs": pairs,
        "clusters": clusters,
        "in_cluster_count": in_cluster_count,
        "pfi_final_pairs": pfi_final_pairs,
        "discarded_indices": cluster_discarded | pfi_discarded,
    }
