"""Two definitions of Exact Match for Turkish morphological systems."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def em_argmax(pred_indices: Sequence[int], gold_indices: Sequence[int]) -> float:
    """Candidate-index argmax accuracy (legacy disambiguator EM).

    Use ONLY for within-system reporting; NOT cross-system comparable.
    """
    if not gold_indices:
        return 0.0
    return sum(p == g for p, g in zip(pred_indices, gold_indices, strict=False)) / len(gold_indices)


def em_string(pred_parses: Sequence[str], gold_parses: Sequence[str]) -> float:
    """Full canonical-string parse equality. Cross-system comparable.

    Build pred_parses by indexing candidate_strings[i][pred_indices[i]] —
    do NOT re-run the analyzer to reconstruct.
    """
    if not gold_parses:
        return 0.0
    return sum(p == g for p, g in zip(pred_parses, gold_parses, strict=False)) / len(gold_parses)


def pred_index_to_strings(
    pred_indices: Sequence[int],
    candidate_strings: Sequence[Sequence[str]],
) -> list[str]:
    """Map per-sample argmax index → predicted canonical string.

    Returns empty string for out-of-range indices (signals a model bug; surfaces
    without crashing).
    """
    out: list[str] = []
    for idx, cands in zip(pred_indices, candidate_strings, strict=False):
        out.append(cands[idx] if 0 <= idx < len(cands) else "")
    return out
