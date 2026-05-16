"""Three-tier confidence policy for TR-Gold-Morph entries.

Prior `silver-auto` / `silver-agreed` naming was ambiguous and inconsistent
with the README's gold/silver/bronze claim. This module defines the canonical cuts.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    GOLD   = "gold"     # manually verified by linguist (IAA >= 0.85)
    SILVER = "silver"   # confidence >= 0.95 AND ensemble unanimous
    BRONZE = "bronze"   # 0.70 <= confidence < 0.95 OR ensemble-disagree w/ conf >= 0.80
    DROP   = "drop"     # below 0.70 — excluded from published dataset


@dataclass(frozen=True, slots=True)
class TierThresholds:
    silver_min_confidence: float = 0.95
    bronze_min_confidence: float = 0.70
    bronze_disagree_floor: float = 0.80


def assign_tier(
    confidence: float,
    *,
    ensemble_unanimous: bool,
    manually_verified: bool = False,
    th: TierThresholds = TierThresholds(),
) -> Tier:
    """Flat decision table — every (confidence-band, unanimity) cell is explicit.

                       unanimous=True       unanimous=False
    confidence ≥ 0.95: SILVER               BRONZE  (high conf but disagreement → bronze)
    0.80 ≤ c < 0.95 :  BRONZE               BRONZE
    0.70 ≤ c < 0.80 :  BRONZE               DROP    (low conf + disagreement → drop)
    confidence < 0.70: DROP                 DROP
    """
    if manually_verified:
        return Tier.GOLD
    if confidence >= th.silver_min_confidence:           # ≥ 0.95
        return Tier.SILVER if ensemble_unanimous else Tier.BRONZE
    if confidence >= th.bronze_disagree_floor:           # 0.80 ≤ c < 0.95
        return Tier.BRONZE
    if confidence >= th.bronze_min_confidence:           # 0.70 ≤ c < 0.80
        return Tier.BRONZE if ensemble_unanimous else Tier.DROP
    return Tier.DROP                                     # < 0.70
