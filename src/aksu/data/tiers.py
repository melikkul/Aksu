"""Tier confidence policy for TR-Gold-Morph entries.

Exports two assignment functions:
  assign_tier_v1  — original 3-tier (gold/silver/bronze) for v1 backward compat
  assign_tier_v2  — 5-tier (gold/silver-confident/silver-auto/silver-marginal/bronze/drop)
                    used for v2 build (reads autolabel.py output fields)

The v2 Tier enum values are a superset of v1:
  GOLD / SILVER_CONFIDENT / SILVER_AUTO / SILVER_MARGINAL / BRONZE / DROP
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    GOLD             = "gold"
    SILVER_CONFIDENT = "silver-confident"  # multi-candidate, full ensemble agreement
    SILVER_AUTO      = "silver-auto"       # Zeyrek unambiguous (1 candidate)
    SILVER_MARGINAL  = "silver-marginal"   # 4/5 seeds agreement
    BRONZE           = "bronze"            # 3/5 seeds agreement — research use only
    DROP             = "drop"              # excluded from published dataset

    # v1 compat aliases (same string values as v1)
    SILVER  = "silver"   # v1 silver (= silver-confident in v2)
    # bronze already matches


@dataclass(frozen=True, slots=True)
class TierThresholds:
    silver_min_confidence: float = 0.95   # v1 compat
    bronze_min_confidence: float = 0.70
    bronze_disagree_floor: float = 0.80


# ---------------------------------------------------------------------------
# v1 — original 3-tier policy (preserved for backward compat)
# ---------------------------------------------------------------------------

def assign_tier(
    confidence: float,
    *,
    ensemble_unanimous: bool,
    manually_verified: bool = False,
    th: TierThresholds = TierThresholds(),
) -> Tier:
    """Original v1 3-tier assignment.

    Decision table:
                           unanimous=True       unanimous=False
    confidence ≥ 0.95:     SILVER               BRONZE
    0.80 ≤ c < 0.95:       BRONZE               BRONZE
    0.70 ≤ c < 0.80:       BRONZE               DROP
    confidence < 0.70:     DROP                 DROP
    """
    if manually_verified:
        return Tier.GOLD
    if confidence >= th.silver_min_confidence:
        return Tier.SILVER if ensemble_unanimous else Tier.BRONZE
    if confidence >= th.bronze_disagree_floor:
        return Tier.BRONZE
    if confidence >= th.bronze_min_confidence:
        return Tier.BRONZE if ensemble_unanimous else Tier.DROP
    return Tier.DROP


# ---------------------------------------------------------------------------
# v2 — 5-tier policy (uses autolabel.py output fields)
# ---------------------------------------------------------------------------

def assign_tier_v2(
    candidate_count: int,
    seed_agreement: int,
    disambig_score: float,
    method: str,
    *,
    manually_verified: bool = False,
    min_seeds_for_silver_confident: int = 5,
    min_seeds_for_silver_marginal: int = 4,
    min_seeds_for_bronze: int = 3,
    silver_confident_score: float = 0.85,
    silver_marginal_score: float = 0.70,
) -> Tier:
    """v2 5-tier assignment from autolabel fields.

    Tier decision table:
      manually_verified            → GOLD
      method == unambiguous        → SILVER_AUTO  (candidate_count == 1)
      seed_agreement ≥ 5
        AND disambig_score ≥ 0.85  → SILVER_CONFIDENT
      seed_agreement ≥ 4
        AND disambig_score ≥ 0.70  → SILVER_MARGINAL
      seed_agreement ≥ 3           → BRONZE
      everything else              → DROP
    """
    if manually_verified:
        return Tier.GOLD

    if method in ("zeyrek_oov", "zeyrek_empty", "drop_low_seeds"):
        return Tier.DROP

    if method == "unambiguous" or candidate_count == 1:
        return Tier.SILVER_AUTO

    # Multi-candidate — check ensemble agreement
    if (
        seed_agreement >= min_seeds_for_silver_confident
        and disambig_score >= silver_confident_score
    ):
        return Tier.SILVER_CONFIDENT

    if (
        seed_agreement >= min_seeds_for_silver_marginal
        and disambig_score >= silver_marginal_score
    ):
        return Tier.SILVER_MARGINAL

    if seed_agreement >= min_seeds_for_bronze:
        return Tier.BRONZE

    return Tier.DROP


# ---------------------------------------------------------------------------
# v1-compat collapse
# ---------------------------------------------------------------------------

def v2_tier_to_v1_compat(tier: Tier) -> str | None:
    """Map v2 tier to v1-compat tier name. Returns None for excluded tiers."""
    mapping: dict[Tier, str | None] = {
        Tier.GOLD:             "gold",
        Tier.SILVER_CONFIDENT: "silver-auto",
        Tier.SILVER_AUTO:      "silver-auto",
        Tier.SILVER_MARGINAL:  "silver-agreed",
        Tier.BRONZE:           None,  # excluded from v1-compat
        Tier.DROP:             None,
        Tier.SILVER:           "silver-auto",  # legacy alias
    }
    return mapping.get(tier)
