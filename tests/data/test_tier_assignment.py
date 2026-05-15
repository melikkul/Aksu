"""Boundary tests for the three-tier confidence policy (C-Step 3)."""
import pytest
from aksu.data.tiers import Tier, assign_tier


@pytest.mark.parametrize("conf,unan,expected", [
    # conf ≥ 0.95
    (1.00, True,  Tier.SILVER),
    (1.00, False, Tier.BRONZE),
    (0.95, True,  Tier.SILVER),
    (0.95, False, Tier.BRONZE),
    # 0.80 ≤ conf < 0.95
    (0.949, True,  Tier.BRONZE),
    (0.949, False, Tier.BRONZE),
    (0.80,  True,  Tier.BRONZE),
    (0.80,  False, Tier.BRONZE),
    # 0.70 ≤ conf < 0.80
    (0.799, True,  Tier.BRONZE),
    (0.799, False, Tier.DROP),
    (0.70,  True,  Tier.BRONZE),
    (0.70,  False, Tier.DROP),
    # conf < 0.70
    (0.699, True,  Tier.DROP),
    (0.699, False, Tier.DROP),
    (0.0,   True,  Tier.DROP),
    (0.0,   False, Tier.DROP),
])
def test_tier_boundaries(conf: float, unan: bool, expected: Tier) -> None:
    assert assign_tier(conf, ensemble_unanimous=unan) == expected


def test_manually_verified_overrides_all() -> None:
    assert assign_tier(0.0, ensemble_unanimous=False, manually_verified=True) == Tier.GOLD
    assert assign_tier(1.0, ensemble_unanimous=True, manually_verified=True) == Tier.GOLD


def test_tier_enum_values() -> None:
    assert Tier.GOLD.value == "gold"
    assert Tier.SILVER.value == "silver"
    assert Tier.BRONZE.value == "bronze"
    assert Tier.DROP.value == "drop"
