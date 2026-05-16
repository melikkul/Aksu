"""Unit tests for BERTurk ensemble wiring in autolabel.py (§H+ A).

Tests tie-breaking logic, NaN/error-seed fallback, and <3 usable seeds → drop.
These tests mock the disambiguator to avoid loading actual model weights.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers — build fake model that returns preset logits
# ---------------------------------------------------------------------------

def _make_model(logits: list[list[float]]) -> MagicMock:
    """Return a mock BERTurkDisambiguator with preset per-call logits."""
    model = MagicMock()
    # Each call to model() yields one row of logits (one seed's vote)
    return_vals = [
        (torch.tensor([row], dtype=torch.float32), None)
        for row in logits
    ]
    model.side_effect = return_vals
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return model


# ---------------------------------------------------------------------------
# Import the function under test
# ---------------------------------------------------------------------------

from aksu.data.build.autolabel import _ensemble_score


# ---------------------------------------------------------------------------
# Tests: majority vote (no tie)
# ---------------------------------------------------------------------------

def test_clear_majority_vote() -> None:
    """3/5 seeds vote for candidate 1 → candidate 1 wins."""
    # 5 seeds, 3 candidates each.  Seed 0,1,2 argmax=1, seeds 3,4 argmax=0
    logits_per_seed = [
        [0.1, 0.8, 0.1],   # seed 0: argmax=1
        [0.2, 0.7, 0.1],   # seed 1: argmax=1
        [0.3, 0.6, 0.1],   # seed 2: argmax=1
        [0.9, 0.05, 0.05], # seed 3: argmax=0
        [0.8, 0.1, 0.1],   # seed 4: argmax=0
    ]
    models = [_make_model([row]) for row in logits_per_seed]
    candidates = ["kedi +Noun", "ev +Noun", "git +Verb"]
    token2idx: dict[str, int] = {}

    best_idx, disambig_score, seed_agreement, seed_votes = _ensemble_score(
        models, candidates, token2idx, context_sentence="test cümle", target_position=0,
    )

    assert best_idx == 1
    assert seed_agreement == 3
    assert len(seed_votes) == 5


def test_unanimous_vote() -> None:
    """All 5 seeds agree → seed_agreement=5."""
    logits_per_seed = [[0.1, 0.85, 0.05]] * 5
    models = [_make_model([row]) for row in logits_per_seed]
    candidates = ["kedi +Noun", "ev +Noun", "git +Verb"]
    token2idx: dict[str, int] = {}

    best_idx, disambig_score, seed_agreement, seed_votes = _ensemble_score(
        models, candidates, token2idx, context_sentence="test", target_position=0,
    )

    assert seed_agreement == 5
    assert best_idx == 1


# ---------------------------------------------------------------------------
# Tests: tie-breaking (§H+ A)
# ---------------------------------------------------------------------------

def test_tie_broken_by_average_softmax() -> None:
    """2-2-1 tie: average softmax decides. Candidate with higher avg softmax wins."""
    # Seeds 0,1 vote for cand 0; seeds 2,3 vote for cand 1; seed 4 votes for cand 2
    # For tie between cand 0 and cand 1: average softmax across ALL seeds is used
    # seed0: [0.7, 0.2, 0.1] → cand0 wins locally
    # seed1: [0.6, 0.3, 0.1] → cand0 wins locally
    # seed2: [0.3, 0.6, 0.1] → cand1 wins locally
    # seed3: [0.2, 0.7, 0.1] → cand1 wins locally
    # seed4: [0.1, 0.2, 0.7] → cand2 wins locally
    # Among tied (cand0, cand1): avg softmax for cand0 = (0.7+0.6+0.3+0.2+0.1)/5=0.38
    #                             avg softmax for cand1 = (0.2+0.3+0.6+0.7+0.2)/5=0.40
    # So cand1 wins.
    logits_per_seed = [
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
        [0.3, 0.6, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ]
    models = [_make_model([row]) for row in logits_per_seed]
    candidates = ["kedi +Noun", "ev +Noun", "git +Verb"]
    token2idx: dict[str, int] = {}

    best_idx, disambig_score, seed_agreement, seed_votes = _ensemble_score(
        models, candidates, token2idx, context_sentence="test", target_position=0,
    )

    # Tie between cand0 (2 votes) and cand1 (2 votes): cand1 has higher avg softmax
    assert best_idx == 1
    # seed_agreement reflects the winning count (2 in a 2-2-1 tie)
    assert seed_agreement == 2


def test_two_way_perfect_tie_softmax_resolves() -> None:
    """Exact 2-2 tie (4 seeds, 2 candidates): avg softmax picks winner."""
    logits_per_seed = [
        [0.9, 0.1],   # seed0 → cand0
        [0.8, 0.2],   # seed1 → cand0
        [0.3, 0.7],   # seed2 → cand1
        [0.4, 0.6],   # seed3 → cand1
    ]
    # avg softmax cand0 = (0.9+0.8+0.3+0.4)/4 = 0.60
    # avg softmax cand1 = (0.1+0.2+0.7+0.6)/4 = 0.40
    # cand0 should win
    models = [_make_model([row]) for row in logits_per_seed]
    candidates = ["kedi +Noun", "ev +Noun"]
    token2idx: dict[str, int] = {}

    best_idx, disambig_score, seed_agreement, seed_votes = _ensemble_score(
        models, candidates, token2idx, context_sentence="test", target_position=0,
    )

    assert best_idx == 0


# ---------------------------------------------------------------------------
# Tests: NaN/error seed fallback (§H+ A)
# ---------------------------------------------------------------------------

def test_nan_seed_dropped_from_vote() -> None:
    """A seed returning NaN logits is dropped; remaining seeds vote."""
    good_logits = [0.2, 0.7, 0.1]   # argmax=1
    nan_logits = [float("nan"), float("nan"), float("nan")]

    # seeds 0-3 return good logits (all vote cand1), seed 4 returns NaN
    seeds_logits = [good_logits] * 4 + [nan_logits]
    models = [_make_model([row]) for row in seeds_logits]
    candidates = ["kedi +Noun", "ev +Noun", "git +Verb"]
    token2idx: dict[str, int] = {}

    best_idx, disambig_score, seed_agreement, seed_votes = _ensemble_score(
        models, candidates, token2idx, context_sentence="test", target_position=0,
    )

    assert best_idx == 1
    # 4 usable seeds, all agree on cand1
    assert seed_agreement == 4


def test_seed_exception_dropped_from_vote() -> None:
    """A seed that raises during forward is excluded; ≥3 remain → proceed."""
    good_logits = [0.1, 0.8, 0.1]
    models_list = []
    for i in range(5):
        m = MagicMock()
        if i == 2:  # seed2 throws
            m.side_effect = RuntimeError("CUDA OOM")
        else:
            tensor_val = torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32)
            m.return_value = (tensor_val, None)
        m.eval = MagicMock(return_value=m)
        m.to = MagicMock(return_value=m)
        models_list.append(m)

    candidates = ["kedi +Noun", "ev +Noun", "git +Verb"]
    token2idx: dict[str, int] = {}

    best_idx, disambig_score, seed_agreement, seed_votes = _ensemble_score(
        models_list, candidates, token2idx, context_sentence="test", target_position=0,
    )

    # 4 usable seeds all vote cand1
    assert best_idx == 1
    assert seed_agreement == 4


def test_fewer_than_3_usable_seeds_returns_drop_signal() -> None:
    """<3 seeds usable → seed_agreement set to n_usable (caller routes to drop)."""
    good_logits = [0.1, 0.8, 0.1]
    models_list = []
    for i in range(5):
        m = MagicMock()
        if i < 3:  # seeds 0,1,2 throw
            m.side_effect = RuntimeError("error")
        else:
            tensor_val = torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32)
            m.return_value = (tensor_val, None)
        m.eval = MagicMock(return_value=m)
        m.to = MagicMock(return_value=m)
        models_list.append(m)

    candidates = ["kedi +Noun", "ev +Noun"]
    token2idx: dict[str, int] = {}

    best_idx, disambig_score, seed_agreement, seed_votes = _ensemble_score(
        models_list, candidates, token2idx, context_sentence="test", target_position=0,
    )

    # Only 2 seeds usable → caller must route to drop (seed_agreement < 3)
    assert seed_agreement < 3


# ---------------------------------------------------------------------------
# Tests: assign_tier_v2 integration
# ---------------------------------------------------------------------------

def test_assign_tier_v2_silver_confident() -> None:
    from aksu.data.tiers import Tier, assign_tier_v2
    tier = assign_tier_v2(candidate_count=3, seed_agreement=5, disambig_score=0.90, method="ensemble")
    assert tier == Tier.SILVER_CONFIDENT


def test_assign_tier_v2_silver_auto() -> None:
    from aksu.data.tiers import Tier, assign_tier_v2
    tier = assign_tier_v2(candidate_count=1, seed_agreement=0, disambig_score=1.0, method="unambiguous")
    assert tier == Tier.SILVER_AUTO


def test_assign_tier_v2_silver_marginal() -> None:
    from aksu.data.tiers import Tier, assign_tier_v2
    tier = assign_tier_v2(candidate_count=2, seed_agreement=4, disambig_score=0.75, method="ensemble")
    assert tier == Tier.SILVER_MARGINAL


def test_assign_tier_v2_bronze() -> None:
    from aksu.data.tiers import Tier, assign_tier_v2
    tier = assign_tier_v2(candidate_count=3, seed_agreement=3, disambig_score=0.65, method="ensemble")
    assert tier == Tier.BRONZE


def test_assign_tier_v2_drop_low_seeds_method() -> None:
    from aksu.data.tiers import Tier, assign_tier_v2
    tier = assign_tier_v2(candidate_count=2, seed_agreement=2, disambig_score=0.5, method="drop_low_seeds")
    assert tier == Tier.DROP


def test_assign_tier_v2_oov_method() -> None:
    from aksu.data.tiers import Tier, assign_tier_v2
    tier = assign_tier_v2(candidate_count=0, seed_agreement=0, disambig_score=0.0, method="zeyrek_oov")
    assert tier == Tier.DROP


def test_assign_tier_v2_gold_manually_verified() -> None:
    from aksu.data.tiers import Tier, assign_tier_v2
    tier = assign_tier_v2(candidate_count=1, seed_agreement=5, disambig_score=1.0, method="gold",
                          manually_verified=True)
    assert tier == Tier.GOLD


def test_v2_tier_to_v1_compat_mapping() -> None:
    from aksu.data.tiers import Tier, v2_tier_to_v1_compat
    assert v2_tier_to_v1_compat(Tier.GOLD) == "gold"
    assert v2_tier_to_v1_compat(Tier.SILVER_CONFIDENT) == "silver-auto"
    assert v2_tier_to_v1_compat(Tier.SILVER_AUTO) == "silver-auto"
    assert v2_tier_to_v1_compat(Tier.SILVER_MARGINAL) == "silver-agreed"
    assert v2_tier_to_v1_compat(Tier.BRONZE) is None
    assert v2_tier_to_v1_compat(Tier.DROP) is None
