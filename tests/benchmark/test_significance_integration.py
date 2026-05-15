"""Integration test: significance chain on a 60-document fixture.

Runs _align_paired → paired_bootstrap_test → holm_bonferroni_correction and
asserts the output JSON contains a corr_p for every pair.
"""
import json
import pytest
import random

from aksu.benchmark.run_all_benchmarks import _align_paired, _run_significance
from aksu.benchmark.significance import paired_bootstrap_test, holm_bonferroni_correction


@pytest.fixture()
def fixture_60docs():
    """Deterministic 60-doc, 3-system fixture (gold is balanced binary labels)."""
    rng = random.Random(42)
    n = 60
    gold = {(0, i): rng.randint(0, 1) for i in range(n)}
    # System A: ~80% correct
    sys_a = {k: v if rng.random() < 0.80 else 1 - v for k, v in gold.items()}
    # System B: ~75% correct
    sys_b = {k: v if rng.random() < 0.75 else 1 - v for k, v in gold.items()}
    # System C: ~70% correct
    sys_c = {k: v if rng.random() < 0.70 else 1 - v for k, v in gold.items()}
    return gold, {"sys_a": sys_a, "sys_b": sys_b, "sys_c": sys_c}


def test_align_paired_basic(fixture_60docs):
    gold, preds = fixture_60docs
    pa, pb, gd = _align_paired(preds["sys_a"], preds["sys_b"], gold)
    assert len(pa) == len(pb) == len(gd) == 60


def test_align_paired_empty_raises(fixture_60docs):
    gold, preds = fixture_60docs
    # disjoint keys → ValueError
    shifted_preds = {(1, k[1]): v for k, v in preds["sys_a"].items()}
    with pytest.raises(ValueError, match="no shared"):
        _align_paired(shifted_preds, preds["sys_b"], gold)


def test_run_significance_output_shape(fixture_60docs):
    gold, preds = fixture_60docs
    result = _run_significance(preds, gold, n_bootstrap=500, seed=99)

    assert "pairs" in result
    assert len(result["pairs"]) == 3  # C(3,2) = 3 pairs

    for pair in result["pairs"]:
        assert "corr_p" in pair
        assert "raw_p" in pair
        assert "mean_diff" in pair
        assert "n_items" in pair
        assert isinstance(pair["significant"], bool)
        assert 0.0 <= pair["corr_p"] <= 1.0


def test_run_significance_json_serializable(fixture_60docs):
    gold, preds = fixture_60docs
    result = _run_significance(preds, gold, n_bootstrap=200, seed=0)
    serialized = json.dumps(result)
    loaded = json.loads(serialized)
    assert len(loaded["pairs"]) == 3


def test_significance_single_pair(fixture_60docs):
    gold, preds = fixture_60docs
    result = _run_significance(
        {"sys_a": preds["sys_a"], "sys_b": preds["sys_b"]},
        gold,
        n_bootstrap=200,
    )
    assert len(result["pairs"]) == 1
    # With only one pair, raw_p == corr_p (Holm-Bonferroni identity for n=1)
    pair = result["pairs"][0]
    assert abs(pair["raw_p"] - pair["corr_p"]) < 1e-9
