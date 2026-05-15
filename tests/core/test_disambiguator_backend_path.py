"""Regression for F1 (analyzer.py): the disambiguator-backend init must not
raise NameError: name 'Path' is not defined. Runs in fast CI (no gpu/slow markers).

Point at non-existent paths so the codepath trips Path(vocab_dir) before any
heavyweight model loading. FileNotFoundError => Path is imported correctly.
NameError => regression.
"""
import pytest


def test_disambiguator_backend_loads_without_pathlib_nameerror(tmp_path):
    from aksu import MorphoAnalyzer

    with pytest.raises((FileNotFoundError, RuntimeError, Exception)) as exc_info:
        MorphoAnalyzer(
            backends=["disambiguator"],
            vocab_dir=str(tmp_path),
            model_path=str(tmp_path / "nonexistent.pt"),
        )
    # The ONLY wrong exception is NameError — that means Path was not imported
    assert not isinstance(exc_info.value, NameError), (
        "NameError means pathlib.Path is not imported at module level in analyzer.py — "
        "regression of F1 bug"
    )
