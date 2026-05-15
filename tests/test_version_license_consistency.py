"""Asserts a single source of truth for version and license.

Closes audit findings λ² (version drift) and D5 (license drift).
"""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_version_consistency():
    import aksu
    from aksu._version import __version__ as v_central

    assert aksu.__version__ == v_central

    cff = (ROOT / "CITATION.cff").read_text()
    assert f"version: {v_central}" in cff, (
        f"CITATION.cff missing 'version: {v_central}'; update CITATION.cff"
    )

    cl = (ROOT / "CHANGELOG.md").read_text()
    assert f"[{v_central}]" in cl, (
        f"CHANGELOG.md missing entry for [{v_central}]"
    )


def test_license_consistency():
    pyproj = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproj["project"]["license"] == "MIT", (
        "pyproject.toml license must be 'MIT' (PEP 639 SPDX expression). "
        "Do NOT add a 'License :: OSI Approved :: MIT License' classifier — "
        "setuptools ≥77 rejects that combination."
    )

    cff = (ROOT / "CITATION.cff").read_text()
    assert "license: MIT" in cff, "CITATION.cff must have 'license: MIT'"

    assert "MIT License" in (ROOT / "LICENSE").read_text(), (
        "LICENSE file must contain 'MIT License'"
    )
