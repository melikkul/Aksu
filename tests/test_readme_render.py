"""CI gate: README.md must be byte-identical to rendering docs/README.md.j2 from metrics.json.

If this test fails, the README was hand-edited without updating the template, or metrics.json
drifted from the last render. Fix by running: python scripts/build_readme.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "docs/README.md.j2"
METRICS = ROOT / "audit/benchmark_results/metrics.json"
README = ROOT / "README.md"


def _render_in_process() -> str:
    """Render via the build_readme module directly (avoids subprocess)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import importlib
        spec = importlib.util.spec_from_file_location(
            "build_readme", ROOT / "scripts/build_readme.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import io, contextlib
        buf = io.StringIO()
        out_path = ROOT / "README.md.render_test_tmp"
        try:
            mod.render(TEMPLATE, METRICS, out_path)
            return out_path.read_text(encoding="utf-8")
        finally:
            if out_path.exists():
                out_path.unlink()
    finally:
        sys.path.pop(0)


@pytest.mark.skipif(not TEMPLATE.exists(), reason="docs/README.md.j2 not found")
@pytest.mark.skipif(not METRICS.exists(), reason="audit/benchmark_results/metrics.json not found")
def test_readme_is_rendered_from_template():
    """README.md must equal the template rendered against metrics.json."""
    try:
        import jinja2  # noqa: F401
    except ImportError:
        pytest.skip("jinja2 not installed")

    committed = README.read_text(encoding="utf-8")
    rendered = _render_in_process()

    if committed != rendered:
        import difflib
        diff = list(difflib.unified_diff(
            committed.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile="README.md (committed)",
            tofile="README.md (rendered from template)",
            n=3,
        ))
        diff_str = "".join(diff[:60])
        pytest.fail(
            f"README.md is not up-to-date with docs/README.md.j2 + metrics.json.\n"
            f"Run: python scripts/build_readme.py\n\n{diff_str}"
        )


@pytest.mark.skipif(not METRICS.exists(), reason="metrics.json not found")
def test_metrics_json_is_valid():
    """metrics.json must be valid JSON with required keys."""
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    required = {
        "em_argmax_ensemble",
        "training_wall_clock_min",
        "zeyrek_tok_per_sec",
        "dataset_v1_entries",
        "classification_macro_f1_atomized_berturk",
    }
    missing = required - set(data.keys())
    assert not missing, f"metrics.json missing required keys: {missing}"
