"""Render docs/README.md.j2 from audit/benchmark_results/metrics.json → README.md.

Usage:
    python scripts/build_readme.py
    python scripts/build_readme.py --template docs/README.md.j2 \\
        --metrics audit/benchmark_results/metrics.json --output README.md

CI gate: tests/test_readme_render.py re-renders and asserts byte-identical output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jinja2():
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
        return Environment, FileSystemLoader, StrictUndefined
    except ImportError:
        print("ERROR: jinja2 not installed. Run: pip install jinja2", file=sys.stderr)
        sys.exit(1)


def render(template_path: Path, metrics_path: Path, output_path: Path) -> str:
    Environment, FileSystemLoader, StrictUndefined = _load_jinja2()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    template = env.get_template(template_path.name)
    rendered = template.render(metrics=metrics)

    output_path.write_text(rendered, encoding="utf-8")
    return rendered


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", default=str(root / "docs/README.md.j2"))
    ap.add_argument("--metrics", default=str(root / "audit/benchmark_results/metrics.json"))
    ap.add_argument("--output", default=str(root / "README.md"))
    args = ap.parse_args()

    tpl = Path(args.template)
    met = Path(args.metrics)
    out = Path(args.output)

    for p in (tpl, met):
        if not p.exists():
            print(f"ERROR: {p} does not exist", file=sys.stderr)
            sys.exit(1)

    render(tpl, met, out)
    print(f"Rendered {tpl} → {out}")


if __name__ == "__main__":
    main()
