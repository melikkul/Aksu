"""One-shot LibCST codemod: kokturk → aksu.kokturk, ariturk → aksu.ariturk,
and bare imports of the in-tree-but-not-namespaced packages: data, train,
benchmark, classify, optimize, annotation, resource → aksu.<same>.

Usage:  PYTHONPATH=. python tools/rename_imports.py src/ tests/ scripts/
"""
import sys
from pathlib import Path

import libcst as cst


RENAMES_PREFIX = {
    "kokturk":    "aksu.kokturk",
    "ariturk":    "aksu.ariturk",
    "data":       "aksu.data",
    "train":      "aksu.train",
    "benchmark":  "aksu.benchmark",
    "classify":   "aksu.classify",
    "optimize":   "aksu.optimize",
    "annotation": "aksu.annotation",
    "resource":   "aksu.resource",
}

# Do NOT rename these when they appear as the head of an import —
# they are stdlib or third-party modules sharing a name with our internal dirs.
STDLIB_SKIP = {"resource"}  # `import resource` is stdlib


def _str(node) -> str:
    if isinstance(node, cst.Name):
        return node.value
    return cst.Module([]).code_for_node(node)


def _parse(dotted: str):
    parts = dotted.split(".")
    expr: cst.BaseExpression = cst.Name(parts[0])
    for p in parts[1:]:
        expr = cst.Attribute(value=expr, attr=cst.Name(p))
    return expr


class RenameImports(cst.CSTTransformer):
    def leave_Import(self, orig, upd):
        new_names = []
        for alias in upd.names:
            name = _str(alias.name)
            head = name.split(".", 1)[0]
            if head in RENAMES_PREFIX and head not in STDLIB_SKIP:
                new_full = name.replace(head, RENAMES_PREFIX[head], 1)
                new_names.append(alias.with_changes(name=_parse(new_full)))
            else:
                new_names.append(alias)
        return upd.with_changes(names=tuple(new_names))

    def leave_ImportFrom(self, orig, upd):
        if upd.module is None:
            return upd
        mod = _str(upd.module)
        head = mod.split(".", 1)[0]
        if head in RENAMES_PREFIX and head not in STDLIB_SKIP:
            new_mod = mod.replace(head, RENAMES_PREFIX[head], 1)
            return upd.with_changes(module=_parse(new_mod))
        return upd


def main():
    roots = [Path(p) for p in sys.argv[1:]]
    transformer = RenameImports()
    changed = 0
    for root in roots:
        for f in root.rglob("*.py"):
            if any(part in f.parts for part in (".venv", ".venv-audit", "egg-info", "compat")):
                continue
            src = f.read_text(encoding="utf-8")
            try:
                tree = cst.parse_module(src)
            except cst.ParserSyntaxError as e:
                print(f"SKIP {f}: {e}", file=sys.stderr)
                continue
            new = tree.visit(transformer).code
            if new != src:
                f.write_text(new, encoding="utf-8")
                print(f"WROTE {f}")
                changed += 1
    print(f"\nTotal files rewritten: {changed}", file=sys.stderr)


if __name__ == "__main__":
    main()
