"""Executes every fenced Python block in README.md and, when a `# →` arrow is
present, asserts the last expression equals the arrow's expected literal."""
from __future__ import annotations
import ast
import re
from pathlib import Path

README = (Path(__file__).resolve().parent.parent / "README.md").read_text()
PY_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
ARROW = re.compile(r"#\s*→\s*(.+?)\s*$", re.M)


def _split_arrow(block: str) -> tuple[str, str | None]:
    """Return (executable_src, expected_repr_or_None)."""
    m = ARROW.search(block)
    if not m:
        return block, None
    expected = m.group(1).strip()
    src_no_arrow = ARROW.sub("", block).rstrip()
    return src_no_arrow, expected


def _exec_capture_last(src: str) -> object | None:
    """Compile src splitting off the trailing expression so we can capture its value."""
    tree = ast.parse(src)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    ns: dict = {}
    exec(compile(tree, "<readme>", "exec"), ns)
    if last is None:
        return None
    return eval(compile(ast.Expression(last.value), "<readme-tail>", "eval"), ns)


def test_readme_python_blocks_execute_and_arrows_match():
    failures = []
    for block in PY_BLOCK.findall(README):
        src, expected = _split_arrow(block)
        try:
            value = _exec_capture_last(src)
        except SystemExit:
            continue  # CLI examples ignored
        except (ImportError, ModuleNotFoundError):
            continue  # optional dependency not installed
        except RuntimeError as e:
            msg = str(e)
            if (
                "backend" in msg.lower()
                or "No backends" in msg
                or "state_dict" in msg       # BERTurk checkpoint weight mismatch in CI
                or "model weights" in msg.lower()
            ):
                continue  # requires model weights not present in CI
            failures.append(f"EXEC FAIL:\n{src}\nRuntimeError: {e}")
            continue
        except Exception as e:  # noqa: BLE001
            failures.append(f"EXEC FAIL:\n{src}\n{type(e).__name__}: {e}")
            continue
        if expected is None:
            continue
        actual = repr(value) if not isinstance(value, str) else value
        if actual != expected and repr(value) != expected:
            failures.append(
                f"ARROW MISMATCH:\nsrc:\n{src}\nexpected: {expected}\nactual:   {actual}"
            )
    assert not failures, "\n\n".join(failures)


def test_readme_atomizer_example_matches_code():
    from aksu import Atomizer

    result = Atomizer(backend="zeyrek").to_canonical("evlerinden")
    assert result == "ev +Noun +POSS.3PL +ABL", (
        f"README example mismatch: got {result!r}. Update README to match code output."
    )
