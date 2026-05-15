"""E-Step 6: Assert every .pt under models/ has a checkpoint_metadata.json sidecar.

Sidecars with provenance="backfilled" must have null for unrecoverable fields;
sidecars with provenance="forward" must have all fields populated.

Files listed in models/.aksuignore are exempt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

MODELS_DIR = Path("models")
AKSUIGNORE = MODELS_DIR / ".aksuignore"

REQUIRED_KEYS = {
    "provenance",
    "checkpoint_path",
    "file_size_bytes",
    "torch_version",
    "python_version",
}
FORWARD_REQUIRED = {
    "training_commit_sha",
    "wall_clock_seconds",
    "host_cpu",
}


def _load_ignore_set() -> set[str]:
    if not AKSUIGNORE.exists():
        return set()
    return {line.strip() for line in AKSUIGNORE.read_text().splitlines() if line.strip() and not line.startswith("#")}


def _all_checkpoints() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    ignore = _load_ignore_set()
    ckpts = []
    for pt in MODELS_DIR.rglob("*.pt"):
        if any(ig in str(pt) for ig in ignore):
            continue
        ckpts.append(pt)
    return ckpts


@pytest.mark.parametrize("ckpt", _all_checkpoints())
def test_checkpoint_has_sidecar(ckpt: Path) -> None:
    sidecar = ckpt.with_name(ckpt.stem + "_metadata.json")
    assert sidecar.exists(), (
        f"Missing sidecar for {ckpt}. Run: "
        f"python scripts/data/write_checkpoint_metadata.py {ckpt}"
    )
    data = json.loads(sidecar.read_text())
    for key in REQUIRED_KEYS:
        assert key in data, f"Sidecar {sidecar} missing required key {key!r}"

    provenance = data.get("provenance")
    assert provenance in ("forward", "backfilled"), (
        f"Sidecar {sidecar} has unknown provenance={provenance!r}"
    )

    if provenance == "forward":
        for key in FORWARD_REQUIRED:
            assert data.get(key) is not None, (
                f"Forward-provenance sidecar {sidecar} must have {key!r} set (not null)"
            )
    else:  # backfilled
        assert data.get("backfilled_at") is not None, (
            f"Backfilled sidecar {sidecar} must have 'backfilled_at' timestamp"
        )
