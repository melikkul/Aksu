"""E-Step 6: Write checkpoint_metadata.json sidecars for all .pt files.

For existing checkpoints, backfills with provenance="backfilled" and null for
fields that cannot be reconstructed (training commit SHA, exact wall-clock).
For new checkpoints produced by the training script, provenance="forward".

Usage:
    python scripts/data/write_checkpoint_metadata.py models/
    python scripts/data/write_checkpoint_metadata.py models/v6/disambiguator/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _torch_version() -> str | None:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None


def write_sidecar(ckpt_path: Path, *, overwrite: bool = False) -> bool:
    """Write a sidecar JSON next to ckpt_path. Returns True if written."""
    sidecar = ckpt_path.with_name(ckpt_path.stem + "_metadata.json")
    if sidecar.exists() and not overwrite:
        return False

    existing: dict = {}
    if sidecar.exists():
        existing = json.loads(sidecar.read_text())

    provenance = existing.get("provenance", "backfilled")

    metadata: dict = {
        "provenance": provenance,
        "checkpoint_path": str(ckpt_path),
        "file_size_bytes": ckpt_path.stat().st_size,
        "backfilled_at": datetime.now(timezone.utc).isoformat() if provenance == "backfilled" else None,
        "training_commit_sha": None if provenance == "backfilled" else _git_sha(),
        "training_config_hash": None,  # cannot reconstruct
        "data_sha256": None,           # cannot reconstruct
        "wall_clock_seconds": None if provenance == "backfilled" else existing.get("wall_clock_seconds"),
        "host_cpu": None if provenance == "backfilled" else existing.get("host_cpu"),
        "host_gpu": None if provenance == "backfilled" else existing.get("host_gpu"),
        "torch_version": _torch_version(),
        "python_version": sys.version.split()[0],
    }
    # Preserve any forward-provenance fields from existing sidecar
    for k, v in existing.items():
        if k not in metadata or metadata[k] is None:
            metadata[k] = v

    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="Path to .pt file or directory containing .pt files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing sidecars")
    args = ap.parse_args()

    written = 0
    skipped = 0
    for target in args.targets:
        path = Path(target)
        if path.is_dir():
            ckpts = list(path.rglob("*.pt"))
        elif path.suffix == ".pt":
            ckpts = [path]
        else:
            print(f"Skipping {path} (not a .pt file or directory)", file=sys.stderr)
            continue

        for ckpt in ckpts:
            if write_sidecar(ckpt, overwrite=args.overwrite):
                print(f"WROTE {ckpt.with_name(ckpt.stem + '_metadata.json')}")
                written += 1
            else:
                skipped += 1

    print(f"Done: {written} written, {skipped} skipped (use --overwrite to force)")


if __name__ == "__main__":
    main()
