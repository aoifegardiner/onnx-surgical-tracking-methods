#!/usr/bin/env python3
"""Download the published ONNX graphs into models/.

The files live in the Hugging Face dataset
https://huggingface.co/datasets/aoifegardiner/ONNX_Tracking_Models
as a flat list. This script places each file in the folder the evaluation
expects, for example models/RAFT/raft_stir_scale1.onnx.

Graphs already present are left in place. Pass --force to download again.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "aoifegardiner/ONNX_Tracking_Models"
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

SHARED = {
    "fnet.onnx": "LITETRACKER",
    "corr_mlp.onnx": "LITETRACKER",
    "updateformer.onnx": "LITETRACKER",
}
PREFIX = {
    "alltracker": "ALLTRACKER",
    "cotracker": "COTRACKER",
    "locotrack": "LOCOTRACK",
    "mft": "MFT",
    "pips": "PIPS",
    "raft": "RAFT",
    "tapnextpp": "TAPNEXTPP",
    "waft": "WAFT",
}


def destination(name: str) -> Path:
    if name in SHARED:
        folder = SHARED[name]
    else:
        prefix = name.split("_", 1)[0]
        if prefix not in PREFIX:
            raise ValueError(f"No folder mapping for {name}")
        folder = PREFIX[prefix]
    return MODELS / folder / name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Download filenames that contain one of these substrings (for example RAFT stir).",
    )
    parser.add_argument("--force", action="store_true", help="Replace files that are already downloaded.")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Install huggingface_hub first: pip install -r requirements.txt", file=sys.stderr)
        return 1

    names = [name for name in list_repo_files(REPO_ID, repo_type="dataset") if name.endswith(".onnx")]
    names.sort()
    if args.only:
        needles = [item.lower() for item in args.only]
        names = [name for name in names if any(needle in name.lower() for needle in needles)]
    if not names:
        print("No matching ONNX files in the dataset.", file=sys.stderr)
        return 1

    print(f"Dataset: https://huggingface.co/datasets/{REPO_ID}")
    print(f"Files:   {len(names)}")
    for name in names:
        dest = destination(name)
        if dest.is_file() and not args.force:
            print(f"keep   {dest.relative_to(ROOT)}")
            continue
        print(f"fetch  {name}")
        cached = Path(hf_hub_download(REPO_ID, filename=name, repo_type="dataset"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dest)
        print(f"       -> {dest.relative_to(ROOT)}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
