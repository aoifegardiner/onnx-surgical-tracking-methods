#!/usr/bin/env python3
"""Download the ONNX graphs the STIR and StereoMIS evaluations open.

The Hugging Face dataset also contains exports this repository does not run
(per-scale CoTracker copies, init graphs, the single TAPNext++ graph, and an
older RAFT file). Those are left on the server.

A file the evaluation asks for that is not in the dataset is reported and
skipped. StereoMIS RAFT and WAFT are in that group.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "aoifegardiner/ONNX_Tracking_Models"
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

DATASETS = ("stir", "stereomis")
SCALES = (1, 2, 4)
# Dense and windowed trackers. CoTracker, LiteTracker, and TAPNext++ are separate.
SCALED_METHODS = ("raft", "waft", "mft", "alltracker", "pips", "locotrack")

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
    "waft": "WAFT",
}


def needed_names(fp16: bool = False) -> list[str]:
    """Filenames resolve_artifact and the hosts open. Order is stable."""
    names: list[str] = []
    for dataset in DATASETS:
        for method in SCALED_METHODS:
            for scale in SCALES:
                stem = "%s_%s_scale%d" % (method, dataset, scale)
                names.append(stem + ".onnx")
                if method == "alltracker":
                    names.append(stem + "_pair.onnx")
                elif method == "locotrack":
                    names.append(stem + "_feats.onnx")
    names.append("cotracker_online_step.onnx")
    if fp16:
        names.append("cotracker_online_step_fp16.onnx")
    names.extend(("fnet.onnx", "corr_mlp.onnx", "updateformer.onnx"))
    return names


def destination(name: str) -> Path:
    if name in SHARED:
        folder = SHARED[name]
    else:
        prefix = name.split("_", 1)[0]
        if prefix not in PREFIX:
            raise ValueError("No folder mapping for %s" % name)
        folder = PREFIX[prefix]
    return MODELS / folder / name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Keep needed filenames that contain one of these substrings (for example raft_stir).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Also download cotracker_online_step_fp16.onnx, used only with --fp16 on the eval scripts.",
    )
    parser.add_argument("--list", action="store_true", help="Print the file list and do not download.")
    parser.add_argument("--force", action="store_true", help="Replace files that are already downloaded.")
    args = parser.parse_args()

    names = needed_names(fp16=args.fp16)
    if args.only:
        needles = [item.lower() for item in args.only]
        names = [name for name in names if any(needle in name.lower() for needle in needles)]
    if not names:
        print("No evaluation graphs match --only.", file=sys.stderr)
        return 1

    if args.list:
        for name in names:
            print(destination(name).relative_to(ROOT))
        return 0

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Install huggingface_hub first: pip install -r requirements.txt", file=sys.stderr)
        return 1

    published = {name for name in list_repo_files(REPO_ID, repo_type="dataset") if name.endswith(".onnx")}
    present = [name for name in names if name in published]
    missing = [name for name in names if name not in published]
    print("Dataset: https://huggingface.co/datasets/%s" % REPO_ID)
    print("Needed:  %d    published: %d    not in the dataset: %d" % (len(names), len(present), len(missing)))
    for name in missing:
        print("absent %s" % name)

    for name in present:
        dest = destination(name)
        if dest.is_file() and not args.force:
            print("keep   %s" % dest.relative_to(ROOT))
            continue
        print("fetch  %s" % name)
        cached = Path(hf_hub_download(REPO_ID, filename=name, repo_type="dataset"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dest)
        print("       -> %s  (%.1f MB)" % (dest.relative_to(ROOT), dest.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
