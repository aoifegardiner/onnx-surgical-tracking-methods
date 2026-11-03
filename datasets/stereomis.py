"""StereoMIS-tracking sequences in the layout used by the ONNX evaluation.

Each sequence directory contains ``track_pts.pckl`` and left frames in
``video_frames/*l.png``. Ground truth is filtered the same way as PointTracker:

* missing annotations become ``[-1, -1]``, then ``[0, 0]``
* points marked not-visible, or sitting at the origin, are dropped
* only points that are valid on the first sliced frame are kept
* frames are sliced ``[start:stop:step]``
"""

from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_DATADIR = os.environ.get("STEREOMIS_DATA", "data/StereoMIS-tracking")

# Annotated 2D tracks in the public tracking split.
DEFAULT_SEQUENCES = ("P1_1", "P2_0", "P2_1", "P2_6", "P3_1", "P3_2")


def load_gt_like_pointtracker(
    gt_file: str,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``gt_2d`` of shape (N, T, 2) and a boolean ``valid`` mask of shape (N, T)."""
    with open(gt_file, "rb") as handle:
        cont = pickle.load(handle)
        gt_points = cont["points"]
    gt_points = np.stack(
        [np.stack([np.array([-1, -1]) if i is None else i for i in pts]) for pts in gt_points]
    )
    gt_points = np.stack(
        [np.stack([np.array([0, 0]) if i is None else i for i in pts]) for pts in gt_points]
    )
    mask = ~np.stack(cont["not_visible"]).astype(bool)
    mask &= gt_points.sum(axis=-1) > 0
    gt_points = gt_points[mask[:, 0]]
    mask = mask[mask[:, 0]]
    frame_slice = slice(start, stop, step)
    gt_2d = np.asarray(gt_points[:, frame_slice], dtype=np.float32)
    valid = np.asarray(mask[:, frame_slice], dtype=bool)
    return gt_2d, valid


@dataclass
class StereoMISSequence:
    name: str
    input_folder: str
    frame_paths: List[str]
    gt_2d: np.ndarray
    valid: np.ndarray
    height: int
    width: int

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)

    @property
    def num_points(self) -> int:
        return int(self.gt_2d.shape[0])

    def query_points(self) -> np.ndarray:
        """Frame-0 ground truth, the queries the trackers start from."""
        return np.asarray(self.gt_2d[:, 0], dtype=np.float32)


def list_sequences(datadir: str) -> List[str]:
    root = Path(datadir)
    if not root.is_dir():
        raise FileNotFoundError(f"StereoMIS datadir not found: {datadir}")
    names = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "track_pts.pckl").is_file() and (entry / "video_frames").is_dir():
            names.append(entry.name)
    return names


def load_sequence(
    datadir: str,
    name: str,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
) -> StereoMISSequence:
    folder = os.path.join(datadir, name)
    gt_file = os.path.join(folder, "track_pts.pckl")
    if not os.path.isfile(gt_file):
        raise FileNotFoundError(f"{name} has no track_pts.pckl at {gt_file}")

    frame_paths = sorted(glob.glob(os.path.join(folder, "video_frames", "*l.png")))
    if not frame_paths:
        raise FileNotFoundError(f"{name} has no left frames in {folder}/video_frames")
    frame_paths = frame_paths[slice(start, stop, step)]
    gt_2d, valid = load_gt_like_pointtracker(gt_file, start=start, stop=stop, step=step)

    if gt_2d.shape[1] != len(frame_paths):
        raise ValueError(
            f"{name}: GT has {gt_2d.shape[1]} frames after slicing, "
            f"but found {len(frame_paths)} left images"
        )
    if gt_2d.shape[0] < 1:
        raise ValueError(f"{name}: no query points remain after filtering")

    probe = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    if probe is None:
        raise FileNotFoundError(f"Could not read {frame_paths[0]}")
    height, width = probe.shape[:2]
    return StereoMISSequence(
        name=name,
        input_folder=folder,
        frame_paths=frame_paths,
        gt_2d=gt_2d,
        valid=valid,
        height=height,
        width=width,
    )


def load_sequences(
    datadir: str = DEFAULT_DATADIR,
    sequences: Optional[Sequence[str]] = None,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
) -> List[StereoMISSequence]:
    available = list_sequences(datadir)
    if sequences is None or len(sequences) == 0:
        requested = [name for name in DEFAULT_SEQUENCES if name in available]
        extra = [name for name in available if name not in requested]
        names = requested + extra
    else:
        names = list(sequences)
        missing = [name for name in names if name not in available]
        if missing:
            raise FileNotFoundError(
                f"Requested StereoMIS sequences without track_pts.pckl: {missing}. "
                f"Available: {available}"
            )
    return [load_sequence(datadir, name, start=start, stop=stop, step=step) for name in names]
