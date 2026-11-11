"""Holoscan-faithful MFT: ONNX RAFT-OU + official delta chaining.

The published ``mft_*_scale*.onnx`` graph is RAFT-OU only. This module runs the
same ``MFT.track`` state machine as ``STIRMetricsmain.src.datatest.mft.MFTTracker``,
substituting the PyTorch flower for ``OnnxRaftOuFlower``.

On Clara AGX this is the body of ``MFTHostOp`` (Python operator that owns the
ORT session and calls RAFT-OU once per delta). On the DGX it is used directly
by the eval scripts because Holoscan SDK is not installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import time

import cv2
import numpy as np
import torch

from onnx_hosts.mft import MFT, MFTConfig, convert_to_point_tracking

from holoscan_agx.onnx_flower import OnnxRaftOuFlower


def _nchw_rgb_to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    chw = np.asarray(frame)
    if chw.ndim == 4:
        chw = chw[0]
    rgb = np.transpose(chw, (1, 2, 0))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class OnnxMFTTracker:
    """Online MFT over an ONNX RAFT-OU engine."""

    def __init__(self, onnx_path: Path, occlusion_threshold: float = 0.02):
        config = MFTConfig(occlusion_threshold=occlusion_threshold)
        # Do not construct the PyTorch RAFT flower; ONNX replaces it.
        self.tracker = MFT.__new__(MFT)
        self.tracker.C = config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker.device = device
        self.tracker.flower = OnnxRaftOuFlower(onnx_path, device=device)
        self.initialized = False
        self.queries: Optional[torch.Tensor] = None
        self.latencies_ms: List[float] = []

    def reset(self) -> None:
        self.initialized = False
        self.queries = None
        self.latencies_ms = []

    def push_frame(self, frame_nchw: np.ndarray, queries_xy: Optional[np.ndarray] = None) -> np.ndarray:
        t0 = time.perf_counter()
        bgr = _nchw_rgb_to_bgr_uint8(frame_nchw)
        if not self.initialized:
            if queries_xy is None:
                raise RuntimeError("queries_xy required on the first MFT frame")
            device = self.tracker.device
            self.queries = torch.from_numpy(np.asarray(queries_xy, dtype=np.float32)).to(device)
            self.tracker.init(bgr)
            self.initialized = True
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            return np.asarray(queries_xy, dtype=np.float32)
        self.tracker.flower.call_latencies_ms = []
        meta = self.tracker.track(bgr, query_points=self.queries)
        coords, _occ = convert_to_point_tracking(meta.result, self.queries)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return np.asarray(coords, dtype=np.float32)

    def track_sequence(
        self,
        frames_nchw: Sequence[np.ndarray],
        queries_xy: np.ndarray,
    ) -> Tuple[np.ndarray, List[float]]:
        n_real = int(queries_xy.shape[0])
        t_all = len(frames_nchw)
        pred = np.zeros((n_real, t_all, 2), dtype=np.float32)
        self.reset()
        for t, frame in enumerate(frames_nchw):
            pred[:, t] = self.push_frame(frame, queries_xy if t == 0 else None)
        return pred, list(self.latencies_ms[1:] if len(self.latencies_ms) > 1 else self.latencies_ms)
