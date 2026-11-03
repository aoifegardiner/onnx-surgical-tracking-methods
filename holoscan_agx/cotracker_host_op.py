"""Holoscan operator: CoTracker online step with GPU-resident previous_* state."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from export_onnx.cotracker_ort import CoTrackerStepRunner
from holoscan_agx.operators import Operator, OperatorSpec


class CoTrackerHostOp(Operator):
    """Sliding online windows (T=16, step=8) around ``CoTrackerStepRunner``.

    InferenceOp cannot keep ``previous_*`` on device or enable CUDA graphs.
    This op does both. A full window is required before the first forward;
    the last incomplete step keeps the last predicted xy.
    """

    def __init__(
        self,
        fragment,
        *args,
        onnx_path: str,
        height: int,
        width: int,
        queries: np.ndarray,
        native_hw: tuple,
        extra: Optional[dict] = None,
        **kwargs,
    ):
        extra = dict(extra or {})
        self.height = int(height)
        self.width = int(width)
        self.native_hw = (int(native_hw[0]), int(native_hw[1]))
        self.window = int(extra.get("window_len", 16))
        self.step = int(extra.get("step", self.window // 2))
        self.runner = CoTrackerStepRunner(Path(onnx_path))
        self.n_points = self.runner.n_points
        pts = np.asarray(queries, dtype=np.float32).reshape(-1, 2)
        nat_h, nat_w = self.native_hw
        sx = (self.width - 1) / max(nat_w - 1, 1)
        sy = (self.height - 1) / max(nat_h - 1, 1)
        work = pts.copy()
        if work.size:
            work[:, 0] *= sx
            work[:, 1] *= sy
        self.n_real = int(work.shape[0])
        self.pts_work = work
        self.states = None
        self.frames: List[np.ndarray] = []
        self.win_start = 0
        self.last_tracks = None
        self.last_start = 0
        self.fallback_xy = work.astype(np.float32)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("coords")

    def compute(self, op_input, op_output, context):
        from holoscan_agx.operators import _as_nchw_rgb

        frame = _as_nchw_rgb(np.asarray(op_input.receive("in")), self.height, self.width)
        self.frames.append(frame[0])
        n = len(self.frames)
        while self.win_start + self.window <= n:
            sl = self.frames[self.win_start : self.win_start + self.window]
            video = np.stack(sl, axis=0)[None].astype(np.float32)
            tracks, self.states = self.runner.run_all_points(video, self.pts_work, self.states)
            self.last_tracks = tracks
            self.last_start = self.win_start
            self.win_start += self.step
        if self.last_tracks is None or self.n_real == 0:
            xy = self.fallback_xy
        else:
            local = min(max(n - 1 - self.last_start, 0), self.window - 1)
            xy = self.last_tracks[0, local, : self.n_real].astype(np.float32)
        nat_h, nat_w = self.native_hw
        native = xy.copy()
        native[:, 0] *= (nat_w - 1) / max(self.width - 1, 1)
        native[:, 1] *= (nat_h - 1) / max(self.height - 1, 1)
        op_output.emit(native.astype(np.float32), "coords")
