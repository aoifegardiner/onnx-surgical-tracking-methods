"""Holoscan Python operator: growing keep-first LocoTrack around ORT."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from holoscan_agx.locotrack_pipeline import OnnxLocoTrackTracker
from holoscan_agx.operators import Operator, OperatorSpec


class LocoTrackHostOp(Operator):
    """Per-frame LocoTrackSTIR-style inference.

    Accumulates RGB frames, re-runs the ONNX clip on the keep-first growing
    window (cap 300), emits current xy in native pixels. Queries stay at t=0.
    """

    def __init__(
        self,
        fragment,
        *args,
        onnx_path: str,
        height: int,
        width: int,
        scale: int,
        queries: np.ndarray,
        native_hw: tuple,
        extra: Optional[dict] = None,
        **kwargs,
    ):
        self.height = int(height)
        self.width = int(width)
        self.native_hw = native_hw
        extra = dict(extra or {})
        extra.setdefault("height", self.height)
        extra.setdefault("width", self.width)
        self.tracker = OnnxLocoTrackTracker(Path(onnx_path), extra=extra)
        pts = np.asarray(queries, dtype=np.float32).reshape(-1, 2)
        nat_h, nat_w = int(native_hw[0]), int(native_hw[1])
        sx = (self.width - 1) / max(nat_w - 1, 1)
        sy = (self.height - 1) / max(nat_h - 1, 1)
        self.queries_work = pts.copy()
        if pts.size:
            self.queries_work[:, 0] *= sx
            self.queries_work[:, 1] *= sy
        del scale
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("coords")

    def compute(self, op_input, op_output, context):
        from holoscan_agx.operators import _as_nchw_rgb

        frame = _as_nchw_rgb(np.asarray(op_input.receive("in")), self.height, self.width)
        q = None if self.tracker.queries is not None else self.queries_work
        coords_work = self.tracker.push_frame(frame, q)
        coords_native = coords_work.copy()
        nat_h, nat_w = int(self.native_hw[0]), int(self.native_hw[1])
        coords_native[:, 0] *= (nat_w - 1) / max(self.width - 1, 1)
        coords_native[:, 1] *= (nat_h - 1) / max(self.height - 1, 1)
        op_output.emit(coords_native.astype(np.float32), "coords")
