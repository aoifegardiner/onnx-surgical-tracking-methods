"""Holoscan Python operator: full MFT around the RAFT-OU ONNX engine."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from holoscan_agx.mft_pipeline import OnnxMFTTracker
from holoscan_agx.operators import Operator, OperatorSpec


class MFTHostOp(Operator):
    """Per-frame MFT with ONNX RAFT-OU called once per delta.

    Input is a resized RGB frame (HxWx3 or NCHW). Output is current xy in
    model pixels (scale back in SampleFlowOp / eval).
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
        **kwargs,
    ):
        self.height = int(height)
        self.width = int(width)
        self.scale = float(scale)
        self.native_hw = native_hw
        self.tracker = OnnxMFTTracker(Path(onnx_path))
        pts = np.asarray(queries, dtype=np.float32).reshape(-1, 2)
        nat_h, nat_w = int(native_hw[0]), int(native_hw[1])
        sx = (self.width - 1) / max(nat_w - 1, 1)
        sy = (self.height - 1) / max(nat_h - 1, 1)
        self.queries_work = pts.copy()
        if pts.size:
            self.queries_work[:, 0] *= sx
            self.queries_work[:, 1] *= sy
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("coords")

    def compute(self, op_input, op_output, context):
        from holoscan_agx.operators import _as_nchw_rgb

        frame = _as_nchw_rgb(np.asarray(op_input.receive("in")), self.height, self.width)[0]
        q = None if self.tracker.initialized else self.queries_work
        coords_work = self.tracker.push_frame(frame, q)
        coords_native = coords_work.copy()
        nat_h, nat_w = int(self.native_hw[0]), int(self.native_hw[1])
        coords_native[:, 0] *= (nat_w - 1) / max(self.width - 1, 1)
        coords_native[:, 1] *= (nat_h - 1) / max(self.height - 1, 1)
        op_output.emit(coords_native.astype(np.float32), "coords")
