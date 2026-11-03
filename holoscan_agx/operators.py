"""Holoscan operators that reconstruct the STIR 2D tracking loop on Clara AGX.

This module is intended to run on the AGX (Holoscan SDK 2.x/3.x). It keeps
Python/CUDA for state and post-process, and leaves the network itself to
``InferenceOp`` (TensorRT engine built from the ONNX files in onnx_models/).

Operator graph (dense flow: RAFT / WAFT / MFT flower):

    VideoStreamReplayerOp
            |
    FormatConverterOp          RGB, resize to dataset/scale HxW
                               STIR 1024x1280; StereoMIS 512x640
            |
       FramePairOp             hold previous frame, emit (image1, image2)
            |
       InferenceOp             TensorRT / ONNX, outputs dense flow
            |
      SampleFlowOp             bilinear sample at query points, scale back
            |
      MetricsSinkOp            CUDA-synced latency, same schema as LatencyLogger
            |
         HolovizOp             optional overlay

MFT adds MFTChainOp between InferenceOp and SampleFlowOp (delta memory).
AllTracker and LocoTrack use host operators (variable clip / window) instead of
InferenceOp. PIPS uses ClipBufferOp + InferenceOp.
CoTracker uses CoTrackerHostOp (CUDA graph, GPU previous_*).
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

try:
    from holoscan.core import Operator, OperatorSpec
except ImportError:  # pragma: no cover - workstation without Holoscan
    Operator = object  # type: ignore

    class OperatorSpec:  # type: ignore
        def input(self, *args, **kwargs):
            return self

        def output(self, *args, **kwargs):
            return self

        def param(self, *args, **kwargs):
            return self


def _as_nchw_rgb(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """uint8 HxWxC RGB -> float32 [1,3,H,W] in [0, 255]."""
    import cv2

    if frame.ndim == 3 and frame.shape[2] == 3:
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        chw = np.transpose(frame, (2, 0, 1))
        return np.ascontiguousarray(chw[None].astype(np.float32))
    raise ValueError(f"Expected HxWx3 RGB, got {frame.shape}")


class FramePairOp(Operator):
    """Turn a live RGB stream into consecutive (prev, curr) pairs."""

    def __init__(self, fragment, *args, height: int, width: int, **kwargs):
        self.height = int(height)
        self.width = int(width)
        self._prev: Optional[np.ndarray] = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("image1")
        spec.output("image2")

    def compute(self, op_input, op_output, context):
        frame = op_input.receive("in")
        current = _as_nchw_rgb(np.asarray(frame), self.height, self.width)
        if self._prev is None:
            self._prev = current
            return
        op_output.emit(self._prev, "image1")
        op_output.emit(current, "image2")
        self._prev = current


class ClipBufferOp(Operator):
    """Fixed-length clip for windowed trackers (PIPS, CoTracker).

    Emits once the buffer is full (or ``min_len``). LocoTrack no longer uses
    this op: Holoscan InferenceOp cannot change T, so ``LocoTrackHostOp``
    grows the keep-first clip and calls ORT itself.
    """

    def __init__(
        self,
        fragment,
        *args,
        height: int,
        width: int,
        max_len: int,
        layout: str = "ncthw",
        keep_first: bool = False,
        min_len: int = None,
        pad_to_max: bool = True,
        **kwargs,
    ):
        self.height = int(height)
        self.width = int(width)
        self.max_len = int(max_len)
        self.layout = layout
        self.keep_first = bool(keep_first)
        self.min_len = int(min_len) if min_len is not None else (2 if keep_first else int(max_len))
        self.pad_to_max = bool(pad_to_max)
        self.minus_one_one = bool(kwargs.pop("minus_one_one", False))
        self._frames: Deque[np.ndarray] = deque()
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("clip")

    def compute(self, op_input, op_output, context):
        frame = _as_nchw_rgb(np.asarray(op_input.receive("in")), self.height, self.width)
        self._frames.append(frame)
        if self.keep_first and len(self._frames) > self.max_len:
            first = self._frames[0]
            rest = list(self._frames)[1:][-(self.max_len - 1) :]
            self._frames = deque([first] + rest)
        elif len(self._frames) > self.max_len:
            self._frames.popleft()
        if len(self._frames) < self.min_len:
            return
        frames = list(self._frames)
        if self.pad_to_max and len(frames) < self.max_len:
            frames = frames + [frames[-1]] * (self.max_len - len(frames))
        stacked = np.concatenate(frames, axis=0)  # [T,3,H,W]
        if self.layout == "ntchw":
            clip = stacked[None]
        elif self.layout == "ncthw":
            clip = np.transpose(stacked[None], (0, 2, 1, 3, 4))
        elif self.layout == "nthwc":
            clip = np.transpose(stacked[None], (0, 1, 3, 4, 2))
        else:
            raise ValueError(self.layout)
        clip = np.ascontiguousarray(clip.astype(np.float32))
        if self.minus_one_one:
            clip = clip / 127.5 - 1.0
        op_output.emit(np.ascontiguousarray(clip.astype(np.float32)), "clip")


class PadQueriesOp(Operator):
    """Pad native-xy queries to N=64 and scale into the ONNX frame.

    CoTracker / LocoTrack: (t, x, y) with t=0. PIPS: repeated xy on S.
    """

    def __init__(
        self,
        fragment,
        *args,
        queries: np.ndarray,
        max_points: int = 64,
        native_hw: tuple,
        model_hw: tuple,
        layout: str = "xy",
        window: int = 1,
        **kwargs,
    ):
        pts = np.asarray(queries, dtype=np.float32)
        n = int(max_points)
        padded = np.zeros((n, 2), dtype=np.float32)
        if pts.shape[0]:
            take = pts[:n]
            padded[: take.shape[0]] = take
            padded[take.shape[0] :] = take[-1]
        nat_h, nat_w = (int(native_hw[0]), int(native_hw[1]))
        mod_h, mod_w = (int(model_hw[0]), int(model_hw[1]))
        sx = (mod_w - 1) / max(nat_w - 1, 1)
        sy = (mod_h - 1) / max(nat_h - 1, 1)
        xy = padded.copy()
        xy[:, 0] *= sx
        xy[:, 1] *= sy
        if layout == "txy":
            q = np.zeros((1, n, 3), dtype=np.float32)
            q[0, :, 1] = xy[:, 0]
            q[0, :, 2] = xy[:, 1]
            self._tensor = q
        elif layout == "tyx":
            q = np.zeros((1, n, 3), dtype=np.float32)
            q[0, :, 1] = xy[:, 1]
            q[0, :, 2] = xy[:, 0]
            self._tensor = q
        elif layout == "trajs":
            trajs = np.zeros((1, int(window), n, 2), dtype=np.float32)
            trajs[:, :, :, :] = xy
            self._tensor = trajs
        else:
            self._tensor = xy[None]
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("tick")
        spec.output("queries")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("tick")
        op_output.emit(self._tensor, "queries")


class SampleFlowOp(Operator):
    """Sample dense flow at query points and return STIR-scale xy."""

    def __init__(self, fragment, *args, scale: int, max_points: int, **kwargs):
        self.scale = float(scale)
        self.max_points = int(max_points)
        self.queries: Optional[np.ndarray] = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("flow")
        spec.input("queries")
        spec.output("coords")

    def compute(self, op_input, op_output, context):
        flow = np.asarray(op_input.receive("flow"))  # [1,2,H,W]
        queries = np.asarray(op_input.receive("queries"), dtype=np.float32)
        queries_scaled = queries / self.scale
        _, _, height, width = flow.shape
        xs = queries_scaled[:, 0]
        ys = queries_scaled[:, 1]
        x0 = np.clip(np.floor(xs).astype(np.int32), 0, width - 2)
        y0 = np.clip(np.floor(ys).astype(np.int32), 0, height - 2)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = xs - x0
        wy = ys - y0
        flow_x = flow[0, 0]
        flow_y = flow[0, 1]
        sampled_x = (
            flow_x[y0, x0] * (1 - wx) * (1 - wy)
            + flow_x[y0, x1] * wx * (1 - wy)
            + flow_x[y1, x0] * (1 - wx) * wy
            + flow_x[y1, x1] * wx * wy
        )
        sampled_y = (
            flow_y[y0, x0] * (1 - wx) * (1 - wy)
            + flow_y[y0, x1] * wx * (1 - wy)
            + flow_y[y1, x0] * (1 - wx) * wy
            + flow_y[y1, x1] * wx * wy
        )
        coords = np.stack([xs + sampled_x, ys + sampled_y], axis=-1) * self.scale
        op_output.emit(coords.astype(np.float32), "coords")


class MFTChainOp(Operator):
    """Deprecated consecutive-pair stub.

    Official MFT needs RAFT-OU(t-delta, t) for deltas in {1,2,4,...,inf}, then
    ``chain_results`` + sigma/occlusion selection. That cannot be done from a
    single consecutive flow. Use ``MFTHostOp`` / ``OnnxMFTTracker`` instead.
    """

    def __init__(self, fragment, *args, occlusion_threshold: float = 0.02, **kwargs):
        self.occlusion_threshold = float(occlusion_threshold)
        self.memory: Dict[int, Dict] = {}
        self.frame_idx = 0
        self.deltas = [1, 2, 4, 8, 16, 32]
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("flow")
        spec.input("occlusion")
        spec.input("sigma")
        spec.output("flow")
        spec.output("occlusion")

    def compute(self, op_input, op_output, context):
        flow = np.asarray(op_input.receive("flow"))
        occlusion = np.asarray(op_input.receive("occlusion"))
        sigma = np.asarray(op_input.receive("sigma"))
        self.frame_idx += 1
        self.memory[self.frame_idx] = {"flow": flow, "occlusion": occlusion, "sigma": sigma}
        keep = {self.frame_idx - d for d in self.deltas}
        keep.add(self.frame_idx)
        self.memory = {k: v for k, v in self.memory.items() if k in keep}
        op_output.emit(flow, "flow")
        op_output.emit(occlusion, "occlusion")


class MetricsSinkOp(Operator):
    """CUDA-synced per-frame latency, matching STIR LatencyLogger JSON."""

    def __init__(self, fragment, *args, sequence_name: str, out_dir: str, **kwargs):
        self.sequence_name = sequence_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.latencies: List[float] = []
        self._t0: Optional[float] = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def mark_start(self):
        self._t0 = time.perf_counter()

    def compute(self, op_input, op_output, context):
        payload = op_input.receive("in")
        now = time.perf_counter()
        if self._t0 is not None:
            self.latencies.append((now - self._t0) * 1000.0)
        self._t0 = now
        op_output.emit(payload, "out")

    def stop(self):
        arr = np.asarray(self.latencies, dtype=np.float64)
        stats = {}
        if arr.size:
            stats = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "P95": float(np.percentile(arr, 95)),
                "P99": float(np.percentile(arr, 99)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "len": int(arr.size),
            }
        (self.out_dir / f"latencies_{self.sequence_name}.json").write_text(json.dumps(self.latencies))
        (self.out_dir / f"latency_stats_{self.sequence_name}.json").write_text(json.dumps({"all": stats}, indent=2))
