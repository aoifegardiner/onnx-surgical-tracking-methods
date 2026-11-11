"""ONNX Runtime host for scale-1 TAPNext++, matching ``TAPNextPP.track_frame``.

Each query count has its own static graph (see ``tapnextpp_online.py``). The
session for the current count stays on GPU: recurrent state is IO-bound, and
only the tracks come back to the host. Preprocess is the official one: RGB
uint8, bilinear resize to 256, normalize to [-1, 1], queries in ``[t, y, x]``.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MODEL_SIZE = 256
DEPTH = 12
LRU_WIDTH = 768
CONV_CACHE = 3
VIDEO_TOKENS = 32 * 32
MAX_POINTS = 34

INPUT_NAMES = ["video", "query_points", "step", "rg_lru", "conv1d"]
OUTPUT_NAMES = ["tracks", "visible_logits", "new_rg_lru", "new_conv1d", "new_step"]
STATE_PAIRS = (("rg_lru", "new_rg_lru"), ("conv1d", "new_conv1d"), ("step", "new_step"))


def step_path(onnx_dir: Path, num_points: int) -> Path:
    return Path(onnx_dir) / ("tapnextpp_online_step_n%d.onnx" % int(num_points))


def _make_session(onnx_path: Path, cuda_graph: bool):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        cuda_opts = {
            "device_id": "0",
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": "1",
        }
        if cuda_graph:
            cuda_opts["enable_cuda_graph"] = "1"
        providers.append(("CUDAExecutionProvider", cuda_opts))
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(onnx_path), so, providers=providers or ["CPUExecutionProvider"])


class OnnxTapNextPP:
    """One online frame per call. Load the graph for this sequence's point count first."""

    def __init__(self, onnx_dir: str | Path, cuda_graph: bool = False):
        self.onnx_dir = Path(onnx_dir)
        self.cuda_graph = bool(cuda_graph)
        self.sess = None
        self.num_points: Optional[int] = None
        self.providers: List[str] = []
        self._inputs: Dict[str, object] = {}
        self._outputs: Dict[str, object] = {}
        self._binding = None
        self._disp_hw: Optional[Tuple[int, int]] = None
        self.single = os.environ.get("TAPNEXTPP_SINGLE", "") == "1"
        self._real_n: Optional[int] = None
        self._input_names = list(INPUT_NAMES)
        if self.single:
            self._input_names.append("point_valid")

    def close(self) -> None:
        self._binding = None
        self._inputs = {}
        self._outputs = {}
        self.sess = None
        self.num_points = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def ensure(self, num_points: int) -> None:
        """Load and warm the static graph for this query count. Call outside the timer."""
        num_points = int(num_points)
        if self.single:
            if num_points < 1 or num_points > MAX_POINTS:
                raise ValueError("single TAPNext++ graph covers 1..%d points, got %d" % (MAX_POINTS, num_points))
            if self.sess is not None:
                return
            path = self.onnx_dir / "tapnextpp_online_step.onnx"
            graph_points = MAX_POINTS
        else:
            if self.sess is not None and self.num_points == num_points:
                return
            path = step_path(self.onnx_dir, num_points)
            graph_points = num_points
        if not path.is_file():
            raise FileNotFoundError("no TAPNext++ ONNX for %d points: %s" % (graph_points, path))
        self.close()
        try:
            self.sess = _make_session(path, cuda_graph=self.cuda_graph)
        except Exception as exc:
            if not self.cuda_graph:
                raise
            print("TAPNext++ CUDA graph session failed (%s); loading without it" % exc, flush=True)
            self.cuda_graph = False
            self.sess = _make_session(path, cuda_graph=False)
        self.providers = list(self.sess.get_providers())
        if "CUDAExecutionProvider" not in self.providers:
            self.close()
            raise RuntimeError("TAPNext++ ONNX CUDA provider did not load: %s" % self.providers)
        self.num_points = graph_points
        print("TAPNext++ graph %s  slots=%d" % (path.name, graph_points), flush=True)
        self._alloc()
        try:
            self._warmup()
        except Exception as exc:
            print("TAPNext++ CUDA graph capture failed (%s); rerunning without it" % exc, flush=True)
            self.close()
            self.cuda_graph = False
            self.sess = _make_session(path, cuda_graph=False)
            self.providers = list(self.sess.get_providers())
            self.num_points = graph_points
            self._alloc()

    def _alloc(self) -> None:
        import torch

        q = int(self.num_points)
        tokens = VIDEO_TOKENS + q
        shapes = {
            "video": (1, 1, MODEL_SIZE, MODEL_SIZE, 3),
            "query_points": (1, q, 3),
            "step": (),
            "point_valid": (MAX_POINTS,),
            "rg_lru": (DEPTH, tokens, LRU_WIDTH),
            "conv1d": (DEPTH, tokens, CONV_CACHE, LRU_WIDTH),
            "tracks": (1, 1, q, 2),
            "visible_logits": (1, 1, q, 1),
            "new_rg_lru": (DEPTH, tokens, LRU_WIDTH),
            "new_conv1d": (DEPTH, tokens, CONV_CACHE, LRU_WIDTH),
            "new_step": (),
        }
        self._inputs = {
            name: torch.empty(shapes[name], dtype=torch.float32, device="cuda") for name in self._input_names
        }
        self._outputs = {
            name: torch.empty(shapes[name], dtype=torch.float32, device="cuda") for name in OUTPUT_NAMES
        }
        for tensor in self._inputs.values():
            tensor.zero_()
        self._check_shapes(shapes)
        self._binding = self.sess.io_binding()
        for name, tensor in self._inputs.items():
            self._binding.bind_input(
                name, "cuda", 0, np.float32, list(tensor.shape), tensor.data_ptr()
            )
        for name, tensor in self._outputs.items():
            self._binding.bind_output(
                name, "cuda", 0, np.float32, list(tensor.shape), tensor.data_ptr()
            )

    def _check_shapes(self, shapes: dict) -> None:
        def _dims(value):
            dims = []
            for dim in value:
                if isinstance(dim, str) or dim is None:
                    dims.append(None)
                else:
                    dims.append(int(dim))
            return dims

        for name, tensor in {**self._inputs, **self._outputs}.items():
            got = _dims(self._ort_shape(name))
            want = [int(d) for d in tensor.shape]
            if len(got) != len(want):
                raise RuntimeError("ONNX %s rank %s does not match host %s" % (name, got, want))
            for axis, (left, right) in enumerate(zip(got, want)):
                if left is not None and left != right:
                    raise RuntimeError(
                        "ONNX %s axis %d is %s, host allocated %s" % (name, axis, got, want)
                    )

    def _ort_shape(self, name: str):
        for value in list(self.sess.get_inputs()) + list(self.sess.get_outputs()):
            if value.name == name:
                return value.shape
        raise KeyError(name)

    def _warmup(self) -> None:
        import torch

        self._inputs["video"].zero_()
        self._inputs["query_points"].zero_()
        self._inputs["step"].zero_()
        self._inputs["rg_lru"].zero_()
        self._inputs["conv1d"].zero_()
        if "point_valid" in self._inputs:
            self._inputs["point_valid"].zero_()
        self.sess.run_with_iobinding(self._binding)
        torch.cuda.synchronize()
        self._inputs["rg_lru"].zero_()
        self._inputs["conv1d"].zero_()
        self._inputs["step"].zero_()

    def reset(self, points_xy: np.ndarray, height: int, width: int) -> None:
        """Place queries on frame 0 and clear the recurrent cache."""
        import torch

        points_xy = np.asarray(points_xy, dtype=np.float32)
        real_n = int(points_xy.shape[0])
        if self.single:
            if real_n < 1 or real_n > MAX_POINTS:
                raise ValueError("single TAPNext++ graph covers 1..%d points, got %d" % (MAX_POINTS, real_n))
            self._real_n = real_n
            padded = np.zeros((MAX_POINTS, 2), dtype=np.float32)
            padded[:real_n] = points_xy
            points_xy = padded
        elif real_n != self.num_points:
            raise ValueError("session is Q=%s, got %s points" % (self.num_points, real_n))
        model_xy = points_xy * np.array(
            [MODEL_SIZE / float(width), MODEL_SIZE / float(height)], dtype=np.float32
        )
        queries = np.zeros((1, points_xy.shape[0], 3), dtype=np.float32)
        queries[0, :, 1] = model_xy[:, 1]
        queries[0, :, 2] = model_xy[:, 0]
        self._inputs["query_points"].copy_(torch.from_numpy(queries))
        if self.single:
            valid = np.zeros((MAX_POINTS,), dtype=np.float32)
            valid[:real_n] = 1.0
            self._inputs["point_valid"].copy_(torch.from_numpy(valid))
        self._inputs["rg_lru"].zero_()
        self._inputs["conv1d"].zero_()
        self._inputs["step"].zero_()
        self._disp_hw = (int(height), int(width))

    def step_rgb(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One frame. ``frame_rgb`` is HWC uint8. Returns display xy and visibility."""
        import torch
        import torch.nn.functional as F

        if self._disp_hw is None:
            raise RuntimeError("reset() before step_rgb()")
        image = torch.from_numpy(np.ascontiguousarray(frame_rgb)).to("cuda", non_blocking=True)
        image = image.permute(2, 0, 1).unsqueeze(0).float()
        image = F.interpolate(image, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False)
        image = image.div_(127.5).sub_(1.0)
        video = image.permute(0, 2, 3, 1).unsqueeze(0)
        self._inputs["video"].copy_(video)
        self.sess.run_with_iobinding(self._binding)
        torch.cuda.synchronize()
        for prev, new in STATE_PAIRS:
            self._inputs[prev].copy_(self._outputs[new])
        tracks = self._outputs["tracks"][0, 0].detach().float()
        visible = self._outputs["visible_logits"][0, 0, :, 0].detach().float()
        xy = torch.stack((tracks[:, 1], tracks[:, 0]), dim=-1)
        height, width = self._disp_hw
        scale = torch.tensor(
            [width / MODEL_SIZE, height / MODEL_SIZE], dtype=torch.float32, device=xy.device
        )
        positions = (xy * scale).cpu().numpy().astype(np.float32)
        visible = (visible > 0).cpu().numpy().astype(np.float32)
        if self.single:
            real_n = int(self._real_n)
            return positions[:real_n], visible[:real_n]
        return positions, visible

    def track_sequence(self, frames_rgb: list, points_xy: np.ndarray):
        """RGB uint8 frames at the camera resolution. Tracks are [N, T, 2]."""
        import torch

        points_xy = np.asarray(points_xy, dtype=np.float32)
        if not frames_rgb:
            raise RuntimeError("empty clip")
        self.ensure(int(points_xy.shape[0]))
        height, width = frames_rgb[0].shape[:2]
        self.reset(points_xy, height, width)
        tracks = []
        visibility = []
        latencies = []
        for frame in frames_rgb:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            positions, visible = self.step_rgb(frame)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            tracks.append(positions)
            visibility.append(visible)
        return np.stack(tracks, axis=1), np.stack(visibility, axis=0), latencies
