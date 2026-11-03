"""Keep-first growing LocoTrack around an ORT session (dynamic T).

Matches ``LocoTrackSTIR``: queries stay at t=0, each step re-runs on
frames[0:t+1], cap ``max_frames`` (default 300). Holoscan InferenceOp cannot
take a changing clip length, so this host calls ONNX Runtime directly.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from export_onnx.common import LOCOTRACK_MAX_FRAMES, keep_first_growing_clip

PAD_N = 64


def _ort_session(onnx_path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                },
            )
        )
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(onnx_path), so, providers=providers or ["CPUExecutionProvider"])


def _onnx_static_time(sess):
    for inp in sess.get_inputs():
        if inp.name in ("window", "video") and len(inp.shape) >= 2:
            t = inp.shape[1]
            if t is None or isinstance(t, str):
                return None
            return int(t)
    return None


def _pad_xy(pts: np.ndarray, n: int = PAD_N) -> np.ndarray:
    if pts.shape[0] > n:
        return pts[:n].copy()
    out = np.zeros((n, 2), dtype=np.float32)
    if pts.shape[0]:
        out[: pts.shape[0]] = pts
        out[pts.shape[0] :] = pts[-1]
    return out


class OnnxLocoTrackTracker:
    """Streaming LocoTrack: one ORT call per new frame on the growing clip."""

    def __init__(self, onnx_path: Path, extra: Optional[dict] = None):
        extra = dict(extra or {})
        self.sess = _ort_session(onnx_path)
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        self.feat_sess = None
        self.feature_chunk = int(extra.get("feature_chunk") or 0)
        if any(str(name).startswith("feat_") for name in self.input_names):
            feat_path = extra.get("feature_onnx") or str(Path(onnx_path).with_name(Path(onnx_path).stem + "_feats.onnx"))
            self.feat_sess = _ort_session(feat_path)
            if self.feature_chunk < 1:
                shape = self.sess.get_inputs()[0].shape
                from export_onnx.common import locotrack_feature_chunk

                height = int(shape[3]) if isinstance(shape[3], int) else 256
                width = int(shape[4]) if isinstance(shape[4], int) else 320
                self.feature_chunk = locotrack_feature_chunk(height, width)
        self.use_window = "window" in self.input_names
        self.pad_n = int(extra.get("N", PAD_N))
        self.max_frames = int(extra.get("max_frames", extra.get("T", LOCOTRACK_MAX_FRAMES)))
        static_t = _onnx_static_time(self.sess)
        self.static_t = static_t
        if static_t is not None:
            self.max_frames = min(self.max_frames, static_t)
        self.latencies_ms: List[float] = []
        self.frames: List[np.ndarray] = []
        self.n_real = 0
        self.queries = None
        self.last_coords: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.frames = []
        self.queries = None
        self.last_coords = None
        self.n_real = 0
        self.latencies_ms = []

    def _make_queries(self, xy_work: np.ndarray) -> None:
        pts = np.asarray(xy_work, dtype=np.float32).reshape(-1, 2)
        self.n_real = int(pts.shape[0])
        padded = _pad_xy(pts, n=self.pad_n)
        q = np.zeros((1, self.pad_n, 3), dtype=np.float32)
        q[0, :, 0] = 0.0
        if self.use_window:
            q[0, :, 1] = padded[:, 0]
            q[0, :, 2] = padded[:, 1]
        else:
            q[0, :, 1] = padded[:, 1]
            q[0, :, 2] = padded[:, 0]
        self.queries = q
        self.last_coords = pts.copy()

    def _feeds(self, clip: Sequence[np.ndarray]) -> dict:
        sl = list(clip)
        if self.static_t is not None and len(sl) < self.static_t:
            sl = sl + [sl[-1]] * (self.static_t - len(sl))
        if self.use_window:
            vid = np.stack(sl, axis=0)[None].astype(np.float32)
            return {"window": np.ascontiguousarray(vid), "queries": self.queries}
        vid = np.stack([np.transpose(f, (1, 2, 0)) for f in sl], axis=0)[None]
        vid = (vid / 127.5 - 1.0).astype(np.float32)
        return {"video": np.ascontiguousarray(vid), "query_points": self.queries}

    def push_frame(self, frame_nchw: np.ndarray, queries_xy_work: Optional[np.ndarray] = None) -> np.ndarray:
        """Append one NCHW RGB frame in [0, 255]. Return current xy in model pixels."""
        frame = np.asarray(frame_nchw, dtype=np.float32)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.shape[0] != 3:
            raise ValueError("expected NCHW or CHW RGB, got %s" % (frame.shape,))
        if self.queries is None:
            if queries_xy_work is None:
                raise RuntimeError("LocoTrack needs query xy on the first frame")
            self._make_queries(queries_xy_work)
        self.frames.append(np.ascontiguousarray(frame))
        if len(self.frames) < 2:
            return self.last_coords.copy()
        t = len(self.frames) - 1
        sl = keep_first_growing_clip(self.frames, t, self.max_frames)
        feeds = self._feeds(sl)
        if self.feat_sess is not None:
            from export_onnx.common import assemble_locotrack_feeds

            feeds = assemble_locotrack_feeds(
                self.sess, self.feat_sess, feeds["window"], feeds["queries"], self.feature_chunk
            )
        t0 = time.perf_counter()
        tracks = self.sess.run(None, feeds)[0]
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        pred = tracks[0, : self.n_real, -1].astype(np.float32)
        self.last_coords = pred
        return pred.copy()
