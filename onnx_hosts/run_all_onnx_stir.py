#!/usr/bin/env python3
"""Evaluate the published STIR ONNX graphs: endpoint accuracy + per-call latency.

Writes the same JSON family as flow2d.py under --outdir (typically
<output-root>/<METHOD>/scale<N>/). One process per method/scale so GPU memory
is released between jobs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from onnx_hosts.common import LOCOTRACK_MAX_FRAMES, keep_first_growing_clip


def _prepend_nvidia_libs() -> None:
    """onnxruntime-gpu 1.23 needs CUDA 12 / cuDNN 9 from pip nvidia-* wheels."""
    import ctypes

    site = Path(sys.prefix) / "lib" / ("python%d.%d" % (sys.version_info.major, sys.version_info.minor))
    nvidia = site / "site-packages" / "nvidia"
    extra = [str(p) for p in nvidia.glob("*/lib") if p.is_dir()]
    if extra:
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(extra + ([prev] if prev else []))
    for name in (
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libnvrtc.so.12",
        "libnvJitLink.so.12",
        "libcufft.so.11",
        "libcurand.so.10",
        "libcusolver.so.11",
        "libcusparse.so.12",
    ):
        hits = list(nvidia.glob("*/lib/" + name))
        if hits:
            try:
                ctypes.CDLL(str(hits[0]), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_prepend_nvidia_libs()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ONNX_ROOT = ROOT / "models"
_stir = os.environ.get("STIR_DATA", "").strip()
if _stir:
    DEFAULT_DATA = Path(_stir)
elif (ROOT / "data" / "STIRChallenge_2024").is_dir():
    DEFAULT_DATA = ROOT / "data" / "STIRChallenge_2024"
else:
    DEFAULT_DATA = Path("data/STIRChallenge_2024")

METHODS = (
    "RAFT",
    "WAFT",
    "MFT",
    "ALLTRACKER",
    "PIPS",
    "LOCOTRACK",
    "COTRACKER",
    "LITETRACKER",
    "TAPNEXTPP",
)
SCALED_METHODS = {"RAFT", "WAFT", "MFT", "ALLTRACKER", "PIPS", "LOCOTRACK"}
FULLRES_METHODS = {"COTRACKER", "LITETRACKER", "COTRACKER_ONLINE", "TAPNEXTPP"}
PAD_N = 64
STIR_NATIVE_HW = (1024, 1280)
THRESHOLDS_PX = [4, 8, 16, 32, 64]


def load_meta(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def loadimcv(framename: Path) -> np.ndarray:
    frameim = cv2.imread(str(framename))
    if frameim is None:
        raise RuntimeError("cannot read %s" % framename)
    frameim = cv2.cvtColor(frameim, cv2.COLOR_BGR2RGB)
    return (frameim / 255.0).astype(np.float32)


def centers_from_seg(im: np.ndarray) -> np.ndarray:
    if im.ndim == 3:
        gray = cv2.cvtColor((np.clip(im, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        centers.append([x + w // 2, y + h // 2])
    return np.asarray(centers, dtype=np.float32).reshape(-1, 2)


def list_sequences(datadir: Path):
    """STIR 2D eval: left clips only (60 sequences in STIRChallenge_2024)."""
    root = Path(datadir)
    seqs = sorted(p for p in root.glob("*/left*/seq*") if p.is_dir())
    out = []
    for seq in seqs:
        if (seq / "segmentation" / "icgstartseg.png").is_file() and list((seq / "frames").glob("*.mp4")):
            out.append(seq)
    return out


def preprocess_bgr(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    chw = np.transpose(rgb.astype(np.float32), (2, 0, 1))
    return chw[None]


def bilinear_sample_flow(flow: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if flow.ndim == 3 and flow.shape[0] == 2:
        fx, fy = flow[0], flow[1]
    elif flow.ndim == 3 and flow.shape[-1] == 2:
        fx, fy = flow[..., 0], flow[..., 1]
    else:
        raise ValueError("unexpected flow shape %s" % (flow.shape,))
    h, w = fx.shape
    x = np.clip(pts[:, 0], 0.0, w - 1.0)
    y = np.clip(pts[:, 1], 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0

    def samp(img):
        ia = img[y0, x0]
        ib = img[y0, x1]
        ic = img[y1, x0]
        id_ = img[y1, x1]
        return (ia * (1 - wx) + ib * wx) * (1 - wy) + (ic * (1 - wx) + id_ * wx) * wy

    return np.stack([samp(fx), samp(fy)], axis=-1).astype(np.float32)


def nearest_distances(pred: np.ndarray, end: np.ndarray) -> np.ndarray:
    from scipy.spatial import KDTree

    if pred.size == 0 or end.size == 0:
        return np.zeros((0,), dtype=np.float64)
    tree = KDTree(end)
    dist, _ = tree.query(pred)
    return np.atleast_1d(np.asarray(dist, dtype=np.float64))


def calculate_accuracy(distances) -> dict:
    distances = np.asarray(distances, dtype=float)
    num = int(distances.size)
    by_thr = {}
    accs = []
    for thr in THRESHOLDS_PX:
        val = float("nan") if num == 0 else float(np.sum(distances <= thr) / num)
        by_thr[str(thr)] = val
        accs.append(val)
    return {
        "thresholds_px": list(THRESHOLDS_PX),
        "accuracy": by_thr,
        "avg": float(np.nanmean(accs)) if accs else float("nan"),
        "std": float(np.nanstd(accs)) if accs else float("nan"),
        "num_points": num,
        "mean_distance": float(np.mean(distances)) if num else float("nan"),
        "median_distance": float(np.median(distances)) if num else float("nan"),
    }


def latency_stats(latencies_ms, skip: int = 10) -> dict:
    arr = np.asarray(latencies_ms, dtype=float)

    def _stats(x):
        if x.size == 0:
            return {
                "median": float("nan"),
                "P95": float("nan"),
                "P99": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "len": 0,
            }
        return {
            "median": float(np.median(x)),
            "P95": float(np.percentile(x, 95)),
            "P99": float(np.percentile(x, 99)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "len": int(x.size),
        }

    skipped = arr[skip:] if arr.size > skip else arr
    return {"all": _stats(arr), "first_%d_frames_skipped" % skip: _stats(skipped)}


STIR_CAMERA_FPS = 26.0
STEREOMIS_CAMERA_FPS = 15.0


def sync_timing() -> None:
    """Match PyTorch LogLatency: drain the GPU before/after the timed block."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def time_pipeline(fn):
    """Run ``fn()`` and return ``(result, wall_s)`` including host + ORT."""
    sync_timing()
    t0 = time.perf_counter()
    out = fn()
    sync_timing()
    return out, time.perf_counter() - t0


TRACK_PIPELINE_SCOPE = "host+ORT track loop; frames preloaded; no decode; no Holoscan"
AGX_PIPELINE_SCOPE = (
    "after ONNX load until sequence metrics finish; CUDA sync at the start boundary and the end; "
    "decode, resize/preprocess, host+ORT track, end-of-sequence metrics; model load outside; no Holoscan"
)


def pipeline_record(n_frames: int, wall_s: float, camera_fps: float, scope: str = None) -> dict:
    """Frames/sec of the recorded wall clock, not sess.run alone.

    ``n_pairs = n_frames - 1`` is the number of camera ticks after the first
    frame. ``pipeline_fps`` is how many of those ticks finish per second of
    ``wall_s``. Real-time means ``pipeline_fps >= camera_fps``.

    Default scope is the track loop on frames already in memory. ``scope``
    ``AGX_PIPELINE_SCOPE`` is the longer clock: decode, resize, tracking, and
    end-of-sequence metrics, with model load outside.
    """
    n_frames = int(n_frames)
    n_pairs = max(n_frames - 1, 0)
    wall_s = float(wall_s)
    camera_fps = float(camera_fps)
    budget_ms = 1000.0 / camera_fps if camera_fps > 0 else None
    if n_pairs > 0 and wall_s > 0:
        ms_pair = 1000.0 * wall_s / n_pairs
        fps = n_pairs / wall_s
    else:
        ms_pair = None
        fps = None
    return {
        "scope": scope or TRACK_PIPELINE_SCOPE,
        "n_frames": n_frames,
        "n_pairs": n_pairs,
        "wall_s": wall_s,
        "pipeline_ms_per_pair": ms_pair,
        "pipeline_fps": fps,
        "camera_fps": camera_fps,
        "budget_ms_pair": budget_ms,
        "meets_realtime": (ms_pair <= budget_ms) if ms_pair is not None and budget_ms is not None else None,
    }


def aggregate_pipeline(records: dict) -> dict:
    recs = [v for v in records.values() if isinstance(v, dict) and v.get("n_pairs") is not None]
    n_pairs = int(sum(int(r["n_pairs"]) for r in recs))
    wall_s = float(sum(float(r["wall_s"]) for r in recs))
    camera_fps = float(recs[0]["camera_fps"]) if recs else STIR_CAMERA_FPS
    n_frames = int(sum(int(r["n_frames"]) for r in recs))
    scope = next((r.get("scope") for r in recs if r.get("scope")), None)
    agg = pipeline_record(2 if n_pairs else 0, wall_s, camera_fps, scope=scope)
    agg["n_frames"] = n_frames
    agg["n_pairs"] = n_pairs
    agg["n_sequences"] = len(recs)
    agg["wall_s"] = wall_s
    if n_pairs > 0 and wall_s > 0:
        agg["pipeline_ms_per_pair"] = 1000.0 * wall_s / n_pairs
        agg["pipeline_fps"] = n_pairs / wall_s
        agg["budget_ms_pair"] = 1000.0 / camera_fps
        agg["meets_realtime"] = agg["pipeline_ms_per_pair"] <= agg["budget_ms_pair"]
    else:
        agg["pipeline_ms_per_pair"] = None
        agg["pipeline_fps"] = None
        agg["meets_realtime"] = None
    agg["camera_fps"] = camera_fps
    return agg


class LocoTrackSessions:
    """Feature encoder (chunked) plus the tracker graph.

    The tracker session is rebuilt as the clip grows. A single session that
    steps T by one each frame keeps every previous allocation hole, and the
    CUDA arena hits 40 GB while a fresh forward at that T still fits.
    """

    def __init__(self, track, feats, chunk: int, onnx_path=None):
        self.track = track
        self.feats = feats
        self.chunk = int(chunk)
        self.onnx_path = onnx_path
        self._built_for = 0

    def get_inputs(self):
        return self.track.get_inputs()

    def get_providers(self):
        return self.track.get_providers()

    def _reopen_track(self):
        old = self.track
        self.track = None
        del old
        gc.collect()
        self.track = _ort_session(self.onnx_path)

    def run(self, _names, feeds):
        from onnx_hosts.common import assemble_locotrack_feeds

        window = feeds.get("window")
        if window is None or self.feats is None:
            return self.track.run(None, feeds)
        length = int(window.shape[1])
        # Each larger clip leaves a hole the next size cannot reuse. Rebuild the
        # tracker whenever T grows past 16 so the arena matches this clip.
        # Once the keep-first cap holds T fixed, the same session is reused.
        if (
            self.onnx_path is not None
            and self._built_for
            and length != self._built_for
            and length >= 16
        ):
            self._reopen_track()
        self._built_for = length
        full = assemble_locotrack_feeds(self.track, self.feats, window, feeds["queries"], self.chunk)
        return self.track.run(None, full)

    def close(self):
        for sess in (self.track, self.feats):
            if sess is None:
                continue
            close = getattr(sess, "close", None)
            if close is not None:
                close()


def _open_locotrack(onnx_path: Path, extra: dict):
    extra = extra or {}
    track = _ort_session(onnx_path)
    feat_path = extra.get("feature_onnx")
    feats = None
    if feat_path:
        feat_path = Path(feat_path)
        if not feat_path.is_file():
            feat_path = Path(onnx_path).with_name(feat_path.name)
        if not feat_path.is_file() and not feat_path.is_absolute():
            feat_path = ROOT / feat_path
        feats = _ort_session(feat_path)
    chunk = int(extra.get("feature_chunk") or 0)
    if feats is not None and chunk < 1:
        shape = track.get_inputs()[0].shape
        height = int(shape[3]) if isinstance(shape[3], int) else 256
        width = int(shape[4]) if isinstance(shape[4], int) else 320
        from onnx_hosts.common import locotrack_feature_chunk

        chunk = locotrack_feature_chunk(height, width)
    return LocoTrackSessions(track, feats, chunk, onnx_path=onnx_path)


def _ort_session(onnx_path: Path):
    """Build an ORT session with a CUDA arena that can be dropped between clips.

    CoTracker-3 online is ~11k nodes with unrolled GridSample/Einsum/ReduceSum.
    The default BFC arena grows to the peak of the longest clip and then
    fragments; the next sequence fails to allocate ~384 MiB and every later
    clip dies. kSameAsRequested + no mem-pattern, plus a fresh session per
    CoTracker sequence, returns that memory to the driver.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False
    # Growing clips allocate a larger buffer every frame. Without shrinkage the
    # CUDA arena keeps the holes and a later ~700 MB alloc fails while a fresh
    # T=300 forward still fits.
    so.add_session_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    # DEFAULT on cuDNN 9 selects a fallback conv (~4x slower).
                    # HEURISTIC stays on the fast CUDA kernel without a large workspace.
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                },
            )
        )
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(str(onnx_path), so, providers=providers or ["CPUExecutionProvider"])
    return sess


def _release_ort_session(sess):
    if sess is None:
        return None
    close = getattr(sess, "close", None)
    if close is not None:
        close()
    del sess
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return None


def _is_ort_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "failed to allocate",
            "bfc_arena",
            "out of memory",
            "cuda failure",
            "cublas_status_alloc_failed",
        )
    )


def _pad_xy(pts: np.ndarray, n: int = PAD_N) -> np.ndarray:
    if pts.shape[0] > n:
        return pts[:n].copy()
    out = np.zeros((n, 2), dtype=np.float32)
    if pts.shape[0]:
        out[: pts.shape[0]] = pts
        out[pts.shape[0] :] = pts[-1]
    return out


def _load_clip_rgb(video_path: Path, width: int, height: int) -> list:
    """Native RGB uint8. Resize only when the decoded frame is not already that size."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open %s" % video_path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[0] != height or bgr.shape[1] != width:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError("empty video %s" % video_path)
    return frames


def _load_clip_nchw(video_path: Path, width: int, height: int) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open %s" % video_path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(preprocess_bgr(bgr, width, height)[0])
    cap.release()
    if not frames:
        raise RuntimeError("empty video %s" % video_path)
    return frames


def _map_xy(pts: np.ndarray, src_hw, dst_hw) -> np.ndarray:
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    out = np.asarray(pts, dtype=np.float32).copy()
    out[:, 0] *= (dst_w - 1) / max(src_w - 1, 1)
    out[:, 1] *= (dst_h - 1) / max(src_h - 1, 1)
    return out


def _squeeze_flow(flow: np.ndarray) -> np.ndarray:
    while flow.ndim > 3:
        if flow.shape[0] == 1:
            flow = flow[0]
        else:
            flow = flow[-1]
    return flow


def track_dense_flow(sess, frames, pts_full, model_hw, native_hw):
    pts_work = _map_xy(pts_full, native_hw, model_hw)
    latencies = []
    prev = frames[0][None]
    for frame in frames[1:]:
        cur = frame[None]
        t0 = time.perf_counter()
        flow = sess.run(None, {"image1": prev, "image2": cur})[0]
        flow = _squeeze_flow(flow)
        pts_work = pts_work + bilinear_sample_flow(flow, pts_work)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        prev = cur
    return _map_xy(pts_work, model_hw, native_hw), latencies


def track_alltracker(sess, frames, pts_full, model_hw, native_hw, host=None):
    if host is None:
        raise RuntimeError("AllTracker ONNX requires OnnxAllTrackerTracker")
    pts_work = _map_xy(pts_full, native_hw, model_hw)
    pred_nt, lat_ms = host.track_sequence(frames, pts_work, per_frame=True)
    return _map_xy(pred_nt[:, -1], model_hw, native_hw), lat_ms


def _ort_run_dict(sess, feeds):
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, feeds)
    return dict(zip(names, outs))


def track_pips(sess, frames, pts_full, model_hw, native_hw, window=8, extra=None):
    n_real = int(pts_full.shape[0])
    extra = extra or {}
    dim = int(extra.get("feat_dim", 128))
    pts_work = _map_xy(pts_full, native_hw, model_hw)
    latencies = []
    i = 0
    n = len(frames)
    input_names = {inp.name for inp in sess.get_inputs()}
    feats_online = "previous_feats1" in input_names
    feats_old = "feats1" in input_names
    feats = None
    if feats_online or feats_old:
        z = np.zeros((1, window, PAD_N, dim), dtype=np.float32)
        if feats_online:
            feats = {"previous_feats1": z.copy(), "previous_feats2": z.copy(), "previous_feats4": z.copy()}
        else:
            feats = {"feats1": z.copy(), "feats2": z.copy(), "feats4": z.copy()}
    stride = max(window - 1, 1)
    while i < n - 1:
        t0 = time.perf_counter()
        sl = frames[i : i + window]
        actual = len(sl)
        if actual < window:
            sl = sl + [sl[-1]] * (window - actual)
        rgbs = np.stack(sl, axis=0)[None].astype(np.float32)
        trajs_e0 = np.zeros((1, window, PAD_N, 2), dtype=np.float32)
        trajs_e0[:, :, :, :] = _pad_xy(pts_work)
        if "window" in input_names:
            feeds = {"window": rgbs, "trajs_e0": trajs_e0}
        else:
            feeds = {"rgbs": rgbs, "trajs_e0": trajs_e0}
        if feats is not None:
            feeds.update(feats)
        outd = _ort_run_dict(sess, feeds)
        trajs = outd["trajs"]
        last = actual - 1
        pts_work = trajs[0, last, :n_real].astype(np.float32)
        if feats_online:
            feats = {
                "previous_feats1": np.ascontiguousarray(outd["new_feats1"]),
                "previous_feats2": np.ascontiguousarray(outd["new_feats2"]),
                "previous_feats4": np.ascontiguousarray(outd["new_feats4"]),
            }
        elif feats_old:
            feats = {
                "feats1": np.ascontiguousarray(outd["feats1_out"]),
                "feats2": np.ascontiguousarray(outd["feats2_out"]),
                "feats4": np.ascontiguousarray(outd["feats4_out"]),
            }
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if actual < window:
            break
        i += stride
    return _map_xy(pts_work, model_hw, native_hw), latencies


def _onnx_static_time(sess):
    """Return the frozen clip length T, or None if the time axis is dynamic."""
    for inp in sess.get_inputs():
        if inp.name in ("window", "video") and len(inp.shape) >= 2:
            t = inp.shape[1]
            if t is None or isinstance(t, str):
                return None
            return int(t)
    return None


def track_locotrack(sess, frames, pts_full, model_hw, native_hw, clip_t=LOCOTRACK_MAX_FRAMES):
    """Keep-first growing clip, same as LocoTrackSTIR (cap clip_t, default 300).

    Pads with the last frame only when the ONNX graph has a static T (older graphs).
    Dynamic-T graphs get the true prefix length each step.
    """
    n_real = int(pts_full.shape[0])
    query_xy = _map_xy(pts_full, native_hw, model_hw)
    latencies = []
    pred = query_xy.copy()
    q = np.zeros((1, PAD_N, 3), dtype=np.float32)
    padded = _pad_xy(query_xy)
    input_names = [i.name for i in sess.get_inputs()]
    use_window = "window" in input_names
    q[0, :, 0] = 0.0
    if use_window:
        q[0, :, 1] = padded[:, 0]
        q[0, :, 2] = padded[:, 1]
    else:
        q[0, :, 1] = padded[:, 1]
        q[0, :, 2] = padded[:, 0]
    static_t = _onnx_static_time(sess)
    max_frames = int(clip_t)
    if static_t is not None:
        max_frames = min(max_frames, static_t)
    for t in range(1, len(frames)):
        t0 = time.perf_counter()
        sl = keep_first_growing_clip(frames, t, max_frames)
        if static_t is not None and len(sl) < static_t:
            sl = sl + [sl[-1]] * (static_t - len(sl))
        if use_window:
            vid = np.stack(sl, axis=0)[None].astype(np.float32)
            feeds = {"window": vid, "queries": q}
        else:
            vid = np.stack([np.transpose(f, (1, 2, 0)) for f in sl], axis=0)[None]
            vid = (vid / 127.5 - 1.0).astype(np.float32)
            feeds = {"video": vid, "query_points": q}
        tracks = sess.run(None, feeds)[0]
        pred = tracks[0, :n_real, -1].astype(np.float32)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return _map_xy(pred, model_hw, native_hw), latencies


def _cotracker_queries(pts_work: np.ndarray, n: int = PAD_N, dtype=np.float32) -> np.ndarray:
    q = np.zeros((1, int(n), 3), dtype=dtype)
    padded = _pad_xy(pts_work, n=int(n))
    q[0, :, 0] = 0.0
    q[0, :, 1] = padded[:, 0]
    q[0, :, 2] = padded[:, 1]
    return q


def _window_video(frames, start: int, window: int):
    sl = frames[start : start + window]
    actual = len(sl)
    if actual < 1:
        raise RuntimeError("empty CoTracker window at %d" % start)
    if actual < window:
        sl = sl + [sl[-1]] * (window - actual)
    video = np.stack(sl, axis=0)[None].astype(np.float32)
    return video, actual


def _online_state_from_out(outd):
    return {
        "previous_coords": np.ascontiguousarray(outd["new_coords"]),
        "previous_visibility": np.ascontiguousarray(outd["new_visibility"]),
        "previous_confidence": np.ascontiguousarray(outd["new_confidence"]),
        "previous_track_features": np.ascontiguousarray(outd["new_track_features"]),
        "previous_support_features": np.ascontiguousarray(outd["new_support_features"]),
    }


def track_cotracker_online(step_sess, frames, pts_full, model_hw, native_hw, window=16, step=8, extra=None):
    n_real = int(pts_full.shape[0])
    pts_work = _map_xy(pts_full, native_hw, model_hw)
    extra = extra or {}
    n = len(frames)
    last_i = n - 1
    latencies = []
    states = None
    win_start = 0
    tracks = None
    seen = set()
    while win_start < n:
        if win_start in seen:
            raise RuntimeError("CoTracker window start did not advance (stuck at %d)" % win_start)
        seen.add(win_start)
        t0 = time.perf_counter()
        video, actual = _window_video(frames, win_start, window)
        tracks, states = _cotracker_forward_all(step_sess, video, pts_work, states, window, extra)
        if last_i < win_start + window:
            local = min(last_i - win_start, actual - 1)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            return _map_xy(tracks[0, local, :n_real], model_hw, native_hw), latencies
        latencies.append((time.perf_counter() - t0) * 1000.0)
        win_start += step
    local = min(window, n) - 1
    return _map_xy(tracks[0, local, :n_real], model_hw, native_hw), latencies


def _cotracker_forward_all(step_sess, video, pts_work, states, window, extra):
    """One online step for the queries that were actually passed. No padding."""
    if hasattr(step_sess, "run_all_points"):
        return step_sess.run_all_points(video, pts_work, states)
    q_dtype = np.float32
    pts_work = np.asarray(pts_work, dtype=np.float32).reshape(-1, 2)
    n = int(pts_work.shape[0])
    if n == 0:
        return np.zeros((1, int(video.shape[1]), 0, 2), dtype=np.float32), []
    levels = int((extra or {}).get("corr_levels", 4))
    dim = int((extra or {}).get("latent_dim", 128))
    support = int((extra or {}).get("support", 49))
    if not states:
        state = {
            "previous_coords": np.zeros((1, window, n, 2), dtype=q_dtype),
            "previous_visibility": np.zeros((1, window, n), dtype=q_dtype),
            "previous_confidence": np.zeros((1, window, n), dtype=q_dtype),
            "previous_track_features": np.zeros((1, levels, 1, n, dim), dtype=q_dtype),
            "previous_support_features": np.zeros((1, levels, support, n, dim), dtype=q_dtype),
        }
    else:
        state = states[0]
    feeds = {"window": video, "queries": _cotracker_queries(pts_work, n=n, dtype=q_dtype)}
    feeds.update(state)
    outd = _ort_run_dict(step_sess, feeds)
    return outd["tracks"][:, :, :n], [_online_state_from_out(outd)]


def track_litetracker(tracker, frames, pts_full):
    import torch

    n_real = int(pts_full.shape[0])
    q = torch.zeros((1, n_real, 3), dtype=torch.float32)
    q[0, :, 1] = torch.from_numpy(pts_full[:, 0])
    q[0, :, 2] = torch.from_numpy(pts_full[:, 1])
    latencies = []
    pred = None
    tracker.reset()
    for frame in frames:
        t0 = time.perf_counter()
        tens = torch.from_numpy(np.ascontiguousarray(frame[None]))
        coords, _vis, _conf = tracker(tens, q)
        pred = coords[0, 0, :n_real].detach().cpu().numpy().astype(np.float32)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return pred, latencies


def resolve_artifact(method: str, scale: int, fp16: bool = False):
    family = ONNX_ROOT / ("COTRACKER" if method.startswith("COTRACKER") else method)
    if method in ("COTRACKER", "COTRACKER_ONLINE"):
        stem = "cotracker_online_step_fp16" if fp16 else "cotracker_online_step"
        step_json = family / ("%s.json" % stem)
        step_onnx = family / ("%s.onnx" % stem)
        if step_json.is_file() and step_onnx.is_file():
            return load_meta(step_json), step_onnx
        return None, None
    if method == "TAPNEXTPP":
        catalog = family / "tapnextpp_online_step.json"
        if catalog.is_file() and (family / "tapnextpp_online_step_n1.onnx").is_file():
            meta = load_meta(catalog)
            meta = dict(meta)
            meta["height"] = STIR_NATIVE_HW[0]
            meta["width"] = STIR_NATIVE_HW[1]
            return meta, catalog
        return None, None
    if method == "LITETRACKER":
        fnet = family / "fnet.onnx"
        if fnet.is_file() and (family / "corr_mlp.onnx").is_file() and (family / "updateformer.onnx").is_file():
            meta = {
                "method": "LITETRACKER",
                "scale": 1,
                "height": STIR_NATIVE_HW[0],
                "width": STIR_NATIVE_HW[1],
                "extra": {"onnx_dir": str(family)},
            }
            return meta, fnet
        return None, None
    stem = "%s_stir_scale%d" % (method.lower(), scale)
    meta_path = family / ("%s.json" % stem)
    onnx_path = family / ("%s.onnx" % stem)
    if meta_path.is_file() and onnx_path.is_file():
        return load_meta(meta_path), onnx_path
    return None, None


def _to_list(arr) -> list:
    return np.asarray(arr, dtype=float).tolist()


def _track_stir(method, sess, tracker, mft_host, pips_host, at_host, frames, start, model_hw, extra):
    if method in ("RAFT", "WAFT"):
        return track_dense_flow(sess, frames, start, model_hw, STIR_NATIVE_HW)
    if method == "MFT":
        pts_work = _map_xy(start, STIR_NATIVE_HW, model_hw)
        pred_nt, lat_ms = mft_host.track_sequence(frames, pts_work)
        return _map_xy(pred_nt[:, -1], model_hw, STIR_NATIVE_HW), lat_ms
    if method == "ALLTRACKER":
        return track_alltracker(sess, frames, start, model_hw, STIR_NATIVE_HW, host=at_host)
    if method == "PIPS":
        pts_work = _map_xy(start, STIR_NATIVE_HW, model_hw)
        pred_nt, lat_ms = pips_host.track_sequence(frames, pts_work)
        return _map_xy(pred_nt[:, -1], model_hw, STIR_NATIVE_HW), lat_ms
    if method == "LOCOTRACK":
        return track_locotrack(
            sess, frames, start, model_hw, STIR_NATIVE_HW,
            clip_t=int(extra.get("max_frames", extra.get("T", LOCOTRACK_MAX_FRAMES))),
        )
    if method in ("COTRACKER", "COTRACKER_ONLINE"):
        return track_cotracker_online(
            sess,
            frames,
            start,
            model_hw,
            STIR_NATIVE_HW,
            window=int(extra.get("window_len", 16)),
            step=int(extra.get("step", 8)),
            extra=extra,
        )
    if method == "LITETRACKER":
        return track_litetracker(tracker, frames, start)
    if method == "TAPNEXTPP":
        traj, _vis, lat_ms = tracker.track_sequence(frames, start)
        return traj[:, -1], lat_ms
    raise RuntimeError(method)


def eval_one(method: str, scale: int, seqs, args):
    meta, onnx_path = resolve_artifact(method, scale, fp16=bool(getattr(args, "fp16", False)))
    if meta is None:
        print("%s scale=%s  SKIP (no STIR ONNX)" % (method, scale), flush=True)
        return 2
    height = int(meta["height"])
    width = int(meta["width"])
    extra = meta.get("extra") or {}
    model_hw = (height, width)
    print(
        "=== %s scale=%s  %s  %dx%d ===" % (method, scale, onnx_path.name if onnx_path else "", height, width),
        flush=True,
    )

    tracker = None
    sess = None
    mft_host = None
    pips_host = None
    at_host = None
    reset_each = method in ("COTRACKER", "COTRACKER_ONLINE")
    if method == "LITETRACKER":
        from onnx_hosts.litetracker_ort import OnnxLiteTracker

        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        host_device = os.environ.get("LITETRACKER_DEVICE") or None
        tracker = OnnxLiteTracker(
            str(ONNX_ROOT / "LITETRACKER"),
            providers=providers or ["CPUExecutionProvider"],
            device=host_device,
        )
        backend = "onnx:" + ",".join(providers or ["CPUExecutionProvider"]) + "+host:" + str(tracker.device)
        max_step = os.environ.get("LITETRACKER_MAX_STEP")
        if max_step:
            backend += "+max_step:" + max_step
    elif method == "TAPNEXTPP":
        from onnx_hosts.tapnextpp_ort import OnnxTapNextPP

        tracker = OnnxTapNextPP(ONNX_ROOT / "TAPNEXTPP")
        backend = "onnx:CUDAExecutionProvider+host:cuda"
    elif method == "MFT":
        from holoscan_agx.mft_pipeline import OnnxMFTTracker

        mft_host = OnnxMFTTracker(onnx_path)
        backend = "holoscan-host:MFT+onnx-RAFT-OU"
    elif method == "PIPS":
        from holoscan_agx.pips_pipeline import OnnxPipsTracker

        pips_host = OnnxPipsTracker(onnx_path, window=int(extra.get("S", 128)), extra=extra)
        backend = "holoscan-host:PIPS+onnx"
    elif method == "ALLTRACKER":
        from onnx_hosts.alltracker_window import OnnxAllTrackerTracker

        extra = dict(extra)
        extra.setdefault("height", height)
        extra.setdefault("width", width)
        at_host = OnnxAllTrackerTracker(onnx_path, window_len=int(extra.get("window_len", 16)), extra=extra)
        backend = "holoscan-host:ALLTRACKER+onnx"
    elif reset_each:
        from onnx_hosts.cotracker_ort import CoTrackerStepRunner, lock_jetson_clocks

        lock_jetson_clocks()
        if onnx_path is None or not Path(onnx_path).is_file():
            print("CoTracker ONNX missing (fp16=%s): %s" % (bool(getattr(args, "fp16", False)), onnx_path), flush=True)
            return 2
        sess = CoTrackerStepRunner(onnx_path)
        backend = "onnx:" + ",".join(sess.get_providers())
        print(
            "CoTracker: one graph, N=%s  dtype=%s"
            % ("dynamic" if sess.n_points is None else sess.n_points, sess.np_dtype.__name__),
            flush=True,
        )
    elif method == "LOCOTRACK":
        sess = _open_locotrack(onnx_path, extra)
        backend = "onnx:" + ",".join(sess.get_providers())
        if sess.feats is not None:
            print("LocoTrack feature chunk=%d" % sess.chunk, flush=True)
    else:
        sess = _ort_session(onnx_path)
        backend = "onnx:" + ",".join(sess.get_providers())
    print("backend %s" % backend, flush=True)
    use_agx = getattr(args, "wall_scope", "track") == "agx"
    if use_agx:
        print("wall clock: %s" % AGX_PIPELINE_SCOPE, flush=True)
        nearest_distances(np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32))

    positionlists = {}
    start_gt = {}
    end_gt = {}
    errorlists = {}
    latencies_by_seq = {}
    pipeline_by_seq = {}
    all_model_d = []
    all_control_d = []
    epe_sum = 0.0
    control_sum = 0.0
    n_ok = 0

    for seq in seqs:
        name = str(seq.resolve())
        rel = str(seq.relative_to(args.datadir))
        start = centers_from_seg(loadimcv(seq / "segmentation" / "icgstartseg.png"))
        end = centers_from_seg(loadimcv(seq / "segmentation" / "icgendseg.png"))
        vids = sorted((seq / "frames").glob("*.mp4"))
        if not vids:
            print("%s  SKIP (no mp4)" % rel, flush=True)
            continue
        if start.shape[0] < 1:
            print("%s  SKIP (no start points)" % rel, flush=True)
            continue
        if method == "TAPNEXTPP":
            tracker.ensure(int(start.shape[0]))
        frames = None
        n_frames = 0
        pred = None
        lat_ms = None
        wall_s = None
        dist = None
        ctrl = None
        try:
            if reset_each:
                sess = _release_ort_session(sess)
                from onnx_hosts.cotracker_ort import CoTrackerStepRunner

                sess = CoTrackerStepRunner(onnx_path)

            def _go():
                return _track_stir(
                    method, sess, tracker, mft_host, pips_host, at_host, frames, start, model_hw, extra
                )

            try:
                if use_agx:
                    sync_timing()
                    t_agx = time.perf_counter()
                    frames = _load_clip_rgb(vids[0], width, height) if method == "TAPNEXTPP" else _load_clip_nchw(vids[0], width, height)
                    n_frames = len(frames)
                    pred, lat_ms = _go()
                    dist = nearest_distances(pred, end)
                    ctrl = nearest_distances(start, end)
                    sync_timing()
                    wall_s = time.perf_counter() - t_agx
                else:
                    frames = _load_clip_rgb(vids[0], width, height) if method == "TAPNEXTPP" else _load_clip_nchw(vids[0], width, height)
                    n_frames = len(frames)
                    (pred, lat_ms), wall_s = time_pipeline(_go)
            except Exception as exc:
                if reset_each and _is_ort_oom(exc):
                    print("%s  OOM, new ORT session and retry once" % rel, flush=True)
                    sess = _release_ort_session(sess)
                    from onnx_hosts.cotracker_ort import CoTrackerStepRunner

                    sess = CoTrackerStepRunner(onnx_path)
                    if use_agx:
                        sync_timing()
                        t_agx = time.perf_counter()
                        if frames is None:
                            frames = _load_clip_rgb(vids[0], width, height) if method == "TAPNEXTPP" else _load_clip_nchw(vids[0], width, height)
                            n_frames = len(frames)
                        pred, lat_ms = _go()
                        dist = nearest_distances(pred, end)
                        ctrl = nearest_distances(start, end)
                        sync_timing()
                        wall_s = time.perf_counter() - t_agx
                    else:
                        (pred, lat_ms), wall_s = time_pipeline(_go)
                else:
                    raise
        except Exception as exc:
            print("%s  ERROR %s: %s" % (rel, type(exc).__name__, exc), flush=True)
            if method == "LOCOTRACK" and _is_ort_oom(exc):
                print("%s  releasing ORT arena after OOM" % rel, flush=True)
                sess = _release_ort_session(sess)
                sess = _open_locotrack(onnx_path, extra)
            if args.stop_on_error:
                raise
            continue
        finally:
            frames = None
            if reset_each:
                sess = _release_ort_session(sess)
            gc.collect()
        if pred is None:
            continue
        if dist is None:
            dist = nearest_distances(pred, end)
            ctrl = nearest_distances(start, end)
        m_epe = float(dist.mean()) if dist.size else float("nan")
        m_ctrl = float(ctrl.mean()) if ctrl.size else float("nan")
        pipe = pipeline_record(
            n_frames,
            wall_s if wall_s is not None else 0.0,
            STIR_CAMERA_FPS,
            scope=AGX_PIPELINE_SCOPE if use_agx else None,
        )
        pipeline_by_seq[name] = pipe
        print(
            "%s  EPE=%.2f px  control=%.2f  frames=%d  pipeline_fps=%.2f  ms/pair=%.1f  realtime=%s  (camera %.0f fps)"
            % (
                rel,
                m_epe,
                m_ctrl,
                n_frames,
                pipe["pipeline_fps"] if pipe["pipeline_fps"] is not None else float("nan"),
                pipe["pipeline_ms_per_pair"] if pipe["pipeline_ms_per_pair"] is not None else float("nan"),
                "yes" if pipe["meets_realtime"] else "no",
                STIR_CAMERA_FPS,
            ),
            flush=True,
        )
        positionlists[name] = _to_list(pred)
        start_gt[name] = _to_list(start)
        end_gt[name] = _to_list(end)
        errorlists[name] = {"endpointerror_control": m_ctrl, "endpointerror_%s" % method: m_epe}
        latencies_by_seq[name] = [float(x) for x in lat_ms]
        all_model_d.extend(dist.tolist())
        all_control_d.extend(ctrl.tolist())
        if np.isfinite(m_epe):
            epe_sum += m_epe
            control_sum += m_ctrl
            n_ok += 1

    if n_ok == 0:
        print("%s scale=%s  no sequences completed" % (method, scale), flush=True)
        return 1

    errorlists["total"] = {
        "mean_endpointerror_control": control_sum / n_ok,
        "mean_endpointerror_%s" % method: epe_sum / n_ok,
    }
    acc = {
        "num_sequences": n_ok,
        "method": method,
        "scale": scale,
        "backend": backend,
        "onnx": str(onnx_path) if onnx_path else extra.get("onnx_dir"),
        "control": calculate_accuracy(all_control_d),
        "model": calculate_accuracy(all_model_d),
    }
    flat_lat = [x for v in latencies_by_seq.values() for x in v]
    stats = latency_stats(flat_lat, skip=args.latency_skip)
    stats["pipeline"] = aggregate_pipeline(pipeline_by_seq)

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.num_seq < 0 else str(args.num_seq)
    scale_tag = "scale%d" % scale
    suffix = args.jsonsuffix
    (args.outdir / ("positions_%s_%s_%s%s.json" % (tag, method, scale_tag, suffix))).write_text(
        json.dumps(positionlists, indent=2)
    )
    (args.outdir / ("gt_positions_start_%s_%s_%s%s.json" % (tag, method, scale_tag, suffix))).write_text(
        json.dumps(start_gt, indent=2)
    )
    (args.outdir / ("gt_positions_end_%s_%s_%s%s.json" % (tag, method, scale_tag, suffix))).write_text(
        json.dumps(end_gt, indent=2)
    )
    (args.outdir / ("endpointerror_%s_%s_%s%s.json" % (tag, method, scale_tag, suffix))).write_text(
        json.dumps(errorlists, indent=2)
    )
    (args.outdir / ("accuracy_%s_%s_%s%s.json" % (tag, method, scale_tag, suffix))).write_text(
        json.dumps(acc, indent=2)
    )
    (args.outdir / ("latencies_%s_%s_%s%s.json" % (tag, method, scale_tag, suffix))).write_text(
        json.dumps(latencies_by_seq, indent=2)
    )
    pipe_payload = dict(pipeline_by_seq)
    pipe_payload["total"] = stats["pipeline"]
    (args.outdir / ("pipeline_%s_%s_%s%s.json" % (tag, method, scale_tag, suffix))).write_text(
        json.dumps(pipe_payload, indent=2)
    )
    (args.outdir / ("latency_stats_%s_%s_%s%s.json" % (tag, method, scale_tag, suffix))).write_text(
        json.dumps(stats, indent=2)
    )
    pipe_tot = stats["pipeline"]
    print(
        "%s  mean EPE=%.2f px  accuracy_avg=%.4f  pipeline_fps=%.2f  ms/pair=%.1f  realtime=%s  seqs=%d"
        % (
            method,
            epe_sum / n_ok,
            acc["model"]["avg"],
            pipe_tot.get("pipeline_fps") or float("nan"),
            pipe_tot.get("pipeline_ms_per_pair") or float("nan"),
            "yes" if pipe_tot.get("meets_realtime") else "no",
            n_ok,
        ),
        flush=True,
    )
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--num_seq", type=int, default=-1, help="-1 = all sequences")
    parser.add_argument("--method", required=True, choices=list(METHODS) + ["COTRACKER_ONLINE"])
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--jsonsuffix", type=str, default="")
    parser.add_argument("--latency-skip", type=int, default=10)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--wall-scope",
        choices=("track", "agx"),
        default="track",
        help="track: host+ORT on preloaded frames. agx: longer clock from decode through end-of-sequence metrics (model load outside).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Load cotracker_online_step_fp16.onnx (Xavier tensor-core graph).",
    )
    args = parser.parse_args()

    method = args.method
    if method in FULLRES_METHODS and args.scale != 1:
        print("%s is full-resolution only; refusing scale=%s" % (method, args.scale), flush=True)
        return 2

    seqs = list_sequences(args.datadir)
    if not seqs:
        print("no STIR sequences under", args.datadir, file=sys.stderr)
        return 1
    if args.num_seq >= 0:
        seqs = seqs[: args.num_seq]
    print("datadir=%s  sequences=%d" % (args.datadir, len(seqs)), flush=True)
    return eval_one(method, args.scale, seqs, args)


if __name__ == "__main__":
    sys.exit(main())
