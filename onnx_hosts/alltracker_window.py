"""Static AllTracker S-window for ONNX, plus a host that matches ``forward_sliding``.

PyTorch ``AllTrackerSTIR`` calls ``Net.forward_sliding`` (window_len=16, stride=8)
on the growing clip every STIR/StereoMIS step. ONNX cannot take a variable T, so
this module runs one window:

    query [B,3,H,W], window [B,S,3,H,W], flows8, visconfs8
        → flows [B,S,2,H,W], visconfs [B,S,2,H,W], flows8_out, visconfs8_out

``query`` is always clip frame 0 (fmap_anchor). ``window`` is frames
``[ind:ind+S]`` after last-frame padding. Host carries 1/8-res flow/visconf
between overlapping windows exactly as ``forward_sliding`` (eval, T>2).

T==2 still uses the pairwise ``model.forward`` graph (``*_pair.onnx``) so the
first STIR step matches PyTorch.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from onnx_hosts.trace_compat import as_int


def padded_hw(height: int, width: int) -> Tuple[int, int]:
    pad_h = (((int(height) // 64) + 1) * 64 - int(height)) % 64
    pad_w = (((int(width) // 64) + 1) * 64 - int(width)) % 64
    return int(height) + pad_h, int(width) + pad_w


def window_input_names() -> List[str]:
    return ["query", "window", "flows8", "visconfs8"]


def window_output_names() -> List[str]:
    return ["flows", "visconfs", "flows8_out", "visconfs8_out"]


def dummy_window_inputs(
    device: torch.device,
    height: int,
    width: int,
    window_len: int = 16,
    dim: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hp, wp = padded_hw(height, width)
    h8, w8 = hp // 8, wp // 8
    query = torch.rand(1, 3, height, width, device=device) * 255.0
    window = torch.rand(1, window_len, 3, height, width, device=device) * 255.0
    zeros = torch.randn(1, window_len, 2, h8, w8, device=device) * 0.01
    del dim
    return query, window, zeros, zeros.clone()


def window_notes(window_len: int, height: int, width: int) -> List[str]:
    stride = window_len // 2
    return [
        f"AllTracker sliding window. Static S={window_len}, stride={stride}, {height}x{width}.",
        "Host matches Net.forward_sliding: fmap_anchor=frame0, last-frame pad, carry flows8/visconfs8.",
        "T==2 uses the pairwise *_pair.onnx graph (S=1 branch). T>=3 uses this window graph.",
        "Sample traj_maps = flows + pixel_grid at the original query points (last real frame of the clip).",
    ]


class AllTrackerWindowOnnx(nn.Module):
    """One ``forward_window`` with encoder, ImageNet norm, and InputPadder baked in."""

    def __init__(self, model: nn.Module, iters: int = 4, window_len: int = 16):
        super().__init__()
        self.model = model
        self.iters = int(iters)
        self.window_len = int(window_len)

    def _encode(self, images_btchw: torch.Tensor) -> torch.Tensor:
        batch = as_int(images_btchw.shape[0])
        time = as_int(images_btchw.shape[1])
        height = as_int(images_btchw.shape[3])
        width = as_int(images_btchw.shape[4])
        flat = images_btchw.reshape(batch * time, 3, height, width)
        if getattr(self.model, "use_basicencoder", False):
            if getattr(self.model, "full_split", False):
                feats = torch.cat([self.model.fnet(flat), self.model.cnet(flat)], dim=1)
            else:
                feats = self.model.fnet(flat)
        else:
            feats = self.model.dot_conv(self.model.cnn(flat))
        channels = as_int(feats.shape[1])
        h8 = as_int(feats.shape[2])
        w8 = as_int(feats.shape[3])
        return feats.reshape(batch, time, channels, h8, w8)

    def _norm(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        return (images / 255.0 - mean) / std

    def forward(
        self,
        query: torch.Tensor,
        window: torch.Tensor,
        flows8: torch.Tensor,
        visconfs8: torch.Tensor,
    ):
        from nets.blocks import InputPadder

        height = as_int(query.shape[2])
        width = as_int(query.shape[3])
        batch = as_int(window.shape[0])
        seq = as_int(window.shape[1])
        query_n = self._norm(query)
        win_n = self._norm(window.reshape(batch * seq, 3, height, width))
        padder = InputPadder(query_n.shape)
        query_p = padder.pad(query_n)[0]
        win_p = padder.pad(win_n)[0]
        hp, wp = as_int(win_p.shape[2]), as_int(win_p.shape[3])
        win_p = win_p.reshape(batch, seq, 3, hp, wp)
        fmap_anchor = self._encode(query_p[:, None])[:, 0]
        fmaps2 = self._encode(win_p)
        h8 = as_int(flows8.shape[3])
        w8 = as_int(flows8.shape[4])
        flows8_in = flows8.reshape(batch * seq, 2, h8, w8)
        vis8_in = visconfs8.reshape(batch * seq, 2, h8, w8)
        flow_preds, vis_preds, flows8_out, vis8_out, _feat = self.model.forward_window(
            fmap_anchor,
            fmaps2,
            vis8_in,
            iters=self.iters,
            flowfeat=None,
            flows8=flows8_in,
            is_training=False,
        )
        flow_up = padder.unpad(flow_preds[-1]).reshape(batch, seq, 2, height, width)
        vis_up = padder.unpad(torch.sigmoid(vis_preds[-1])).reshape(batch, seq, 2, height, width)
        flows8_out = flows8_out.reshape(batch, seq, 2, h8, w8)
        vis8_out = vis8_out.reshape(batch, seq, 2, h8, w8)
        return flow_up, vis_up, flows8_out, vis8_out


def sliding_starts(n_frames: int, window_len: int, stride: Optional[int] = None) -> Tuple[List[int], int]:
    """Return window starts and tail pad so the last window is length ``window_len``."""
    seq = int(window_len)
    step = seq // 2 if stride is None else int(stride)
    indices: List[int] = []
    start = 0
    while start + seq < n_frames:
        indices.append(start)
        start += step
    indices.append(start)
    pad = indices[-1] + seq - n_frames
    return indices, int(pad)


def sample_traj_xy(flow_2hw: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Sample ``flow + pixel_grid`` at query pixels (AllTracker demo / AllTrackerSTIR)."""
    if flow_2hw.ndim != 3 or flow_2hw.shape[0] != 2:
        raise ValueError("expected flow [2,H,W], got %s" % (flow_2hw.shape,))
    fx, fy = flow_2hw[0], flow_2hw[1]
    height, width = fx.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    traj_x = fx + xx
    traj_y = fy + yy
    x = np.clip(np.asarray(pts_xy[:, 0], dtype=np.float32), 0.0, float(width - 1))
    y = np.clip(np.asarray(pts_xy[:, 1], dtype=np.float32), 0.0, float(height - 1))
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0

    def samp(img: np.ndarray) -> np.ndarray:
        ia = img[y0, x0]
        ib = img[y0, x1]
        ic = img[y1, x0]
        id_ = img[y1, x1]
        return (ia * (1.0 - wx) + ib * wx) * (1.0 - wy) + (ic * (1.0 - wx) + id_ * wx) * wy

    return np.stack([samp(traj_x), samp(traj_y)], axis=-1).astype(np.float32)


class OnnxAllTrackerTracker:
    """Growing-clip host around the S-window ONNX graph (and optional T=2 pair graph)."""

    def __init__(
        self,
        onnx_path: Path,
        window_len: int = 16,
        extra: Optional[dict] = None,
        pair_onnx: Optional[Path] = None,
    ):
        from onnx_hosts.run_all_onnx_stir import _ort_session

        extra = extra or {}
        self.window_len = int(extra.get("window_len", extra.get("S", window_len)))
        self.stride = int(extra.get("stride", self.window_len // 2))
        self.height = int(extra.get("height", 0))
        self.width = int(extra.get("width", 0))
        if extra.get("h8") and extra.get("w8"):
            self.h8, self.w8 = int(extra["h8"]), int(extra["w8"])
        else:
            hp, wp = padded_hw(self.height or 8, self.width or 8)
            self.h8, self.w8 = hp // 8, wp // 8
        self.sess = _ort_session(onnx_path)
        pair_path = pair_onnx
        if pair_path is None and extra.get("pair_onnx"):
            pair_path = Path(extra["pair_onnx"])
        if pair_path is None:
            cand = Path(onnx_path).with_name(Path(onnx_path).stem + "_pair.onnx")
            if cand.is_file():
                pair_path = cand
        self.pair_sess = _ort_session(pair_path) if pair_path and Path(pair_path).is_file() else None
        self.latencies_ms: List[float] = []

    def reset(self) -> None:
        self.latencies_ms = []

    def _run_window(
        self,
        query: np.ndarray,
        window: np.ndarray,
        flows8: np.ndarray,
        visconfs8: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        feeds = {
            "query": np.ascontiguousarray(query),
            "window": np.ascontiguousarray(window),
            "flows8": np.ascontiguousarray(flows8),
            "visconfs8": np.ascontiguousarray(visconfs8),
        }
        t0 = time.perf_counter()
        outs = self.sess.run(window_output_names(), feeds)
        dt = (time.perf_counter() - t0) * 1000.0
        return outs[0], outs[1], outs[2], outs[3], dt

    def _run_pair(self, frame0: np.ndarray, frame1: np.ndarray) -> Tuple[np.ndarray, float]:
        images = np.stack([frame0, frame1], axis=0)[None].astype(np.float32)
        t0 = time.perf_counter()
        flow = self.pair_sess.run(None, {"images": images})[0]
        dt = (time.perf_counter() - t0) * 1000.0
        while flow.ndim > 3:
            if flow.shape[0] == 1:
                flow = flow[0]
            else:
                flow = flow[-1]
        return np.asarray(flow, dtype=np.float32), dt

    def forward_sliding(self, frames_nchw: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
        frames = [np.asarray(f, dtype=np.float32) for f in frames_nchw]
        if frames[0].ndim == 4:
            frames = [f[0] if f.shape[0] == 1 else f for f in frames]
        if self.height <= 0 or self.width <= 0:
            self.height = int(frames[0].shape[-2])
            self.width = int(frames[0].shape[-1])
            hp, wp = padded_hw(self.height, self.width)
            self.h8, self.w8 = hp // 8, wp // 8
        n_real = len(frames)
        if n_real < 2:
            raise RuntimeError("AllTracker sliding needs at least 2 frames")
        if n_real == 2 and self.pair_sess is not None:
            flow, dt = self._run_pair(frames[0], frames[1])
            return flow[None], dt

        seq = self.window_len
        indices, pad = sliding_starts(n_real, seq, self.stride)
        padded = list(frames)
        if pad > 0:
            padded.extend([frames[-1]] * pad)
        t_pad = len(padded)
        query = frames[0][None]
        height, width = int(query.shape[2]), int(query.shape[3])
        full = np.zeros((t_pad, 2, height, width), dtype=np.float32)
        visited = np.zeros((t_pad,), dtype=bool)
        flows8 = np.zeros((1, seq, 2, self.h8, self.w8), dtype=np.float32)
        vis8 = np.zeros((1, seq, 2, self.h8, self.w8), dtype=np.float32)
        total_ms = 0.0
        half = seq // 2
        stride = self.stride
        for ii, ind in enumerate(indices):
            if ii > 0:
                keep = flows8[:, stride : stride + half]
                last = flows8[:, stride + half - 1 : stride + half]
                flows8 = np.concatenate([keep, np.repeat(last, half, axis=1)], axis=1)
                keep_v = vis8[:, stride : stride + half]
                last_v = vis8[:, stride + half - 1 : stride + half]
                vis8 = np.concatenate([keep_v, np.repeat(last_v, half, axis=1)], axis=1)
            window = np.stack(padded[ind : ind + seq], axis=0)[None].astype(np.float32)
            flow_w, _vis_w, flows8, vis8, dt = self._run_window(query, window, flows8, vis8)
            total_ms += dt
            visiting = np.zeros((t_pad,), dtype=bool)
            visiting[ind : ind + seq] = True
            to_fill = visiting & (~visited)
            n_fill = int(to_fill.sum())
            if n_fill:
                full[to_fill] = flow_w[0, -n_fill:]
            visited |= visiting
        return full[:n_real], total_ms

    def track_sequence(
        self,
        frames_nchw: Sequence[np.ndarray],
        queries_xy: np.ndarray,
        per_frame: bool = True,
    ) -> Tuple[np.ndarray, List[float]]:
        """Return ``(N,T,2)`` in model pixels.

        ``per_frame=True`` re-runs sliding on the growing clip every step, matching
        AllTrackerSTIR latency. ``per_frame=False`` runs once on the full clip.
        """
        n_pts = int(queries_xy.shape[0])
        t_all = len(frames_nchw)
        pred = np.zeros((n_pts, t_all, 2), dtype=np.float32)
        pred[:, 0] = queries_xy
        self.latencies_ms = []
        if t_all < 2:
            return pred, self.latencies_ms
        if not per_frame:
            t0 = time.perf_counter()
            flows, _dt = self.forward_sliding(frames_nchw)
            pred[:, -1] = sample_traj_xy(flows[-1], queries_xy)
            for t in range(1, t_all - 1):
                pred[:, t] = sample_traj_xy(flows[t], queries_xy)
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            return pred, self.latencies_ms
        for t in range(1, t_all):
            t0 = time.perf_counter()
            flows, _dt = self.forward_sliding(frames_nchw[: t + 1])
            pred[:, t] = sample_traj_xy(flows[-1], queries_xy)
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return pred, self.latencies_ms
