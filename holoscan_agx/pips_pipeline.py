"""Official PIPs++ online host around a static S-window ONNX graph.

Host loop matches ``pips2/test_on_pod.py`` (S_max=128, overlap 1, feat_init,
zero-vel). Short clips (T < S) are padded to the frozen graph length and
sliced back; that is the only ONNX approximation of variable S_local.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from onnx_hosts.pips_official_online import S_MAX_DEFAULT, run_official_online
from onnx_hosts.run_all_onnx_stir import PAD_N, _ort_run_dict, _ort_session, _pad_xy


def _pad_time(arr: np.ndarray, s_max: int, axis: int = 1) -> np.ndarray:
    """Repeat the last slice along ``axis`` so length is ``s_max``."""
    cur = int(arr.shape[axis])
    if cur >= s_max:
        return arr
    pad = s_max - cur
    last = np.take(arr, [-1], axis=axis)
    reps = [1] * arr.ndim
    reps[axis] = pad
    return np.concatenate([arr, np.tile(last, reps)], axis=axis)


class OnnxPipsTracker:
    def __init__(self, onnx_path: Path, window: int = S_MAX_DEFAULT, extra: Optional[dict] = None):
        extra = extra or {}
        self.sess = _ort_session(onnx_path)
        self.window = int(extra.get("S", window))
        self.feat_dim = int(extra.get("feat_dim", 128))
        self.pad_n = int(extra.get("N", PAD_N))
        self.input_names = {inp.name for inp in self.sess.get_inputs()}
        self.latencies_ms: List[float] = []

    def _run_window(
        self,
        n_real: int,
        frames_nchw: Sequence[np.ndarray],
        cur: int,
        end: int,
        trajs_e0: np.ndarray,
        feat_init: Any,
    ) -> Tuple[np.ndarray, Any, float]:
        import time

        t0 = time.perf_counter()
        s_local = end - cur
        sl = list(frames_nchw[cur:end])
        if s_local < self.window:
            sl = sl + [sl[-1]] * (self.window - s_local)
        rgbs = np.stack(sl, axis=0)[None].astype(np.float32)

        trajs_pad = np.zeros((1, self.window, self.pad_n, 2), dtype=np.float32)
        padded_xy_seq = np.zeros((self.window, self.pad_n, 2), dtype=np.float32)
        for t in range(s_local):
            padded_xy_seq[t] = _pad_xy(trajs_e0[t], n=self.pad_n)
        if s_local < self.window:
            padded_xy_seq[s_local:] = padded_xy_seq[s_local - 1]
        trajs_pad[0] = padded_xy_seq

        z = np.zeros((1, self.window, self.pad_n, self.feat_dim), dtype=np.float32)
        if feat_init is None:
            f1, f2, f4 = z.copy(), z.copy(), z.copy()
        else:
            f1, f2, f4 = feat_init
            f1 = _pad_time(np.asarray(f1, dtype=np.float32), self.window, axis=1)
            f2 = _pad_time(np.asarray(f2, dtype=np.float32), self.window, axis=1)
            f4 = _pad_time(np.asarray(f4, dtype=np.float32), self.window, axis=1)

        rgb_name = "window" if "window" in self.input_names else "rgbs"
        feeds: Dict[str, np.ndarray] = {rgb_name: rgbs, "trajs_e0": trajs_pad}
        if "previous_feats1" in self.input_names:
            feeds["previous_feats1"] = np.ascontiguousarray(f1)
            feeds["previous_feats2"] = np.ascontiguousarray(f2)
            feeds["previous_feats4"] = np.ascontiguousarray(f4)
        elif "feats1" in self.input_names:
            feeds["feats1"] = np.ascontiguousarray(f1)
            feeds["feats2"] = np.ascontiguousarray(f2)
            feeds["feats4"] = np.ascontiguousarray(f4)

        outd = _ort_run_dict(self.sess, feeds)

        trajs = np.asarray(outd["trajs"], dtype=np.float32)[0, :s_local, :n_real]
        new_init = None
        if "new_feats1" in outd:
            new_init = (
                np.ascontiguousarray(outd["new_feats1"][:, :s_local]),
                np.ascontiguousarray(outd["new_feats2"][:, :s_local]),
                np.ascontiguousarray(outd["new_feats4"][:, :s_local]),
            )
        elif "feats1_out" in outd:
            new_init = (
                np.ascontiguousarray(outd["feats1_out"][:, :s_local]),
                np.ascontiguousarray(outd["feats2_out"][:, :s_local]),
                np.ascontiguousarray(outd["feats4_out"][:, :s_local]),
            )
        dt = (time.perf_counter() - t0) * 1000.0
        return trajs, new_init, dt

    def track_sequence(
        self,
        frames_nchw: Sequence[np.ndarray],
        queries_xy: np.ndarray,
    ) -> Tuple[np.ndarray, List[float]]:
        n_real = int(queries_xy.shape[0])
        t_all = len(frames_nchw)

        def run_window(cur, end, trajs_e0, feat_init):
            return self._run_window(n_real, frames_nchw, cur, end, trajs_e0, feat_init)

        trajs_tn, latencies, _ = run_official_online(
            queries_xy.astype(np.float32),
            t_all,
            run_window,
            s_max=self.window,
        )
        # STIR host expects (N, T, 2)
        pred = np.transpose(trajs_tn, (1, 0, 2))
        self.latencies_ms = latencies
        return pred, latencies
