"""Official PIPs++ long-video host loop (PointOdyssey ``test_on_fullseq``).

Windows of ``S_max`` (default 128), overlap 1, ``feat_init`` chaining, and
zero-velocity fill of the future. The last window snaps back so its length is
still ``S_max`` when the clip is longer than ``S_max``.

``run_window`` is supplied by PyTorch or ONNX; this module only owns indexing.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

S_MAX_DEFAULT = 128

WindowFn = Callable[
    [int, int, np.ndarray, Any],
    Tuple[np.ndarray, Any, float],
]


def official_windows(n_frames: int, s_max: int = S_MAX_DEFAULT) -> List[Tuple[int, int]]:
    """Return ``(cur_frame, end_frame)`` pairs matching ``pips2/test_on_pod.py``."""
    if n_frames < 1:
        return []
    s_max = int(s_max)
    cur_frame = 0
    out: List[Tuple[int, int]] = []
    while True:
        end_frame = cur_frame + s_max
        if end_frame > n_frames:
            diff = end_frame - n_frames
            end_frame = end_frame - diff
            cur_frame = max(cur_frame - diff, 0)
        out.append((int(cur_frame), int(end_frame)))
        if end_frame >= n_frames:
            return out
        cur_frame = cur_frame + s_max - 1


def run_official_online(
    queries_xy: np.ndarray,
    n_frames: int,
    run_window: WindowFn,
    s_max: int = S_MAX_DEFAULT,
    cache: Optional[Dict[Tuple[int, int], Tuple[np.ndarray, Any]]] = None,
) -> Tuple[np.ndarray, List[float], Dict[Tuple[int, int], Tuple[np.ndarray, Any]]]:
    """Track queries over a clip of ``n_frames`` with the official window schedule.

    Args:
        queries_xy: ``(N, 2)`` in the same coordinate system as ``run_window``.
        n_frames: clip length ``T``.
        run_window: ``(cur, end, trajs_e0[S,N,2], feat_init) ->
            (trajs[S,N,2], feat_init, latency_ms)``.
        s_max: PIPs++ window length (PointOdyssey default 128).
        cache: optional map of completed **interior** windows to
            ``(trajs, feat_init)`` so a growing prefix is not O(T^2).

    Returns:
        ``trajs_e`` of shape ``(T, N, 2)``, per-window latencies, updated cache.
    """
    queries_xy = np.asarray(queries_xy, dtype=np.float32)
    if queries_xy.ndim != 2 or queries_xy.shape[1] != 2:
        raise ValueError("queries_xy must be (N, 2), got %s" % (queries_xy.shape,))
    n_frames = int(n_frames)
    n_pts = int(queries_xy.shape[0])
    if n_frames < 1:
        return np.zeros((0, n_pts, 2), dtype=np.float32), [], cache if cache is not None else {}

    trajs_e = np.repeat(queries_xy[None, :, :], n_frames, axis=0)
    windows = official_windows(n_frames, s_max)
    feat_init: Any = None
    latencies: List[float] = []
    if cache is None:
        cache = {}

    for i, (cur, end) in enumerate(windows):
        is_last = i == len(windows) - 1
        key = (cur, end)
        if (not is_last) and key in cache:
            cached_trajs, feat_init = cache[key]
            trajs_e[cur:end] = cached_trajs
            if end < n_frames:
                trajs_e[end:] = trajs_e[end - 1 : end]
            continue
        traj_seq = trajs_e[cur:end].copy()
        preds, feat_init, dt = run_window(cur, end, traj_seq, feat_init)
        preds = np.asarray(preds, dtype=np.float32)
        if preds.shape != (end - cur, n_pts, 2):
            raise ValueError(
                "run_window returned %s, expected (%d, %d, 2)" % (preds.shape, end - cur, n_pts)
            )
        trajs_e[cur:end] = preds
        if end < n_frames:
            trajs_e[end:] = trajs_e[end - 1 : end]
        latencies.append(float(dt))
        if not is_last:
            cache[key] = (preds.copy(), feat_init)
    return trajs_e, latencies, cache
