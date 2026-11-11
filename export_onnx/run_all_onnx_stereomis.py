#!/usr/bin/env python3
"""Evaluate exported StereoMIS ONNX models: TAP metrics + per-call latency.

Writes the same JSON family as flow2d_stereomis.py under --outdir.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_all_onnx_stir as ox  # noqa: E402
from datasets.metrics import delta_2d, mte, surv_2d  # noqa: E402
from datasets.stereomis import DEFAULT_DATADIR, load_sequences  # noqa: E402

METHODS = ox.METHODS
FULLRES_METHODS = ox.FULLRES_METHODS
PAD_N = ox.PAD_N
ONNX_ROOT = ox.ONNX_ROOT
STEREOMIS_NATIVE_HW = (512, 640)


def resolve_artifact(method: str, scale: int, fp16: bool = False):
    family = ONNX_ROOT / ("COTRACKER" if method.startswith("COTRACKER") else method)
    if method in ("COTRACKER", "COTRACKER_ONLINE"):
        stem = "cotracker_online_step_fp16" if fp16 else "cotracker_online_step"
        step_json = family / ("%s.json" % stem)
        step_onnx = family / ("%s.onnx" % stem)
        if step_json.is_file() and step_onnx.is_file():
            return ox.load_meta(step_json), step_onnx
        return None, None
    if method == "TAPNEXTPP":
        catalog = family / "tapnextpp_online_step.json"
        if catalog.is_file() and (family / "tapnextpp_online_step_n1.onnx").is_file():
            meta = ox.load_meta(catalog)
            meta = dict(meta)
            meta["height"] = STEREOMIS_NATIVE_HW[0]
            meta["width"] = STEREOMIS_NATIVE_HW[1]
            return meta, catalog
        return None, None
    if method == "LITETRACKER":
        fnet = family / "fnet.onnx"
        if fnet.is_file() and (family / "corr_mlp.onnx").is_file() and (family / "updateformer.onnx").is_file():
            meta = {
                "method": "LITETRACKER",
                "scale": 1,
                "height": STEREOMIS_NATIVE_HW[0],
                "width": STEREOMIS_NATIVE_HW[1],
                "extra": {"onnx_dir": str(family)},
            }
            return meta, fnet
        return None, None
    stem = "%s_stereomis_scale%d" % (method.lower(), scale)
    meta_path = family / ("%s.json" % stem)
    onnx_path = family / ("%s.onnx" % stem)
    if meta_path.is_file() and onnx_path.is_file():
        return ox.load_meta(meta_path), onnx_path
    return None, None


def load_frames_rgb(paths, width: int, height: int):
    """Native RGB uint8. Resize only when the decoded frame is not already that size."""
    frames = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cannot read %s" % path)
        if bgr.shape[0] != height or bgr.shape[1] != width:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def load_frames_nchw(paths, width: int, height: int):
    frames = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cannot read %s" % path)
        frames.append(ox.preprocess_bgr(bgr, width, height)[0])
    return frames


def evaluate_prediction(pred: np.ndarray, sequence) -> dict:
    gt = sequence.gt_2d
    valid = sequence.valid
    height, width = sequence.height, sequence.width
    last_errors = []
    for point_index in range(sequence.num_points):
        valid_frames = np.flatnonzero(valid[point_index])
        if valid_frames.size == 0:
            continue
        last_t = int(valid_frames[-1])
        last_errors.append(np.linalg.norm(pred[point_index, last_t] - gt[point_index, last_t]))
    endpoint = float(np.mean(last_errors)) if last_errors else float("nan")
    return {
        "MTE_2D": float(mte(pred, gt, valid)),
        "delta_2D": float(delta_2d(pred, gt, valid, height, width)),
        "survival_2D": float(surv_2d(pred, gt, valid, height, width)),
        "endpointerror": endpoint,
        "num_points": int(sequence.num_points),
        "num_frames": int(sequence.num_frames),
        "height": int(height),
        "width": int(width),
    }


def track_dense_traj(sess, frames, pts_full, model_hw, native_hw):
    n_real = int(pts_full.shape[0])
    t_all = len(frames)
    pred = np.zeros((n_real, t_all, 2), dtype=np.float32)
    pred[:, 0] = pts_full
    pts_work = ox._map_xy(pts_full, native_hw, model_hw)
    latencies = []
    prev = frames[0][None]
    for t, frame in enumerate(frames[1:], start=1):
        cur = frame[None]
        t0 = time.perf_counter()
        flow = sess.run(None, {"image1": prev, "image2": cur})[0]
        flow = ox._squeeze_flow(flow)
        pts_work = pts_work + ox.bilinear_sample_flow(flow, pts_work)
        pred[:, t] = ox._map_xy(pts_work, model_hw, native_hw)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        prev = cur
    return pred, latencies


def track_alltracker_traj(sess, frames, pts_full, model_hw, native_hw, host=None):
    if host is None:
        raise RuntimeError("AllTracker ONNX requires OnnxAllTrackerTracker")
    pts_work = ox._map_xy(pts_full, native_hw, model_hw)
    pred_nt, lat_ms = host.track_sequence(frames, pts_work, per_frame=True)
    n, t, _ = pred_nt.shape
    pred = ox._map_xy(pred_nt.reshape(n * t, 2), model_hw, native_hw).reshape(n, t, 2)
    return pred, lat_ms


def track_pips_traj(sess, frames, pts_full, model_hw, native_hw, window=8, extra=None):
    n_real = int(pts_full.shape[0])
    t_all = len(frames)
    pred = np.zeros((n_real, t_all, 2), dtype=np.float32)
    pred[:, 0] = pts_full
    extra = extra or {}
    dim = int(extra.get("feat_dim", 128))
    pts_work = ox._map_xy(pts_full, native_hw, model_hw)
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
        trajs_e0[:, :, :, :] = ox._pad_xy(pts_work)
        if "window" in input_names:
            feeds = {"window": rgbs, "trajs_e0": trajs_e0}
        else:
            feeds = {"rgbs": rgbs, "trajs_e0": trajs_e0}
        if feats is not None:
            feeds.update(feats)
        outd = ox._ort_run_dict(sess, feeds)
        trajs = outd["trajs"]
        for local in range(actual):
            xy = ox._map_xy(trajs[0, local, :n_real].astype(np.float32), model_hw, native_hw)
            pred[:, i + local] = xy
        pts_work = trajs[0, actual - 1, :n_real].astype(np.float32)
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
    return pred, latencies


def track_locotrack_traj(sess, frames, pts_full, model_hw, native_hw, clip_t=None):
    if clip_t is None:
        clip_t = ox.LOCOTRACK_MAX_FRAMES
    n_real = int(pts_full.shape[0])
    t_all = len(frames)
    pred = np.zeros((n_real, t_all, 2), dtype=np.float32)
    pred[:, 0] = pts_full
    query_xy = ox._map_xy(pts_full, native_hw, model_hw)
    latencies = []
    q = np.zeros((1, PAD_N, 3), dtype=np.float32)
    padded = ox._pad_xy(query_xy)
    input_names = [i.name for i in sess.get_inputs()]
    use_window = "window" in input_names
    q[0, :, 0] = 0.0
    if use_window:
        q[0, :, 1] = padded[:, 0]
        q[0, :, 2] = padded[:, 1]
    else:
        q[0, :, 1] = padded[:, 1]
        q[0, :, 2] = padded[:, 0]
    static_t = ox._onnx_static_time(sess)
    max_frames = int(clip_t)
    if static_t is not None:
        max_frames = min(max_frames, static_t)
    for t in range(1, len(frames)):
        t0 = time.perf_counter()
        sl = ox.keep_first_growing_clip(frames, t, max_frames)
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
        pred[:, t] = ox._map_xy(tracks[0, :n_real, -1].astype(np.float32), model_hw, native_hw)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return pred, latencies


def track_cotracker_traj(step_sess, frames, pts_full, model_hw, native_hw, window=16, step=8, extra=None):
    n_real = int(pts_full.shape[0])
    t_all = len(frames)
    pred = np.zeros((n_real, t_all, 2), dtype=np.float32)
    pred[:, 0] = pts_full
    pts_work = ox._map_xy(pts_full, native_hw, model_hw)
    extra = extra or {}
    n = len(frames)
    latencies = []
    states = None
    win_start = 0
    seen = set()
    while win_start < n:
        if win_start in seen:
            raise RuntimeError("CoTracker window start did not advance (stuck at %d)" % win_start)
        seen.add(win_start)
        t0 = time.perf_counter()
        video, actual = ox._window_video(frames, win_start, window)
        tracks, states = ox._cotracker_forward_all(step_sess, video, pts_work, states, window, extra)
        for local in range(actual):
            idx = win_start + local
            if idx >= t_all:
                break
            pred[:, idx] = ox._map_xy(tracks[0, local, :n_real].astype(np.float32), model_hw, native_hw)
        if win_start + window >= n:
            latencies.append((time.perf_counter() - t0) * 1000.0)
            break
        latencies.append((time.perf_counter() - t0) * 1000.0)
        win_start += step
    return pred, latencies


def track_litetracker_traj(tracker, frames, pts_full):
    import torch

    n_real = int(pts_full.shape[0])
    t_all = len(frames)
    pred = np.zeros((n_real, t_all, 2), dtype=np.float32)
    q = torch.zeros((1, n_real, 3), dtype=torch.float32)
    q[0, :, 1] = torch.from_numpy(pts_full[:, 0])
    q[0, :, 2] = torch.from_numpy(pts_full[:, 1])
    latencies = []
    tracker.reset()
    for t, frame in enumerate(frames):
        t0 = time.perf_counter()
        tens = torch.from_numpy(np.ascontiguousarray(frame[None]))
        coords, _vis, _conf = tracker(tens, q)
        pred[:, t] = coords[0, 0, :n_real].detach().cpu().numpy().astype(np.float32)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return pred, latencies


def _track_stereomis(method, sess, tracker, mft_host, pips_host, at_host, frames, start, model_hw, native_hw, extra):
    if method in ("RAFT", "WAFT"):
        return track_dense_traj(sess, frames, start, model_hw, native_hw)
    if method == "MFT":
        pts_work = ox._map_xy(start, native_hw, model_hw)
        pred_nt, lat_ms = mft_host.track_sequence(frames, pts_work)
        n, t, _ = pred_nt.shape
        pred = ox._map_xy(pred_nt.reshape(n * t, 2), model_hw, native_hw).reshape(n, t, 2)
        return pred, lat_ms
    if method == "ALLTRACKER":
        return track_alltracker_traj(sess, frames, start, model_hw, native_hw, host=at_host)
    if method == "PIPS":
        pts_work = ox._map_xy(start, native_hw, model_hw)
        pred_nt, lat_ms = pips_host.track_sequence(frames, pts_work)
        n, t, _ = pred_nt.shape
        pred = ox._map_xy(pred_nt.reshape(n * t, 2), model_hw, native_hw).reshape(n, t, 2)
        return pred, lat_ms
    if method == "LOCOTRACK":
        return track_locotrack_traj(
            sess, frames, start, model_hw, native_hw,
            clip_t=int(extra.get("max_frames", extra.get("T", ox.LOCOTRACK_MAX_FRAMES))),
        )
    if method in ("COTRACKER", "COTRACKER_ONLINE"):
        return track_cotracker_traj(
            sess,
            frames,
            start,
            model_hw,
            native_hw,
            window=int(extra.get("window_len", 16)),
            step=int(extra.get("step", 8)),
            extra=extra,
        )
    if method == "LITETRACKER":
        return track_litetracker_traj(tracker, frames, start)
    if method == "TAPNEXTPP":
        traj, _vis, lat_ms = tracker.track_sequence(frames, start)
        return traj, lat_ms
    raise RuntimeError(method)


def eval_one(method: str, scale: int, sequences, args):
    meta, onnx_path = resolve_artifact(method, scale, fp16=bool(getattr(args, "fp16", False)))
    if meta is None:
        print("%s scale=%s  SKIP (no StereoMIS ONNX)" % (method, scale), flush=True)
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
        from export_onnx.litetracker_ort import OnnxLiteTracker
        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        # Host defaults to CPU: the CUDA correlation rounding drops StereoMIS P2_1
        # point 2. LITETRACKER_DEVICE=cuda with LITETRACKER_MAX_STEP (feature pixels)
        # keeps that step on GPU and inside the pull-back range.
        host_device = os.environ.get("LITETRACKER_DEVICE", "cpu")
        tracker = OnnxLiteTracker(
            str(ONNX_ROOT / "LITETRACKER"),
            providers=providers or ["CPUExecutionProvider"],
            device=host_device,
        )
        backend = "onnx:" + ",".join(providers or ["CPUExecutionProvider"]) + "+host:" + host_device
        max_step = os.environ.get("LITETRACKER_MAX_STEP")
        if max_step:
            backend += "+max_step:" + max_step
    elif method == "TAPNEXTPP":
        from export_onnx.tapnextpp_ort import OnnxTapNextPP

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
        from export_onnx.alltracker_window import OnnxAllTrackerTracker

        extra = dict(extra)
        extra.setdefault("height", height)
        extra.setdefault("width", width)
        at_host = OnnxAllTrackerTracker(onnx_path, window_len=int(extra.get("window_len", 16)), extra=extra)
        backend = "holoscan-host:ALLTRACKER+onnx"
    elif reset_each:
        from export_onnx.cotracker_ort import CoTrackerStepRunner, lock_jetson_clocks

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
        sess = ox._open_locotrack(onnx_path, extra)
        backend = "onnx:" + ",".join(sess.get_providers())
        if sess.feats is not None:
            print("LocoTrack feature chunk=%d" % sess.chunk, flush=True)
    else:
        sess = ox._ort_session(onnx_path)
        backend = "onnx:" + ",".join(sess.get_providers())
    print("backend %s" % backend, flush=True)
    use_agx = getattr(args, "wall_scope", "track") == "agx"
    if use_agx:
        print("wall clock: %s" % ox.AGX_PIPELINE_SCOPE, flush=True)

    metrics_by_seq = {}
    positionlists = {}
    latencies_by_seq = {}
    pipeline_by_seq = {}
    metric_sums = defaultdict(float)
    n_ok = 0

    for sequence in sequences:
        native_hw = (int(sequence.height), int(sequence.width))
        start = sequence.query_points()
        if method == "TAPNEXTPP":
            tracker.ensure(int(start.shape[0]))
        frames = None
        n_frames = 0
        pred = None
        lat_ms = None
        wall_s = None
        metrics = None
        try:
            if reset_each:
                sess = ox._release_ort_session(sess)
                from export_onnx.cotracker_ort import CoTrackerStepRunner

                sess = CoTrackerStepRunner(onnx_path)

            def _go():
                return _track_stereomis(
                    method, sess, tracker, mft_host, pips_host, at_host, frames, start, model_hw, native_hw, extra
                )

            try:
                if use_agx:
                    ox.sync_timing()
                    t_agx = time.perf_counter()
                    frames = load_frames_rgb(sequence.frame_paths, width, height) if method == "TAPNEXTPP" else load_frames_nchw(sequence.frame_paths, width, height)
                    n_frames = len(frames)
                    pred, lat_ms = _go()
                    metrics = evaluate_prediction(pred, sequence)
                    ox.sync_timing()
                    wall_s = time.perf_counter() - t_agx
                else:
                    frames = load_frames_rgb(sequence.frame_paths, width, height) if method == "TAPNEXTPP" else load_frames_nchw(sequence.frame_paths, width, height)
                    n_frames = len(frames)
                    (pred, lat_ms), wall_s = ox.time_pipeline(_go)
            except Exception as exc:
                if reset_each and ox._is_ort_oom(exc):
                    print("%s  OOM, new ORT session and retry once" % sequence.name, flush=True)
                    sess = ox._release_ort_session(sess)
                    from export_onnx.cotracker_ort import CoTrackerStepRunner

                    sess = CoTrackerStepRunner(onnx_path)
                    if use_agx:
                        ox.sync_timing()
                        t_agx = time.perf_counter()
                        if frames is None:
                            frames = load_frames_rgb(sequence.frame_paths, width, height) if method == "TAPNEXTPP" else load_frames_nchw(sequence.frame_paths, width, height)
                            n_frames = len(frames)
                        pred, lat_ms = _go()
                        metrics = evaluate_prediction(pred, sequence)
                        ox.sync_timing()
                        wall_s = time.perf_counter() - t_agx
                    else:
                        (pred, lat_ms), wall_s = ox.time_pipeline(_go)
                else:
                    raise
        except Exception as exc:
            print("%s  ERROR %s: %s" % (sequence.name, type(exc).__name__, exc), flush=True)
            if method == "LOCOTRACK" and ox._is_ort_oom(exc):
                print("%s  releasing ORT arena after OOM" % sequence.name, flush=True)
                sess = ox._release_ort_session(sess)
                sess = ox._open_locotrack(onnx_path, extra)
            if args.stop_on_error:
                raise
            continue
        finally:
            frames = None
            if reset_each:
                sess = ox._release_ort_session(sess)
            gc.collect()
        if pred is None:
            continue
        if metrics is None:
            metrics = evaluate_prediction(pred, sequence)
        metrics_by_seq[sequence.name] = metrics
        positionlists[sequence.name] = pred[:, -1].astype(float).tolist()
        latencies_by_seq[sequence.name] = [float(x) for x in lat_ms]
        pipe = ox.pipeline_record(
            n_frames,
            wall_s if wall_s is not None else 0.0,
            ox.STEREOMIS_CAMERA_FPS,
            scope=ox.AGX_PIPELINE_SCOPE if use_agx else None,
        )
        pipeline_by_seq[sequence.name] = pipe
        for key, value in metrics.items():
            if key in {"num_points", "num_frames", "height", "width"}:
                continue
            if np.isfinite(value):
                metric_sums[key] += float(value)
        n_ok += 1
        print(
            "%s  MTE=%.3f  delta=%.3f  surv=%.3f  EPE=%.2f  frames=%d  pipeline_fps=%.2f  ms/pair=%.1f  realtime=%s"
            % (
                sequence.name,
                metrics["MTE_2D"],
                metrics["delta_2D"],
                metrics["survival_2D"],
                metrics["endpointerror"],
                n_frames,
                pipe["pipeline_fps"] if pipe["pipeline_fps"] is not None else float("nan"),
                pipe["pipeline_ms_per_pair"] if pipe["pipeline_ms_per_pair"] is not None else float("nan"),
                "yes" if pipe["meets_realtime"] else "no",
            ),
            flush=True,
        )

    if n_ok == 0:
        print("%s scale=%s  no sequences completed" % (method, scale), flush=True)
        return 1

    totals = {("mean_%s" % key): metric_sums[key] / n_ok for key in ("MTE_2D", "delta_2D", "survival_2D", "endpointerror") if key in metric_sums}
    totals["num_sequences"] = n_ok
    totals["backend"] = backend
    totals["onnx"] = str(onnx_path) if onnx_path else extra.get("onnx_dir")
    metrics_by_seq["total"] = totals
    flat_lat = [x for v in latencies_by_seq.values() for x in v]
    stats = ox.latency_stats(flat_lat, skip=args.latency_skip)
    stats["pipeline"] = ox.aggregate_pipeline(pipeline_by_seq)

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.num_seq < 0 else str(args.num_seq)
    scale_tag = "scale%d" % scale
    suffix = args.jsonsuffix
    (args.outdir / ("metrics_%s_%s_%s%s.json" % (tag, method, scale_tag, suffix))).write_text(
        json.dumps(metrics_by_seq, indent=2)
    )
    (args.outdir / ("positions_%s_%s_%s%s.json" % (tag, method, scale_tag, suffix))).write_text(
        json.dumps(positionlists, indent=2)
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
        "%s  mean MTE=%.3f  delta=%.3f  surv=%.3f  EPE=%.2f  pipeline_fps=%.2f  ms/pair=%.1f  realtime=%s  seqs=%d"
        % (
            method,
            totals.get("mean_MTE_2D", float("nan")),
            totals.get("mean_delta_2D", float("nan")),
            totals.get("mean_survival_2D", float("nan")),
            totals.get("mean_endpointerror", float("nan")),
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
    parser.add_argument("--datadir", type=str, default=DEFAULT_DATADIR)
    parser.add_argument("--num_seq", type=int, default=-1, help="-1 = all annotated sequences")
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

    sequences = load_sequences(datadir=args.datadir)
    if args.num_seq >= 0:
        sequences = sequences[: args.num_seq]
    if not sequences:
        print("no StereoMIS sequences under", args.datadir, file=sys.stderr)
        return 1
    print(
        "datadir=%s  sequences=%d  %s"
        % (args.datadir, len(sequences), [(s.name, s.num_points, s.num_frames) for s in sequences]),
        flush=True,
    )
    return eval_one(method, args.scale, sequences, args)


if __name__ == "__main__":
    sys.exit(main())
