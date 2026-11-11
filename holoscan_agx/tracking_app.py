#!/usr/bin/env python3
"""Holoscan application skeleton for STIR and StereoMIS trackers on Clara AGX.

Run this on the AGX after converting ONNX -> TensorRT:

    python holoscan_agx/tracking_app.py \\
        --dataset stir --model RAFT --scale 4 \\
        --onnx models/RAFT/raft_stir_scale4.onnx \\
        --source /data/stir_clip

    python holoscan_agx/tracking_app.py \\
        --dataset stereomis --model RAFT --scale 4 \\
        --onnx models/RAFT/raft_stereomis_scale4.onnx \\
        --source "$STEREOMIS_DATA/P1_1/video_frames"

The graph mirrors flow2d.py / flow2d_stereomis.py:
    source -> resize to model HxW -> (pair or clip + queries) -> InferenceOp -> sample -> metrics

Dense flow (RAFT/WAFT/MFT): TensorRT ok if GridSample is supported; otherwise onnxrt.
PIPS / CoTracker: Holoscan ``backend: onnxrt`` — Xavier TensorRT rejects GridSample.
CoTracker runs in ``CoTrackerHostOp`` (CUDA graph, GPU ``previous_*``), not InferenceOp.
CoTracker / PIPS online graphs are 384x512 (shared Holoscan canvas).
LocoTrack uses ``LocoTrackHostOp`` (ORT, not InferenceOp) so the clip length can
grow each frame up to 300, matching LocoTrackSTIR. Queries (t,x,y) stay at t=0.
Eval-scale HxW matches PyTorch. PIPS one step graph carries previous_feats*.

StereoMIS is left-camera 512x640 at 15 fps. Scale 4 is 128x160.
STIR is 1024x1280 at 26 fps. Scale 4 is 256x320.
StereoMIS scale 2 and STIR scale 4 share 256x320, so those RAFT/MFT engines can be reused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from onnx_hosts.common import DATASETS, model_hw, native_hw, resolve_dataset


def _load_queries(path: Path | None, n: int = 4) -> "np.ndarray":
    import numpy as np

    if path is None:
        return np.zeros((0, 2), dtype=np.float32)
    if path.suffix == ".json":
        import json

        data = json.loads(path.read_text())
        return np.asarray(data, dtype=np.float32).reshape(-1, 2)
    return np.load(str(path)).astype(np.float32).reshape(-1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="RAFT, WAFT, MFT, ALLTRACKER, PIPS, LOCOTRACK, COTRACKER")
    parser.add_argument("--dataset", default="stir", choices=["stir", "stereomis"])
    parser.add_argument("--scale", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--source", type=str, required=True, help="Directory of frames or a video file.")
    parser.add_argument("--queries", type=Path, default=None, help="Nx2 float32 numpy/json query points in native pixels.")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--framerate", type=float, default=None)
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()
    args.dataset = resolve_dataset(args.dataset)
    if args.framerate is None:
        args.framerate = DATASETS[args.dataset]["fps"]
    if args.outdir is None:
        args.outdir = _WORKSPACE / "outputs" / ("agx_stereomis" if args.dataset == "stereomis" else "agx_stir")
    return args


def compose_app(args: argparse.Namespace):
    from holoscan.core import Application
    from holoscan.operators import (
        FormatConverterOp,
        HolovizOp,
        InferenceOp,
        VideoStreamReplayerOp,
    )
    from holoscan.resources import UnboundedAllocator, CudaStreamPool

    from holoscan_agx.operators import (
        ClipBufferOp,
        FramePairOp,
        MetricsSinkOp,
        PadQueriesOp,
        SampleFlowOp,
    )

    model = args.model.upper()
    sidecar = args.onnx.with_suffix(".json")
    input_names = ["image1", "image2"]
    output_names = ["flow"]
    spec = {}
    if sidecar.exists():
        import json

        spec = json.loads(sidecar.read_text())
        input_names = spec["input_names"]
        output_names = spec["output_names"]

    if model in {"COTRACKER", "PIPS"}:
        height = int(spec.get("height") or spec.get("extra", {}).get("height") or 384)
        width = int(spec.get("width") or spec.get("extra", {}).get("width") or 512)
    else:
        height, width = model_hw(args.scale, dataset=args.dataset)
    if spec.get("height") and spec.get("width"):
        height = int(spec["height"])
        width = int(spec["width"])

    point = model in {"PIPS", "COTRACKER"}
    backend = "onnxrt" if point or model == "LOCOTRACK" else "trt"
    queries_xy = _load_queries(args.queries)
    native = native_hw(args.dataset)
    class TrackingApp(Application):
        def compose(self):
            allocator = UnboundedAllocator(self, name="allocator")
            cuda_stream_pool = CudaStreamPool(self, name="cuda_stream", dev_id=0)

            source = VideoStreamReplayerOp(
                self,
                name="replayer",
                directory=str(args.source),
                basename=args.dataset,
                frame_rate=args.framerate,
                repeat=False,
                realtime=False,
            )
            converter = FormatConverterOp(
                self,
                name="converter",
                pool=allocator,
                in_dtype="rgb888",
                out_dtype="rgb888",
                resize_width=width,
                resize_height=height,
            )
            inference = None
            if model not in {"MFT", "ALLTRACKER", "LOCOTRACK", "COTRACKER"}:
                inference = InferenceOp(
                    self,
                    name="inference",
                    allocator=allocator,
                    backend=backend,
                    model_path_map={model.lower(): str(args.onnx)},
                    pre_processor_map={model.lower(): input_names},
                    inference_map={model.lower(): output_names},
                    infer_on_cpu=False,
                    input_on_cuda=True,
                    output_on_cuda=True,
                )
            metrics = MetricsSinkOp(
                self,
                name="metrics",
                sequence_name=f"{model}_{args.dataset}_scale{args.scale}",
                out_dir=str(args.outdir / model / f"scale{args.scale}"),
            )

            self.add_flow(source, converter, {("output", "source_video")})

            if model in {"RAFT", "WAFT"}:
                pair = FramePairOp(self, name="pair", height=height, width=width)
                sample = SampleFlowOp(self, name="sample", scale=args.scale, max_points=64)
                self.add_flow(converter, pair, {("tensor", "in")})
                self.add_flow(pair, inference, {("image1", input_names[0]), ("image2", input_names[1])})
                self.add_flow(inference, sample, {(output_names[0], "flow")})
                self.add_flow(sample, metrics, {("coords", "in")})
            elif model == "MFT":
                # RAFT-OU ONNX is not the full tracker. MFTHostOp runs official
                # delta chaining and calls the engine once per (t-delta, t) pair.
                from holoscan_agx.mft_host_op import MFTHostOp

                mft_host = MFTHostOp(
                    self,
                    name="mft_host",
                    onnx_path=str(args.onnx),
                    height=height,
                    width=width,
                    scale=args.scale,
                    queries=queries_xy,
                    native_hw=native,
                )
                self.add_flow(converter, mft_host, {("tensor", "in")})
                self.add_flow(mft_host, metrics, {("coords", "in")})
            elif model == "ALLTRACKER":
                from holoscan_agx.alltracker_host_op import AllTrackerHostOp

                at_host = AllTrackerHostOp(
                    self,
                    name="alltracker_host",
                    onnx_path=str(args.onnx),
                    height=height,
                    width=width,
                    scale=args.scale,
                    queries=queries_xy,
                    native_hw=native,
                    extra=spec.get("extra") or {},
                )
                self.add_flow(converter, at_host, {("tensor", "in")})
                self.add_flow(at_host, metrics, {("coords", "in")})
            elif model == "LOCOTRACK":
                from holoscan_agx.locotrack_host_op import LocoTrackHostOp

                loco_host = LocoTrackHostOp(
                    self,
                    name="locotrack_host",
                    onnx_path=str(args.onnx),
                    height=height,
                    width=width,
                    scale=args.scale,
                    queries=queries_xy,
                    native_hw=native,
                    extra=spec.get("extra") or {},
                )
                self.add_flow(converter, loco_host, {("tensor", "in")})
                self.add_flow(loco_host, metrics, {("coords", "in")})
            elif model == "COTRACKER":
                from holoscan_agx.cotracker_host_op import CoTrackerHostOp

                cot_host = CoTrackerHostOp(
                    self,
                    name="cotracker_host",
                    onnx_path=str(args.onnx),
                    height=height,
                    width=width,
                    queries=queries_xy,
                    native_hw=native,
                    extra=spec.get("extra") or {},
                )
                self.add_flow(converter, cot_host, {("tensor", "in")})
                self.add_flow(cot_host, metrics, {("coords", "in")})
            else:
                extra = spec.get("extra") or {}
                if model == "PIPS":
                    max_len = int(extra.get("S", 128))
                    layout = "ntchw"
                    keep_first = False
                    q_layout = "trajs"
                else:
                    max_len = int(extra.get("window_len", 16))
                    layout = "ntchw"
                    keep_first = False
                    q_layout = "txy"
                clip = ClipBufferOp(
                    self,
                    name="clip",
                    height=height,
                    width=width,
                    max_len=max_len,
                    layout=layout,
                    keep_first=keep_first,
                    minus_one_one=False,
                    min_len=max_len,
                    pad_to_max=True,
                )
                qop = PadQueriesOp(
                    self,
                    name="queries",
                    queries=queries_xy,
                    max_points=int(extra.get("N", 64)),
                    native_hw=native,
                    model_hw=(height, width),
                    layout=q_layout,
                    window=max_len,
                )
                self.add_flow(converter, clip, {("tensor", "in")})
                self.add_flow(clip, qop, {("clip", "tick")})
                if model == "PIPS":
                    self.add_flow(qop, inference, {("queries", "trajs_e0")})
                    self.add_flow(clip, inference, {("clip", "window")})
                else:
                    self.add_flow(clip, inference, {("clip", "window")})
                    self.add_flow(qop, inference, {("queries", "queries")})
                self.add_flow(inference, metrics, {(output_names[0], "in")})

            if args.viz:
                viz = HolovizOp(self, name="viz", allocator=allocator, cuda_stream_pool=cuda_stream_pool)
                self.add_flow(converter, viz, {("tensor", "receivers")})

    return TrackingApp()


def main() -> None:
    args = parse_args()
    try:
        import holoscan  # noqa: F401
    except ImportError:
        height, width = model_hw(args.scale, dataset=args.dataset)
        print(
            "Holoscan SDK is not installed in this environment. "
            "Copy holoscan_agx/ onto the Clara AGX and run there.\n"
            f"Planned graph: {args.dataset} {args.model} scale={args.scale}  "
            f"{height}x{width}  {args.framerate:g} fps  onnx={args.onnx}"
        )
        if args.model.upper() == "LOCOTRACK":
            print(
                "LocoTrack: converter -> LocoTrackHostOp (keep-first growing T<=300, ORT) "
                "-> MetricsSinkOp. InferenceOp is not used (variable clip length)."
            )
        if args.model.upper() == "COTRACKER":
            print(
                "CoTracker: converter -> CoTrackerHostOp (FP16/FP32 step graph, CUDA graph, "
                "previous_* kept on GPU). TensorRT is not used (GridSample)."
            )
        print("See the AGX Holoscan plan canvas for the operator map, TensorRT steps, and metrics.")
        return

    app = compose_app(args)
    app.run()


if __name__ == "__main__":
    main()
