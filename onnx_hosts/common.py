"""Shared helpers for the STIR and StereoMIS ONNX hosts."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

# Native (H, W) used by the eval wrappers. StereoMIS is retargeted in
# stereomis_wrappers.prepare_tracker_for_stereomis (640 x 512 frames).
DATASETS = {
    "stir": {"native_hw": (1024, 1280), "fps": 26.0, "frame_budget_ms": 1000.0 / 26.0},
    "stereomis": {"native_hw": (512, 640), "fps": 15.0, "frame_budget_ms": 1000.0 / 15.0},
}
STIR_HW = DATASETS["stir"]["native_hw"]
STEREOMIS_HW = DATASETS["stereomis"]["native_hw"]

# CoTracker / PIPs++ / LocoTrack Holoscan canvas. CoTracker is trained at this
# size; PIPs stride 8 divides it; LocoTrack refines from 256x256 up to it.
WINDOWED_HW = (384, 512)

# Match LocoTrackSTIR: keep-first growing clip, cap at 300. ONNX traces a short
# dummy T and marks time dynamic so each eval step can feed the true length.
LOCOTRACK_MAX_FRAMES = 300
LOCOTRACK_TRACE_T = 2


def resolve_dataset(name: str) -> str:
    key = name.lower().replace("-", "")
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {tuple(DATASETS)}.")
    return key


def native_hw(dataset: str = "stir") -> Tuple[int, int]:
    return DATASETS[resolve_dataset(dataset)]["native_hw"]


def model_hw(
    scale: int,
    dataset: str = "stir",
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Tuple[int, int]:
    """Return (H, W) after the same integer downsample used in STIR / StereoMIS eval."""
    native_h, native_w = native_hw(dataset)
    h = (height if height is not None else native_h) // int(scale)
    w = (width if width is not None else native_w) // int(scale)
    if h < 8 or w < 8:
        raise ValueError(f"Resolved size {h}x{w} is too small for scale={scale} dataset={dataset}")
    return h, w


def sample_flow_at_points(flow: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample a dense flow field at pixel coordinates.

    Args:
        flow: [B, 2, H, W]
        points_xy: [B, N, 2] in pixel coords (x, y)
    Returns:
        [B, N, 2]
    """
    _, _, height, width = flow.shape
    denom_x = torch.clamp(flow.new_tensor(float(width - 1)), min=1.0)
    denom_y = torch.clamp(flow.new_tensor(float(height - 1)), min=1.0)
    grid_x = 2.0 * points_xy[..., 0] / denom_x - 1.0
    grid_y = 2.0 * points_xy[..., 1] / denom_y - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        flow,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).permute(0, 2, 1)


class FlowThenSample(nn.Module):
    """Wrap a dense-flow network so ONNX returns point displacements.

    The inner module must map (image1, image2) -> flow [B, 2, H, W].
    Optional extra outputs (occlusion, visconf, ...) are dropped.
    """

    def __init__(self, flow_module: nn.Module):
        super().__init__()
        self.flow_module = flow_module

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, points: torch.Tensor):
        result = self.flow_module(image1, image2)
        flow = result[0] if isinstance(result, (tuple, list)) else result
        sampled = sample_flow_at_points(flow, points)
        return points + sampled


@dataclass
class GraphSpec:
    method: str
    dataset: str
    scale: int
    height: int
    width: int
    onnx_path: str
    input_names: List[str]
    output_names: List[str]
    dummy_shapes: Dict[str, List[int]]
    opset: int
    iters: Optional[int] = None
    notes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def write_sidecar(self) -> Path:
        sidecar = Path(self.onnx_path).with_suffix(".json")
        sidecar.write_text(json.dumps(asdict(self), indent=2))
        return sidecar


def freeze_eval(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def cotracker_num_points_axes(
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> Dict[str, Dict[int, str]]:
    """Point count N is dynamic. T, H, W, and the feature ranks stay fixed.

    queries is (1, N, 3). Track state is (1, T, N, ...) or (1, levels, ..., N, dim).
    """
    point_axis = {
        "queries": 1,
        "previous_coords": 2,
        "previous_visibility": 2,
        "previous_confidence": 2,
        "previous_track_features": 3,
        "previous_support_features": 3,
        "tracks": 2,
        "new_coords": 2,
        "new_visibility": 2,
        "new_confidence": 2,
        "new_track_features": 3,
        "new_support_features": 3,
    }
    axes: Dict[str, Dict[int, str]] = {}
    for name in list(input_names) + list(output_names):
        axis = point_axis.get(name)
        if axis is not None:
            axes[name] = {axis: "num_points"}
    return axes


def default_dynamic_axes(
    input_names: Sequence[str],
    output_names: Sequence[str],
    point_inputs: Sequence[str] = ("points", "queries"),
) -> Dict[str, Dict[int, str]]:
    """Dynamic batch (and point count when present). TensorRT still prefers static."""
    axes: Dict[str, Dict[int, str]] = {}
    for name in list(input_names) + list(output_names):
        axes[name] = {0: "batch"}
        if name in point_inputs:
            axes[name][1] = "num_points"
    return axes


def locotrack_feature_chunk(height: int, width: int) -> int:
    """Frames per ResNet call.

    A single fp32 forward of the full clip fills a 40 GB GPU near T=60 at
    256x320, because every ResNet activation stays live. Chunking the encoder
    keeps that peak at a short clip; the tracker still sees the whole buffer.
    """
    pixels = max(int(height) * int(width), 1)
    chunk = int(round(24 * (256 * 320) / pixels))
    return max(2, min(32, chunk))


def locotrack_time_dynamic_axes(
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> Dict[str, Dict[int, str]]:
    """Mark clip length T dynamic so ORT can grow the keep-first buffer like PyTorch."""
    axes: Dict[str, Dict[int, str]] = {}
    for name in input_names:
        if name in {"window", "video"} or name.startswith("feat_"):
            axes[name] = {1: "time"}
    for name in output_names:
        if name == "tracks":
            axes[name] = {2: "time"}
        elif name == "occlusion" or name.startswith("feat_"):
            axes[name] = {2: "time"} if name == "occlusion" else {1: "time"}
    return axes


def keep_first_growing_clip(frames: Sequence, t: int, max_frames: int = LOCOTRACK_MAX_FRAMES) -> list:
    """Prefix frames[0:t+1] with queries anchored at frame 0; drop the middle if over cap.

    Same rule as ``LocoTrackSTIR.trackpoints2D`` (keep-first, max_frames=300).
    Does not pad. Pad only when the ONNX graph has a static T.
    """
    sl = list(frames[: t + 1])
    cap = max(int(max_frames), 2)
    if len(sl) > cap:
        sl = [sl[0]] + sl[-(cap - 1) :]
    return sl


def trace_onnx(
    model: nn.Module,
    dummy_inputs: Tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    tracer: str = "legacy",
    do_constant_folding: bool = True,
    warmup: bool = True,
    freeze_output_shapes: bool = True,
) -> Path:
    """Trace a frozen module to ONNX. Tries dynamo when requested, else the legacy tracer."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = freeze_eval(model)

    if warmup:
        with torch.inference_mode():
            _ = model(*dummy_inputs)
        del _
        if dummy_inputs and dummy_inputs[0].is_cuda:
            torch.cuda.empty_cache()

    trace_kwargs = dict(
        input_names=list(input_names),
        output_names=list(output_names),
        opset_version=opset,
        do_constant_folding=do_constant_folding,
        dynamo=tracer == "dynamo",
    )
    if dynamic_axes:
        trace_kwargs["dynamic_axes"] = dynamic_axes

    try:
        torch.onnx.export(model, dummy_inputs, str(output_path), **trace_kwargs)
    except TypeError:
        trace_kwargs.pop("dynamo", None)
        torch.onnx.export(model, dummy_inputs, str(output_path), **trace_kwargs)
    except Exception:
        if tracer == "dynamo":
            trace_kwargs["dynamo"] = False
            torch.onnx.export(model, dummy_inputs, str(output_path), **trace_kwargs)
        else:
            raise

    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(output_path)))
    except ImportError:
        pass

    with torch.inference_mode():
        traced_out = model(*dummy_inputs)
    if isinstance(traced_out, torch.Tensor):
        traced_out = (traced_out,)
    if freeze_output_shapes:
        freeze_onnx_output_shapes(output_path, output_names, traced_out)
    return output_path


def freeze_onnx_output_shapes(
    onnx_path: Path,
    output_names: Sequence[str],
    outputs: Sequence[torch.Tensor],
) -> None:
    """Replace symbolic output dims (ScatterND/ReduceMean) with the traced static shape.

    TensorRT and Holoscan InferenceOp both need concrete H/W/T/N at engine build.
    """
    try:
        import onnx
        from onnx import TensorProto
    except ImportError:
        return

    model = onnx.load(str(onnx_path))
    name_to_shape = {
        name: [int(d) for d in tensor.shape] for name, tensor in zip(output_names, outputs)
    }
    for value_info in model.graph.output:
        shape = name_to_shape.get(value_info.name)
        if not shape:
            continue
        value_info.type.tensor_type.elem_type = TensorProto.FLOAT
        del value_info.type.tensor_type.shape.dim[:]
        for size in shape:
            dim = value_info.type.tensor_type.shape.dim.add()
            dim.dim_value = int(size)
    onnx.save(model, str(onnx_path))


def assemble_locotrack_feeds(track_sess, feat_sess, window, queries, chunk: int) -> dict:
    """Run the feature graph in short chunks, then build the tracker feeds.

    ``window`` is NTCHW float32. Feature tensors are concatenated on dim 1.
    """
    import numpy as np

    feeds = {
        "window": np.ascontiguousarray(window),
        "queries": np.ascontiguousarray(queries),
    }
    if feat_sess is None:
        return feeds
    step = max(int(chunk), 1)
    length = int(window.shape[1])
    parts = None
    for start in range(0, length, step):
        piece = feat_sess.run(None, {"window": np.ascontiguousarray(window[:, start : start + step])})
        if parts is None:
            parts = [[arr] for arr in piece]
        else:
            for index, arr in enumerate(piece):
                parts[index].append(arr)
    feats = [group[0] if len(group) == 1 else np.concatenate(group, axis=1) for group in parts]
    produced = [out.name for out in feat_sess.get_outputs()]
    if len(produced) != len(feats):
        raise RuntimeError("feature graph returned %d tensors, expected %d" % (len(feats), len(produced)))
    by_name = {name: arr for name, arr in zip(produced, feats)}
    feat_inputs = [inp.name for inp in track_sess.get_inputs() if str(inp.name).startswith("feat_")]
    missing = [name for name in feat_inputs if name not in by_name]
    if missing:
        raise RuntimeError("tracker features missing from the encoder: %s" % missing)
    for name in feat_inputs:
        feeds[name] = np.ascontiguousarray(by_name[name])
    accepted = {inp.name for inp in track_sess.get_inputs()}
    return {name: value for name, value in feeds.items() if name in accepted}


def describe_file(path: Path) -> str:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return f"{path} ({size_mb:.1f} MiB)"
