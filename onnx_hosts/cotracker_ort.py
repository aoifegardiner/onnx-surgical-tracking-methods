"""Clara AGX ONNX Runtime path for the CoTracker online step graph.

The step graph fixes T and the iteration count. N, the number of tracked
points, is dynamic: each sequence feeds its own queries. On Xavier
the useful session knobs are:

* FP16 weights/activations so Volta tensor cores run the backbone
* CUDA graphs, which cut launch overhead between the many small kernels
* IO binding so ``previous_*`` stays on the GPU between windows

``use_tf32`` is left unset: Volta has no TF32. TensorRT is not used; the
step graph's GridSample nodes are rejected by TensorRT 8.4 on L4T R34.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_STATE_PAIRS = (
    ("previous_coords", "new_coords"),
    ("previous_visibility", "new_visibility"),
    ("previous_confidence", "new_confidence"),
    ("previous_track_features", "new_track_features"),
    ("previous_support_features", "new_support_features"),
)


def lock_jetson_clocks() -> bool:
    """Hold MAXN clocks for a benchmark. No-op when ``jetson_clocks`` is absent."""
    exe = shutil.which("jetson_clocks")
    if exe is None:
        print("jetson_clocks not on PATH; GPU may idle near 319 MHz until the first forward", flush=True)
        return False
    proc = subprocess.run([exe], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        print("jetson_clocks exited %s: %s" % (proc.returncode, (proc.stderr or proc.stdout).strip()), flush=True)
        return False
    print("jetson_clocks: clocks locked", flush=True)
    return True


def _np_dtype(ort_type: str):
    if ort_type in ("tensor(float16)", "float16"):
        return np.float16
    return np.float32


def make_cotracker_session(onnx_path: Path, cuda_graph: bool = True):
    """ORT session for the step graph. CUDA graphs are optional and Volta-safe."""
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


def _ort_dim(value):
    """Positive ints are static. A name or 0 means the axis is dynamic."""
    if isinstance(value, str) or value is None:
        return None
    value = int(value)
    return value if value > 0 else None


class CoTrackerStepRunner:
    """One step graph. Recurrent ``previous_*`` buffers stay on device between windows.

    CUDA graph replay needs fixed addresses, so it is used only when N is static.
    A dynamic point axis reallocates those buffers when the sequence's point count changes.
    """

    def __init__(self, onnx_path: Path, cuda_graph: bool = True):
        self.onnx_path = Path(onnx_path)
        self.input_shapes: Dict[str, list] = {}
        self.output_shapes: Dict[str, list] = {}
        self._active_n: Optional[int] = None
        self.cuda_graph = bool(cuda_graph)
        self.sess = make_cotracker_session(self.onnx_path, cuda_graph=self.cuda_graph)
        self._bind_cuda = self._can_bind_cuda()
        self._read_io()
        if self.n_points is None:
            # Replay cannot change N. Run the same graph with fresh bindings per count.
            self.cuda_graph = False
            if self._bind_cuda:
                self.sess = make_cotracker_session(self.onnx_path, cuda_graph=False)
                self._read_io()
        elif self.cuda_graph and not self._bind_cuda:
            self.cuda_graph = False
            self.sess = make_cotracker_session(self.onnx_path, cuda_graph=False)
            self._read_io()
        self._inputs: Dict[str, object] = {}
        self._outputs: Dict[str, object] = {}
        self._binding = None
        self._captured = False
        if self._bind_cuda and self.n_points is not None:
            self._alloc_cuda(int(self.n_points))
            try:
                self._warmup_capture()
            except Exception as exc:
                print("CUDA graph capture failed (%s); rerunning without it" % exc, flush=True)
                self.close()
                self.cuda_graph = False
                self.sess = make_cotracker_session(self.onnx_path, cuda_graph=False)
                self._read_io()
                self._bind_cuda = self._can_bind_cuda()
                if self._bind_cuda:
                    self._alloc_cuda(int(self.n_points))
                self._captured = False

    def _read_io(self):
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        self.output_names = [out.name for out in self.sess.get_outputs()]
        self.np_dtype = _np_dtype(self.sess.get_inputs()[0].type)
        self.input_shapes = {
            inp.name: [_ort_dim(d) for d in inp.shape] for inp in self.sess.get_inputs()
        }
        self.output_shapes = {
            out.name: [_ort_dim(d) for d in out.shape] for out in self.sess.get_outputs()
        }
        self.n_points = self.input_shapes["queries"][1]
        self.window_len = int(self.input_shapes["window"][1])

    def _concrete_shapes(self, n_points: int) -> Dict[str, list]:
        concrete = {}
        for name, shape in {**self.input_shapes, **self.output_shapes}.items():
            dims = []
            for dim in shape:
                if dim is None:
                    dims.append(int(n_points))
                else:
                    dims.append(int(dim))
            concrete[name] = dims
        return concrete

    def get_providers(self):
        providers = list(self.sess.get_providers())
        tag = "cuda_graph" if self.cuda_graph and self._captured else "iobind" if self._bind_cuda else "numpy"
        return providers + [tag]

    def close(self):
        self._binding = None
        self._inputs.clear()
        self._outputs.clear()
        self.sess = None

    def reset_state(self):
        if self._bind_cuda:
            for name, _new in _STATE_PAIRS:
                self._inputs[name].zero_()
            return
        if hasattr(self, "_host_state"):
            for name in self._host_state:
                self._host_state[name].fill(0)

    def snapshot_state(self):
        """Copy recurrent state so the next point-batch can reuse these buffers."""
        if self._bind_cuda:
            return {name: self._inputs[name].detach().clone() for name, _new in _STATE_PAIRS}
        state = getattr(self, "_host_state", None)
        if state is None:
            return None
        return {name: np.ascontiguousarray(value) for name, value in state.items()}

    def load_state(self, state):
        if state is None:
            self.reset_state()
            return
        if self._bind_cuda:
            for name, _new in _STATE_PAIRS:
                self._inputs[name].copy_(state[name])
            return
        self._host_state = {name: np.ascontiguousarray(value) for name, value in state.items()}

    def _prepare_points(self, n_points: int):
        n_points = int(n_points)
        if self._active_n == n_points and (self._inputs or getattr(self, "_host_state", None)):
            return
        self._active_n = n_points
        if self._bind_cuda:
            self._captured = False
            self._alloc_cuda(n_points)
            return
        self._host_state = {
            name: np.zeros(self._host_shape(name), dtype=self.np_dtype) for name, _ in _STATE_PAIRS
        }

    def run_all_points(self, window: np.ndarray, pts_xy: np.ndarray, states):
        """Track these queries. N is whatever the sequence passed in."""
        pts_xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
        if pts_xy.shape[0] == 0:
            time = int(window.shape[1])
            return np.zeros((1, time, 0, 2), dtype=np.float32), []
        n = int(pts_xy.shape[0])
        if self.n_points is not None and n != int(self.n_points):
            raise ValueError(
                "CoTracker step graph has a fixed N=%d, so it cannot run %d points. "
                "Rebuild cotracker_online_step.onnx with a dynamic point axis."
                % (self.n_points, n)
            )
        self._prepare_points(n)
        state = states[0] if states else None
        self.load_state(state)
        queries = self._queries_from_xy(pts_xy)
        out = self.run_window(window, queries)
        return out["tracks"], [self.snapshot_state()]

    def _queries_from_xy(self, pts: np.ndarray):
        take = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        queries = np.zeros((1, take.shape[0], 3), dtype=self.np_dtype)
        queries[0, :, 1] = take[:, 0]
        queries[0, :, 2] = take[:, 1]
        return queries

    def run_window(self, window: np.ndarray, queries: np.ndarray) -> Dict[str, np.ndarray]:
        if self._bind_cuda:
            return self._run_bound(window, queries)
        feeds = {"window": np.ascontiguousarray(window, dtype=self.np_dtype), "queries": np.ascontiguousarray(queries, dtype=self.np_dtype)}
        if not getattr(self, "_host_state", None):
            self._prepare_points(int(queries.shape[1]))
        feeds.update(self._host_state)
        outs = self.sess.run(self.output_names, feeds)
        outd = dict(zip(self.output_names, outs))
        for prev, new in _STATE_PAIRS:
            self._host_state[prev] = np.ascontiguousarray(outd[new])
        return outd

    def _host_shape(self, name: str):
        shape = self.input_shapes[name]
        if self._active_n is None:
            raise RuntimeError("CoTracker point count is not set")
        return [int(self._active_n) if dim is None else int(dim) for dim in shape]

    def _can_bind_cuda(self) -> bool:
        if "CUDAExecutionProvider" not in self.sess.get_providers():
            return False
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def _alloc_cuda(self, n_points: int):
        import torch

        self._torch_dtype = torch.float16 if self.np_dtype == np.float16 else torch.float32
        concrete = self._concrete_shapes(int(n_points))
        self._inputs = {
            name: torch.empty(concrete[name], dtype=self._torch_dtype, device="cuda")
            for name in self.input_shapes
        }
        for tensor in self._inputs.values():
            tensor.zero_()
        self._outputs = {
            name: torch.empty(concrete[name], dtype=self._torch_dtype, device="cuda")
            for name in self.output_shapes
        }
        self._binding = self.sess.io_binding()
        self._bind_all()
        self._active_n = int(n_points)

    def _bind_all(self):
        binding = self._binding
        for name, tensor in self._inputs.items():
            binding.bind_input(
                name,
                "cuda",
                0,
                self.np_dtype,
                list(tensor.shape),
                tensor.data_ptr(),
            )
        for name, tensor in self._outputs.items():
            binding.bind_output(
                name,
                "cuda",
                0,
                self.np_dtype,
                list(tensor.shape),
                tensor.data_ptr(),
            )

    def _warmup_capture(self):
        window = np.zeros(tuple(self._inputs["window"].shape), dtype=self.np_dtype)
        queries = np.zeros(tuple(self._inputs["queries"].shape), dtype=self.np_dtype)
        self._run_bound(window, queries)
        self.reset_state()
        self._captured = True

    def _run_bound(self, window: np.ndarray, queries: np.ndarray) -> Dict[str, np.ndarray]:
        import torch

        self._copy_input("window", window)
        self._copy_input("queries", queries)
        self.sess.run_with_iobinding(self._binding)
        torch.cuda.synchronize()
        for prev, new in _STATE_PAIRS:
            self._inputs[prev].copy_(self._outputs[new])
        tracks = self._outputs["tracks"].detach().float().cpu().numpy()
        return {"tracks": tracks}

    def _copy_input(self, name: str, array: np.ndarray):
        import torch

        src = torch.from_numpy(np.ascontiguousarray(array))
        if src.dtype != self._torch_dtype:
            src = src.to(self._torch_dtype)
        self._inputs[name].copy_(src)


def convert_step_to_fp16(src: Path, dst: Path) -> Path:
    """Write an FP16 step graph. GridSample and LayerNormalization stay FP32.

    Those two ops are the ones most likely to move end-point error. The
    backbone matmuls and convolutions become FP16 so Xavier tensor cores run.
    """
    import onnx

    model = onnx.load(str(src))
    converted = None
    errors = []
    try:
        from onnxconverter_common import float16

        converted = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
            op_block_list=["GridSample", "LayerNormalization"],
        )
    except Exception as exc:
        errors.append("onnxconverter_common: %s" % exc)
    if converted is None:
        try:
            from onnxruntime.transformers.float16 import convert_float_to_float16

            converted = convert_float_to_float16(
                model,
                keep_io_types=False,
                op_block_list=["GridSample", "LayerNormalization"],
            )
        except Exception as exc:
            errors.append("onnxruntime.transformers: %s" % exc)
    if converted is None:
        raise RuntimeError("FP16 conversion failed: " + "; ".join(errors))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, str(dst))
    return dst
