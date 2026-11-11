"""ONNX RAFT-OU flower with the same ``compute_flow`` contract as MFT.RAFTWrapper.

MFT's published graph is only the optical-flow network. Delta selection, flow
chaining, and occlusion/sigma propagation stay on the host (Holoscan operator
or this flower + ``MFT.track``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from onnx_hosts.run_all_onnx_stir import _ort_session, _squeeze_flow


def _bgr_uint8_to_nchw_rgb(img: np.ndarray) -> np.ndarray:
    rgb = img[:, :, ::-1]
    chw = np.transpose(rgb.astype(np.float32), (2, 0, 1))
    return np.ascontiguousarray(chw[None])


def _as_1hw(arr: np.ndarray) -> torch.Tensor:
    arr = np.asarray(arr)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        pass
    elif arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3 and arr.shape[0] != 1:
        arr = arr[:1]
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


class OnnxRaftOuFlower:
    """Drop-in for ``MFT.raft.RAFTWrapper`` used by ``MFT.track``."""

    def __init__(self, onnx_path: Path, device: Optional[torch.device] = None):
        self.onnx_path = Path(onnx_path)
        self.sess = _ort_session(self.onnx_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_latency_ms = 0.0
        self.call_latencies_ms: list = []

    def compute_flow(
        self,
        src_img,
        dst_img,
        mode="flow",
        vis=False,
        src_img_identifier=None,
        numpy_out=False,
        init_flow=None,
        vis_debug=False,
    ):
        del vis, src_img_identifier, vis_debug
        if init_flow is not None:
            raise RuntimeError("ONNX RAFT-OU has no flow_init input")
        if mode != "flow":
            raise RuntimeError("OnnxRaftOuFlower only implements mode='flow'")

        image1 = _bgr_uint8_to_nchw_rgb(src_img)
        image2 = _bgr_uint8_to_nchw_rgb(dst_img)
        import time

        t0 = time.perf_counter()
        flow, occlusion, sigma = self.sess.run(
            ["flow", "occlusion", "sigma"],
            {"image1": image1, "image2": image2},
        )
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.call_latencies_ms.append(self.last_latency_ms)

        flow = _squeeze_flow(np.asarray(flow))
        flow_t = torch.from_numpy(np.ascontiguousarray(flow)).float().to(self.device)
        occ_t = _as_1hw(occlusion).to(self.device)
        sig_t = _as_1hw(sigma).to(self.device)
        if numpy_out:
            flow_t = flow_t.cpu()
            occ_t = occ_t.cpu()
            sig_t = sig_t.cpu()
        return flow_t, {"occlusion": occ_t, "sigma": sig_t}
