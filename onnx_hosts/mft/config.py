"""Runtime settings for the ONNX MFT host.

The published graph is RAFT-OU only. These are the MFT delta-chain settings:
deltas ``[inf, 1, 2, 4, 8, 16, 32]`` and an occlusion threshold of 0.02.
The PyTorch RAFT flower config is not loaded.
"""
import numpy as np


class MFTConfig:
    def __init__(self, occlusion_threshold=0.02):
        self.deltas = [np.inf, 1, 2, 4, 8, 16, 32]
        self.occlusion_threshold = float(occlusion_threshold)
        self.timers_enabled = False
        self.cache_delta_infinity = False
