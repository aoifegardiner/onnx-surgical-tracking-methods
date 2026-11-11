"""ONNX LiteTracker host.

The published graphs are ``fnet``, ``corr_mlp``, and ``updateformer``.
This module keeps the online buffers, correlation sampling, and positional
encoding that those graphs do not contain.

Adapted from LiteTracker (Karaoglu et al., MICCAI 2025), CC BY-NC 4.0.
See litetracker_LICENSE.md in this directory.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F


def posenc(x, min_deg: int, max_deg: int):
    """
    Concatenate input tensor `x` with its positional encoding using scales 2^[min_deg, max_deg-1].

    This function computes a positional encoding for `x` by applying sinusoidal functions at multiple frequencies.
    Instead of separately computing [sin(x), cos(x)], it uses the trig identity cos(x) = sin(x + pi/2)
    to perform a single vectorized call to sin([x, x + pi/2]).

    Args:
        x (torch.Tensor): Input tensor to be encoded. Should be in the range [-pi, pi].
        min_deg (int): Minimum (inclusive) degree of the encoding.
        max_deg (int): Maximum (exclusive) degree of the encoding.

    Returns:
        torch.Tensor: Concatenation of the input and its positional encoding.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
    )

    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """
    Generate a 1D sine-cosine positional embedding from a grid of positions.

    This function computes a 1D positional embedding using sine and cosine functions,
    given an embedding dimension and a tensor of positions.

    Args:
        embed_dim (int): The embedding dimension (must be even).
        pos (torch.Tensor): Positions to generate the embedding from, shape (M,) or compatible.

    Returns:
        torch.Tensor: The generated 1D positional embedding of shape (1, M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


def bilinear_sampler(
    input, coords, align_corners: bool = True, padding_mode: str = "border"
):
    """
    Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (torch.Tensor): batch of input images.
        coords (torch.Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        torch.Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[:, :, :, :, [1, 2, 0]]

    if align_corners:
        scales = []
        for i in range(len(sizes)):
            size = sizes[len(sizes) - 1 - i]

            scales.append(torch.tensor(2.0 / max(size - 1, 1), device=coords.device))

        scale_tensor = torch.stack(scales)

        coords = coords * scale_tensor

    else:
        scales = []
        for i in range(len(sizes)):
            size = sizes[len(sizes) - 1 - i]

            scales.append(torch.tensor(2.0 / size, device=coords.device))

        scale_tensor = torch.stack(scales)

        coords = coords * scale_tensor

    coords -= 1

    return F.grid_sample(
        input, coords, align_corners=align_corners, padding_mode=padding_mode
    )


def sample_features5d(input, coords):
    """
    Sample spatio-temporal features from a 5D input tensor at given coordinates.

    This function samples features from a 5D tensor `input` of shape (B, T, C, H, W)
    at the specified spatio-temporal coordinates `coords` of shape (B, R1, R2, 3),
    where each coordinate is (t, x, y). The output is a tensor of sampled features
    with shape (B, R1, R2, C).

    Args:
        input (torch.Tensor): Spatio-temporal feature tensor of shape (B, T, C, H, W).
        coords (torch.Tensor): Spatio-temporal coordinates of shape (B, R1, R2, 3),
            where each coordinate is (t, x, y).

    Returns:
        torch.Tensor: Sampled features of shape (B, R1, R2, C).
    """

    B, T, _, _, _ = input.shape

    # B T C H W -> B C T H W
    input = input.permute(0, 2, 1, 3, 4)

    # B R1 R2 3 -> B R1 R2 1 3
    coords = coords.unsqueeze(3)

    # B C R1 R2 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 3, 1, 4).view(
        B, feats.shape[2], feats.shape[3], feats.shape[1]
    )  # B C R1 R2 1 -> B R1 R2 C


class OnnxLiteTracker:
    """
    A drop-in replacement for LiteTracker that uses ONNX Runtime for inference.

    This class replicates the full stateful tracking pipeline of LiteTracker.forward(),
    but uses ONNX models for the three neural network forward passes (fnet, corr_mlp,
    updateformer). Sampling, correlation, and the online buffers stay on ``device``
    (CUDA when the CUDA provider is active). Only the three session inputs cross to
    NumPy.

    Usage:
        tracker = OnnxLiteTracker("./onnx_models")
        for frame in video_frames:
            coords, vis, conf = tracker(frame_tensor, queries_tensor)
    """

    def __init__(
        self,
        onnx_dir: str,
        window_len: int = 16,
        stride: int = 4,
        corr_radius: int = 3,
        corr_levels: int = 4,
        model_resolution: tuple = (384, 512),
        iters: int = 1,
        input_dim: int = 1110,
        latent_dim: int = 128,
        providers: list = None,
        device=None,
        max_step=None,
    ):
        """
        Initialize the ONNX-based LiteTracker.

        Args:
            onnx_dir (str): Directory containing the ONNX models.
            window_len (int): Max length of the temporal window for tracking.
            stride (int): Stride scale for feature extraction.
            corr_radius (int): Radius for correlation computation.
            corr_levels (int): Number of correlation pyramid levels.
            model_resolution (tuple): Resolution to which input frames are resized; (H, W).
            iters (int): Number of update iterations per frame.
            input_dim (int): Dimension of the transformer input tokens.
            latent_dim (int): Dimension of the feature encoder output channels.
            providers (list): ONNX Runtime execution providers. Defaults to CUDA + CPU.
            device: Torch device for sampling, correlation, and the online buffers.
                Defaults to CUDA when a CUDA provider is requested and available, so
                those ops stay next to the PyTorch tracker instead of following the
                NumPy copies back to the CPU.
            max_step: Cap on the per-frame coordinate update, in feature pixels.
                ``None`` reads ``LITETRACKER_MAX_STEP`` and leaves the update
                uncapped when that variable is unset. Pass 8 to match the
                StereoMIS and STIR CUDA run.
        """
        import onnxruntime as ort

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.providers = list(providers)
        self.device = self._resolve_device(device, self.providers)
        if max_step is None:
            env_step = os.environ.get("LITETRACKER_MAX_STEP")
            max_step = float(env_step) if env_step else None
        else:
            max_step = float(max_step)
        self.max_step = max_step
        self.window_len = window_len
        self.stride = stride
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.model_resolution = model_resolution
        self.iters = iters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.inv_sigmoid_true_val = 4.6

        # Load ONNX sessions
        self.fnet_session = ort.InferenceSession(
            os.path.join(onnx_dir, "fnet.onnx"), providers=providers
        )
        self.corr_mlp_session = ort.InferenceSession(
            os.path.join(onnx_dir, "corr_mlp.onnx"), providers=providers
        )
        self.updateformer_session = ort.InferenceSession(
            os.path.join(onnx_dir, "updateformer.onnx"), providers=providers
        )

        # Precompute time embedding (same as LiteTracker.__init__)
        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(1, window_len, 1)
        self.time_emb = get_1d_sincos_pos_embed_from_grid(input_dim, time_grid[0]).to(self.device)

        self.reset()

    @staticmethod
    def _resolve_device(device, providers):
        if device is None:
            device = os.environ.get("LITETRACKER_DEVICE") or None
        if device is not None:
            return torch.device(device)
        wants_cuda = any("CUDA" in provider or "Tensorrt" in provider for provider in providers)
        if wants_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Contiguous CPU array for ONNX Runtime. Host tensors may live on CUDA."""
        return tensor.detach().to("cpu").contiguous().numpy()

    def _from_numpy(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        return tensor

    def reset(self):
        """
        Reset all internal state, buffers, and caches for a new video sequence.

        Call this method before processing a new video to clear all temporal
        state from the previous sequence.
        """
        self.online_ind = 0
        self.ema_flow_buffer = None
        self.corr_embs_buffer = None
        self.coords_buffer = None
        self.vis_buffer = None
        self.conf_buffer = None
        self.track_feat_cache = [None] * self.corr_levels

    def _run_fnet(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Run the feature encoder via ONNX Runtime.

        Args:
            frame (torch.Tensor): Normalized input frame of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Feature map of shape [B, latent_dim, H//stride, W//stride].
        """
        result = self.fnet_session.run(
            None, {"frame": self._to_numpy(frame)}
        )
        return self._from_numpy(result[0])

    def _run_corr_mlp(self, corr_volume: torch.Tensor) -> torch.Tensor:
        """
        Run the correlation MLP via ONNX Runtime.

        Args:
            corr_volume (torch.Tensor): Flattened correlation volume of shape [N_points, r^4].

        Returns:
            torch.Tensor: Correlation embeddings of shape [N_points, 256].
        """
        result = self.corr_mlp_session.run(
            None, {"corr_volume": self._to_numpy(corr_volume)}
        )
        return self._from_numpy(result[0])

    def _run_updateformer(
        self, input_tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the update transformer via ONNX Runtime.

        Args:
            input_tensor (torch.Tensor): Transformer input of shape [B, N, T, input_dim].
            mask (torch.Tensor): Attention mask of shape [B, T, N] (bool).

        Returns:
            torch.Tensor: Delta predictions of shape [B, N, T, 4] (dx, dy, dvis, dconf).
        """
        result = self.updateformer_session.run(
            None,
            {
                "input_tensor": self._to_numpy(input_tensor),
                "mask": self._to_numpy(mask),
            },
        )
        return self._from_numpy(result[0])

    def _interpolate_time_embed(self, t: int) -> torch.Tensor:
        """
        Interpolate the temporal positional embedding to match the current window size.

        Uses linear interpolation when the current window size differs from the
        precomputed embedding length.

        Args:
            t (int): Target temporal length.

        Returns:
            torch.Tensor: Interpolated time embedding of shape [1, t, input_dim].
        """
        T = self.time_emb.shape[1]
        if t == T:
            return self.time_emb
        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb

    def _get_support_points(self, coords, r, reshape_back=True):
        """
        Generate a grid of support points around each coordinate for local feature sampling.

        Mirrors LiteTracker.get_support_points() exactly.

        Args:
            coords (torch.Tensor): Input coordinates of shape [B, T, N, 3].
            r (int): Radius for the support grid.
            reshape_back (bool): Whether to reshape the output for downstream use.

        Returns:
            torch.Tensor: Support points for each coordinate.
        """
        B, _, N, _ = coords.shape
        device = coords.device
        centroid_lvl = coords.reshape(B, N, 1, 1, 3)
        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)
        xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack((zgrid, xgrid, ygrid), dim=-1)
        delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl
        if reshape_back:
            third_dim = int((2 * r + 1) ** 2)
            return coords_lvl.reshape(B, N, third_dim, 3).permute(0, 2, 1, 3)
        else:
            return coords_lvl

    def _get_track_feat(self, fmaps, queried_coords, support_radius=0):
        """
        Extract track features at the queried coordinates and their support points.

        Mirrors LiteTracker.get_track_feat() exactly.

        Args:
            fmaps (torch.Tensor): Feature maps of shape [B, T, C, H, W].
            queried_coords (torch.Tensor): Coordinates to sample features from.
            support_radius (int): Radius for support points.

        Returns:
            tuple: (central track features, all support track features).
        """
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1][:, None]), queried_coords[:, None]],
            dim=-1,
        )
        support_points = self._get_support_points(sample_coords, support_radius)
        support_track_feats = sample_features5d(fmaps, support_points)
        return (
            support_track_feats[:, None, support_track_feats.shape[1] // 2],
            support_track_feats,
        )

    def _get_correlation_feat(self, fmaps, queried_coords):
        """
        Compute correlation features for the queried coordinates using bilinear sampling.

        Mirrors LiteTracker.get_correlation_feat() exactly.

        Args:
            fmaps (torch.Tensor): Feature maps of shape [B, T, D, H, W].
            queried_coords (torch.Tensor): Coordinates to sample features from.

        Returns:
            torch.Tensor: Correlation features of shape [B, T, N, r, r, D].
        """
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = self._get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = bilinear_sampler(
            fmaps.reshape(B * T, D, 1, H_, W_), support_points
        )
        return correlation_feat.view(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )

    def __call__(self, frame: torch.Tensor, queries: torch.Tensor):
        """
        Predict tracks for the given frame and queries.

        This replicates LiteTracker.forward() but uses ONNX models for neural network
        forward passes. All stateful orchestration (buffer management, flow momentum,
        coordinate initialization) is handled identically to the PyTorch version.

        Args:
            frame (torch.Tensor): Input frame of shape [B, C, H, W] (float32, 0-255 range).
            queries (torch.Tensor): Point queries of shape [B, N, 3]; first channel is the
                frame index, the rest are (x, y) coordinates.

        Returns:
            tuple:
                coords (torch.Tensor): Predicted coordinates [B, 1, N, 2].
                vis (torch.Tensor): Predicted visibility mask [B, 1, N].
                conf (torch.Tensor): Predicted confidence [B, 1, N].
        """
        if frame.device != self.device:
            frame = frame.to(self.device)
        if queries.device != self.device:
            queries = queries.to(self.device)

        original_shape = frame.shape
        frame = F.interpolate(
            frame, size=self.model_resolution, mode="bilinear", align_corners=True
        )
        queries_scaled = queries.clone()
        queries_scaled[:, :, 1] *= (self.model_resolution[1] - 1) / (
            original_shape[3] - 1
        )  # W
        queries_scaled[:, :, 2] *= (self.model_resolution[0] - 1) / (
            original_shape[2] - 1
        )  # H

        B, C, H, W = frame.shape
        B, N, __ = queries_scaled.shape
        frame = 2 * (frame / 255.0) - 1.0

        T = 1

        queried_frames = queries_scaled[:, :, 0].long()

        # Downscale the query coords for the smaller size feat maps
        queried_coords = queries_scaled[..., 1:3].clone()
        queried_coords = queried_coords / self.stride

        # Run feature encoder via ONNX
        fmaps = self._run_fnet(frame.float())
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), dim=-1, keepdim=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )

        # Build feature pyramid
        fmaps_pyramid = [fmaps]
        track_feat_support_pyramid = []
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)

        is_track_initialized_now = (queried_frames == self.online_ind)[
            :, None, :, None
        ]  # B 1 N 1

        for i in range(self.corr_levels):
            if self.online_ind == 0:
                _, track_feat_support = self._get_track_feat(
                    fmaps_pyramid[i],
                    queried_coords / 2**i,
                    support_radius=self.corr_radius,
                )
                self.track_feat_cache[i] = torch.zeros_like(track_feat_support)
                self.track_feat_cache[i] += (
                    track_feat_support * is_track_initialized_now
                )
            else:
                if is_track_initialized_now.any():
                    _, track_feat_support = self._get_track_feat(
                        fmaps_pyramid[i],
                        queried_coords / 2**i,
                        support_radius=self.corr_radius,
                    )
                    self.track_feat_cache[i] += (
                        track_feat_support * is_track_initialized_now
                    )

            track_feat_support_pyramid.append(self.track_feat_cache[i].unsqueeze(1))

        # Initialize vis and conf for the current frame with zeros
        vis_init = torch.zeros((B, T, N, 1), device=self.device)
        conf_init = torch.zeros((B, T, N, 1), device=self.device)
        coords_init = queried_coords.reshape(B, T, N, 2).float()

        vis_init = torch.where(
            is_track_initialized_now.expand_as(vis_init),
            self.inv_sigmoid_true_val,
            vis_init,
        )
        conf_init = torch.where(
            is_track_initialized_now.expand_as(conf_init),
            self.inv_sigmoid_true_val,
            conf_init,
        )

        if self.online_ind == 0:
            self.ema_flow_buffer = torch.zeros_like(coords_init)

        # Handle tracks that are initialized in the previous frame.
        is_track_previsouly_initialized = (queried_frames < self.online_ind)[
            :, None, :, None
        ]  # B 1 N 1
        if self.online_ind > 0:
            vis_init = torch.where(
                is_track_previsouly_initialized.expand_as(vis_init),
                self.vis_buffer[:, -1],
                vis_init,
            )
            conf_init = torch.where(
                is_track_previsouly_initialized.expand_as(conf_init),
                self.conf_buffer[:, -1],
                conf_init,
            )
            # If there is only one frame processed so far, we initialize the coordinates
            # with the previous frame's coordinates
            if self.online_ind == 1:
                coords_init = torch.where(
                    is_track_previsouly_initialized.expand_as(coords_init),
                    self.coords_buffer[:, -1],
                    coords_init,
                )
            # If there is more, we use the exponential moving average of the flow
            else:
                last_flow = self.coords_buffer[:, -1] - self.coords_buffer[:, -2]
                cached_flow = self.ema_flow_buffer
                alpha = 0.8
                accumulated_flow = alpha * last_flow + (1 - alpha) * cached_flow
                self.ema_flow_buffer = accumulated_flow
                coords_init = torch.where(
                    is_track_previsouly_initialized.expand_as(coords_init),
                    self.coords_buffer[:, -1] + accumulated_flow,
                    coords_init,
                )

        # forward_window logic
        coords, viss, confs = self._forward_window(
            fmaps_pyramid=fmaps_pyramid,
            coords=coords_init,
            track_feat_support_pyramid=track_feat_support_pyramid,
            queried_frames=queried_frames,
            vis=vis_init,
            conf=conf_init,
            iters=self.iters,
            is_track_previsouly_initialized=is_track_previsouly_initialized,
        )

        coords[:, :, :, 0] *= (original_shape[3] - 1) / (
            self.model_resolution[1] - 1
        )  # W
        coords[:, :, :, 1] *= (original_shape[2] - 1) / (
            self.model_resolution[0] - 1
        )  # H

        viss = torch.sigmoid(viss)
        confs = torch.sigmoid(confs)

        viss = viss * confs
        thr = 0.6
        viss = viss > thr

        self.online_ind += 1
        return coords, viss, confs

    def _forward_window(
        self,
        fmaps_pyramid,
        coords,
        track_feat_support_pyramid,
        queried_frames,
        vis,
        conf,
        is_track_previsouly_initialized,
        iters=4,
    ):
        """
        Run the tracking update for a window of frames using ONNX sub-models.

        Replicates LiteTracker.forward_window() exactly, substituting PyTorch
        model calls with ONNX Runtime inference for corr_mlp and updateformer.

        Args:
            fmaps_pyramid (list[torch.Tensor]): Feature maps at different scales [B, T, C, H, W].
            coords (torch.Tensor): Track coordinates for each frame [B, T, N, 2].
            track_feat_support_pyramid (list[torch.Tensor]): Template features at different
                scales [B, 1, r^2, N, C].
            queried_frames (torch.Tensor): Frame indices of the queries [B, N].
            vis (torch.Tensor): Visibility logits for tracks [B, T, N, 1].
            conf (torch.Tensor): Confidence logits for tracks [B, T, N, 1].
            is_track_previsouly_initialized (torch.Tensor): Mask for tracks initialized in
                previous frames [B, 1, N, 1].
            iters (int): Number of update iterations.

        Returns:
            tuple: (coords, vis, conf) for the current window.
        """
        device = fmaps_pyramid[0].device
        B = fmaps_pyramid[0].shape[0]
        N = coords.shape[2]
        r = 2 * self.corr_radius + 1

        num_new_frames = 1
        num_prev_frames = min(self.online_ind, (self.window_len - 1))
        current_window_size = (
            num_prev_frames + num_new_frames
        )  # total number of frames in the current window

        # Compute the frame indices for the current window
        left_ind = max(0, self.online_ind - self.window_len + 1)  # inclusive
        right_ind = self.online_ind + 1  # not inclusive
        frame_indices = (
            torch.arange(left_ind, right_ind, device=device)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(B, -1, N)
        )  # shape [B, T, N]

        # `attention_mask` is a boolean mask: True if the track is initialized at or before
        # this frame
        attention_mask = (
            queried_frames.unsqueeze(1).expand(B, -1, N) <= frame_indices
        )  # B T N

        corr_embs = torch.empty(1, device=device)
        for it in range(iters):
            coords = coords.detach()  # B T N 2
            coords_init = coords.view(-1, N, 2)
            # Extract correlation embeddings from the new frames
            corr_embs_list = []
            for i in range(self.corr_levels):
                corr_feat = self._get_correlation_feat(
                    fmaps_pyramid[i], coords_init / 2**i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                # Run corr_mlp via ONNX
                corr_flat = corr_volume.reshape(
                    B * num_new_frames * N, r * r * r * r
                ).float()
                corr_embs_list.append(self._run_corr_mlp(corr_flat))
            corr_embs = torch.cat(corr_embs_list, dim=-1).view(
                B, num_new_frames, N, -1
            )

            # If it is the first time step, we skip the transformer as there is nothing
            # to compute, simply use the computed `corr_embs` as well as the initial
            # values of `coords`, `vis` and `conf` to initialize the buffers.
            if self.online_ind == 0:
                break

            prev_coords = self.coords_buffer.detach()
            prev_vis = self.vis_buffer.detach()
            prev_conf = self.conf_buffer.detach()
            prev_corr_embs = self.corr_embs_buffer.detach()

            current_window_coords = torch.cat([prev_coords, coords], dim=1)
            current_window_vis = torch.cat([prev_vis, vis], dim=1)
            current_window_conf = torch.cat([prev_conf, conf], dim=1)
            current_window_corr_embs = torch.cat([prev_corr_embs, corr_embs], dim=1)

            transformer_input = [
                current_window_vis,
                current_window_conf,
                current_window_corr_embs,
            ]
            rel_coords_forward = (
                current_window_coords[:, :-1] - current_window_coords[:, 1:]
            )
            rel_coords_backward = (
                current_window_coords[:, 1:] - current_window_coords[:, :-1]
            )
            rel_coords_forward = F.pad(rel_coords_forward, (0, 0, 0, 0, 0, 1))
            rel_coords_backward = F.pad(rel_coords_backward, (0, 0, 0, 0, 1, 0))

            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]], device=device
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale
            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )
            transformer_input.append(rel_pos_emb_input)
            x = (
                (torch.cat(transformer_input, dim=-1))
                .permute(0, 2, 1, 3)
                .reshape(B * N, current_window_size, -1)
            )
            x = x + self._interpolate_time_embed(current_window_size)
            x = x.view(B, N, current_window_size, -1)

            # Run updateformer via ONNX
            delta = self._run_updateformer(x.float(), attention_mask)
            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            delta_vis = delta[..., 2:3].permute(0, 2, 1, 3)
            delta_conf = delta[..., 3:].permute(0, 2, 1, 3)
            # Feature-pixel cap. CUDA correlation rounding on StereoMIS P2_1
            # point 2 walks off the image; this keeps the step inside the
            # pull-back range. None leaves the update uncapped.
            if self.max_step is not None:
                delta_coords = delta_coords.clamp(-self.max_step, self.max_step)

            # Update the values of the current frame only for the points that are
            # initialized before this frame.
            vis[is_track_previsouly_initialized] = (
                vis[is_track_previsouly_initialized]
                + delta_vis[:, -num_new_frames:][is_track_previsouly_initialized]
            )
            conf[is_track_previsouly_initialized] = (
                conf[is_track_previsouly_initialized]
                + delta_conf[:, -num_new_frames:][is_track_previsouly_initialized]
            )
            coords[is_track_previsouly_initialized.expand_as(coords)] = (
                coords[is_track_previsouly_initialized.expand_as(coords)]
                + delta_coords[:, -num_new_frames:][
                    is_track_previsouly_initialized.expand_as(coords)
                ]
            )

        # Update buffers
        if self.online_ind == 0:
            self.coords_buffer = coords
            self.vis_buffer = vis
            self.conf_buffer = conf
            self.corr_embs_buffer = corr_embs
        else:
            self.coords_buffer = torch.cat([self.coords_buffer, coords], dim=1)
            self.vis_buffer = torch.cat([self.vis_buffer, vis], dim=1)
            self.conf_buffer = torch.cat([self.conf_buffer, conf], dim=1)
            self.corr_embs_buffer = torch.cat(
                [self.corr_embs_buffer, corr_embs], dim=1
            )

        if current_window_size == self.window_len:
            self.coords_buffer = self.coords_buffer[:, 1:]
            self.vis_buffer = self.vis_buffer[:, 1:]
            self.conf_buffer = self.conf_buffer[:, 1:]
            self.corr_embs_buffer = self.corr_embs_buffer[:, 1:]

        coords = coords[..., :2] * float(self.stride)
        vis = vis[..., 0]
        conf = conf[..., 0]
        return coords, vis, conf
