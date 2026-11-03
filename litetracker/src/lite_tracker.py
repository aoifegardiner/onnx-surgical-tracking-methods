from __future__ import annotations
"""
lite_tracker.py

Implements the LiteTracker model for efficient multi-object tracking in videos.

This module provides:
    - The LiteTracker neural network class for online tracking
    - Positional encoding utilities
    - Support point and feature extraction for tracking
    - Correlation feature computation and transformer-based updates
    - Methods for initializing, resetting, and running the tracker on video frames

Dependencies: torch, torch.nn, torch.nn.functional, src.model_utils, src.model_blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model_utils import (
    sample_features5d,
    bilinear_sampler,
    get_1d_sincos_pos_embed_from_grid,
)

from src.model_blocks import Mlp, BasicEncoder, EfficientUpdateFormer


torch.manual_seed(0)


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



class LiteTracker(nn.Module):
    def __init__(
        self,
        window_len=16,
        stride=4,
        corr_radius=3,
        corr_levels=4,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        linear_layer_for_vis_conf=True,
        iters=1,
    ):
        """
        Initialize the LiteTracker model.

        Args:
            window_len (int): Max length of the temporal window for tracking. Note that the model can process less than `window_len` frames if there are not enough frames available.
            stride (int): Stride scale for feature extraction.
            corr_radius (int): Radius for correlation computation.
            corr_levels (int): Number of correlation pyramid levels.
            num_virtual_tracks (int): Number of virtual tracks for the transformer.
            model_resolution (tuple): Resolution to which input frames are resized; (width, height).
            linear_layer_for_vis_conf (bool): Whether to use a linear layer for visibility/confidence.
            iters (int): Number of update iterations per frame.
        """
        super().__init__()
        self.window_len = window_len
        self.stride = stride
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.hidden_dim = 256
        self.latent_dim = 128
        self.inv_sigmoid_true_val = 4.6  # empirically chosen value close to 0.9
        self.iters = iters

        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.fnet = BasicEncoder(input_dim=3, output_dim=self.latent_dim, stride=stride)

        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution

        self.input_dim = 1110

        self.updateformer = EfficientUpdateFormer(
            space_depth=3,
            time_depth=3,
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=4,
            mlp_ratio=4.0,
            num_virtual_tracks=num_virtual_tracks,
            linear_layer_for_vis_conf=linear_layer_for_vis_conf,
        )
        self.corr_mlp = Mlp(in_features=49 * 49, hidden_features=384, out_features=256)

        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(
            1, window_len, 1
        )

        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0])
        )
        self.reset()


    def get_support_points(self, coords, r: int, reshape_back: bool = True):
        """
        Generate a grid of support points around each coordinate for local feature sampling.

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

        xgrid, ygrid = torch.meshgrid(dy, dx)#, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack((zgrid, xgrid, ygrid), dim=-1)
        delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl

        if reshape_back:
            third_dim = int((2 * r + 1) ** 2)
            return coords_lvl.reshape(B, N, third_dim, 3).permute(0, 2, 1, 3)
        else:
            return coords_lvl

    def get_track_feat(
        self, fmaps: torch.Tensor, queried_coords: torch.Tensor, support_radius: int = 0
    ):
        """
        Extract track features at the queried coordinates and their support points.

        Args:
            fmaps (torch.Tensor): Feature maps.
            queried_coords (torch.Tensor): Coordinates to sample features from.
            support_radius (int): Radius for support points.

        Returns:
            tuple: (central track features, all support track features)
        """
        sample_coords = torch.cat(
            [
                torch.zeros_like(queried_coords[..., :1][:, None]),
                queried_coords[:, None],
            ],
            dim=-1,
        )
        support_points = self.get_support_points(sample_coords, support_radius)
        support_track_feats = sample_features5d(fmaps, support_points)
        return (
            support_track_feats[:, None, support_track_feats.shape[1] // 2],
            support_track_feats,
        )

    def get_correlation_feat(self, fmaps, queried_coords):
        """
        Compute correlation features for the queried coordinates using bilinear sampling.

        Args:
            fmaps (torch.Tensor): Feature maps of shape [B, T, D, H, W].
            queried_coords (torch.Tensor): Coordinates to sample features from.

        Returns:
            torch.Tensor: Correlation features for each queried coordinate.
        """
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = self.get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = bilinear_sampler(
            fmaps.reshape(B * T, D, 1, H_, W_), support_points
        )
        return correlation_feat.view(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )

    def interpolate_time_embed(self, x: torch.Tensor, t: int):
        """
        Interpolate the temporal positional embedding to match the current window size.

        Args:
            x (torch.Tensor): Input tensor (for dtype reference).
            t (int): Target temporal length.

        Returns:
            torch.Tensor: Interpolated time embedding.
        """
        previous_dtype = x.dtype
        T = self.time_emb.shape[1]

        if t == T:
            return self.time_emb

        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb.to(previous_dtype)

    def init_video_online_processing(self):
        """
        Reset all internal buffers and caches for a new video sequence. Call this method before processing a new video.
        """
        self.reset()

    def forward_window(
        self,
        fmaps_pyramid: list[torch.Tensor],
        coords: torch.Tensor,
        track_feat_support_pyramid: list[torch.Tensor],
        queried_frames: torch.Tensor,
        vis: torch.Tensor,
        conf: torch.Tensor,
        is_track_previsouly_initialized: torch.Tensor,
        iters: int = 4,
    ):
        """
        Run the tracking update for a window of frames.

        Args:
            fmaps_pyramid (list[torch.Tensor]): List of feature maps at different scales [B, T, C, H, W].
            coords (torch.Tensor): Track coordinates for each frame [B, T, N, 2].
            track_feat_support_pyramid (list[torch.Tensor]): List of template features at different scales [B, 1, r^2, N, C].
            queried_frames (torch.Tensor): Frame indices of the queries [B, N].
            vis (torch.Tensor): Visibility logits for tracks [B, T, N, 1].
            conf (torch.Tensor): Confidence logits for tracks [B, T, N, 1].
            is_track_previsouly_initialized (torch.Tensor): Mask for tracks initialized in previous frames [B, T, N, 1].
            iters (int): Number of update iterations.

        Returns:
            tuple: (coords, vis, conf) for the current window.
        """

        device = fmaps_pyramid[0].device
        dtype = fmaps_pyramid[0].dtype
        B = fmaps_pyramid[0].shape[0]
        N = coords.shape[2]
        r = 2 * self.corr_radius + 1

        num_new_frames = 1
        num_prev_frames = min(self.online_ind, (self.window_len - 1))
        current_window_size = (
            num_prev_frames + num_new_frames
        )  # total number of frames in the current window; can be less than or equal to the window length

        # Compute the frame indices for the current window
        # i.e.:
        # - self.online_ind := 0; left_ind := 0; right_ind := 1
        # - self.online_ind := 1; left_ind := 0; right_ind := 2
        # ...
        # - self.online_ind := 15; left_ind := 0; right_ind := 16
        # - self.online_ind := 16; left_ind := 1; right_ind := 17
        # ...
        left_ind = max(0, self.online_ind - self.window_len + 1)  # inclusive
        right_ind = self.online_ind + 1  # not inclusive
        frame_indices = (
            torch.arange(left_ind, right_ind, device=device)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(B, -1, N)
        )  # shape [B, T, N]

        # `frame_track_activation_mask` is a boolean mask for each tracklet, if they are initialized until including this frame the value is `True`
        attention_mask = (
            queried_frames.unsqueeze(1).expand(B, -1, N) <= frame_indices
        )  # B T N

        # Initialize corr_embs with zeros
        corr_embs = torch.empty(1, device=device, dtype=dtype)
        for it in range(iters):
            coords = coords.detach()  # B T N 2
            coords_init = coords.view(-1, N, 2)
            # Extract correlation embeddings from the new frames
            corr_embs_list = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(
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
                corr_embs_list.append(
                    self.corr_mlp(
                        corr_volume.reshape(B * num_new_frames * N, r * r * r * r)
                    )
                )
            corr_embs = torch.cat(corr_embs_list, dim=-1).view(B, num_new_frames, N, -1)

            # If it is the first time step, we skip the transfomer as there is nothing to compute, simply use the computed `corr_embs` as well as the initial values of `coords`, `vis` and `conf` to initialize the buffers.
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
            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )

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
            x = x + self.interpolate_time_embed(x, current_window_size)
            x = x.view(B, N, current_window_size, -1)
            delta = self.updateformer(
                x,
                mask=attention_mask,
            )
            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            delta_vis = delta[..., 2:3].permute(0, 2, 1, 3)
            delta_conf = delta[..., 3:].permute(0, 2, 1, 3)

            # Update the values of the current frame only for the points that are initialized before this frame.
            # `frame_track_initialization_mask` is a boolean mask for each tracklet, if they are initialized in this frame the value is `True`
            #  We use this to make sure we do not update the initial values for this frame and make the confidence 1 occlusion 0
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
            self.corr_embs_buffer = torch.cat([self.corr_embs_buffer, corr_embs], dim=1)

        if current_window_size == self.window_len:
            self.coords_buffer = self.coords_buffer[:, 1:]
            self.vis_buffer = self.vis_buffer[:, 1:]
            self.conf_buffer = self.conf_buffer[:, 1:]
            self.corr_embs_buffer = self.corr_embs_buffer[:, 1:]

        coords = coords[..., :2] * float(self.stride)
        vis = vis[..., 0]
        conf = conf[..., 0]
        return coords, vis, conf

    def reset(self):
        """
        Reset all internal state, buffers, and caches of the tracker.
        """
        self.online_ind = 0
        # Buffers have a max temporal size (`T`) of `self.window_len - 1` and operates in a FIFO manner
        self.ema_flow_buffer = torch.empty(0)  # shape [B, T, N, 2]
        self.corr_embs_buffer = torch.empty(0)  # shape [B, T, N, C]
        self.coords_buffer = torch.empty(
            0
        )  # in the space of 1 / self.stride; shape [B, T, N, 2]
        self.vis_buffer = torch.empty(0)  # in logit space; shape [B, T, N, 1]
        self.conf_buffer = torch.empty(0)  # in logit space; of shape [B, T, N, 1]

        # Caches are used to store the intermediate results of the model
        self.track_feat_cache = [
            torch.empty(0)
        ] * self.corr_levels  # Track features for matching

        print(f"All the caches are reset.")

    def forward(
        self,
        frame: torch.Tensor,
        queries: torch.Tensor,
    ):
        """
        Predict tracks for the given frame and queries.

        Args:
            frame (torch.Tensor): Input frames of shape [B, C, H, W].
            queries (torch.Tensor): Point queries of shape [B, N, 3]; first channel is the frame index, the rest are coordinates.

        Returns:
            tuple:
                coords_predicted (torch.Tensor): Predicted coordinates [B, T, N, 2].
                vis_predicted (torch.Tensor): Predicted visibility mask [B, T, N].
                conf_predicted (torch.Tensor): Predicted confidence [B, T, N].
        """
        original_shape = frame.shape
        frame = F.interpolate(
            frame, size=self.model_resolution, mode="bilinear", align_corners=True
        )  # B, C, H, W
        queries_scaled = queries.clone()

        queries_scaled[:, :, 1] *= (self.model_resolution[1] - 1) / (
            original_shape[3] - 1
        )  # W
        queries_scaled[:, :, 2] *= (self.model_resolution[0] - 1) / (
            original_shape[2] - 1
        )  # H

        B, C, H, W = frame.shape

        device = queries_scaled.device

        B, N, __ = queries_scaled.shape
        frame = 2 * (frame / 255.0) - 1.0

        T = 1

        dtype = frame.dtype

        queried_frames = queries_scaled[:, :, 0].long()

        # Downscale the query coords for the smaller size feat maps
        queried_coords = queries_scaled[..., 1:3].clone()
        queried_coords = queried_coords / self.stride

        fmaps = self.fnet(frame)
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
        fmaps = fmaps.to(dtype)

        fmaps_pyramid = []
        track_feat_support_pyramid = []
        fmaps_pyramid.append(fmaps)
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
                _, track_feat_support = self.get_track_feat(
                    fmaps_pyramid[i],
                    queried_coords / 2**i,
                    support_radius=self.corr_radius,
                )
                self.track_feat_cache[i] = torch.zeros_like(
                    track_feat_support, device=device
                )
                self.track_feat_cache[i] += (
                    track_feat_support * is_track_initialized_now
                )
            else:
                if is_track_initialized_now.any():
                    _, track_feat_support = self.get_track_feat(
                        fmaps_pyramid[i],
                        queried_coords / 2**i,
                        support_radius=self.corr_radius,
                    )
                    self.track_feat_cache[i] += (
                        track_feat_support * is_track_initialized_now
                    )

            track_feat_support_pyramid.append(self.track_feat_cache[i].unsqueeze(1))

        # Initialize vis and conf for the current frame with zeros
        vis_init = torch.zeros((B, T, N, 1), device=device).float()
        conf_init = torch.zeros((B, T, N, 1), device=device).float()
        coords_init = queried_coords.reshape(B, T, N, 2).float()

        vis_init = torch.where(
            is_track_initialized_now.expand_as(vis_init),
            torch.tensor(self.inv_sigmoid_true_val, device=vis_init.device, dtype=vis_init.dtype),
            vis_init.float(),
        )
        conf_init = torch.where(
            is_track_initialized_now.expand_as(conf_init),
            torch.tensor(self.inv_sigmoid_true_val, device=conf_init.device, dtype=conf_init.dtype),
            conf_init.float(),
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
            # If there is only one frame processed so far, we initialize the coordinates with the previous frame's coordinates
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

        coords, viss, confs = self.forward_window(
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
