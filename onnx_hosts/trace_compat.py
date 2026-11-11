"""ONNX tracing helpers used by the hosts.

The legacy tracer cannot lower: Python round() on tensor shapes, adaptive
interpolate with a tensor size, 5D GridSample, or InstanceNorm with unknown C.
These helpers freeze dummy H/W/T to Python ints and rewrite 5D sampling as a
weighted sum of 4D grid_sample calls.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def as_int(value) -> int:
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    return int(value)


def as_int_pair(values: Sequence) -> Tuple[int, int]:
    seq = list(values)
    return as_int(seq[0]), as_int(seq[1])


def grid_sample_2d(
    image: torch.Tensor,
    coords_xy: torch.Tensor,
    align_corners: bool = True,
    padding_mode: str = "border",
) -> torch.Tensor:
    """4D grid_sample with pixel coords (x, y) on the last dim."""
    height = as_int(image.shape[-2])
    width = as_int(image.shape[-1])
    if align_corners:
        scale = coords_xy.new_tensor([2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)])
    else:
        scale = coords_xy.new_tensor([2.0 / max(width, 1), 2.0 / max(height, 1)])
    grid = coords_xy * scale - 1.0
    return F.grid_sample(
        image, grid, mode="bilinear", align_corners=align_corners, padding_mode=padding_mode
    )


def bilinear_sampler_onnx(input: torch.Tensor, coords: torch.Tensor, align_corners=True, padding_mode="border"):
    """Drop-in for CoTracker bilinear_sampler that never emits 5D GridSample."""
    if input.ndim == 4:
        return grid_sample_2d(input, coords, align_corners=align_corners, padding_mode=padding_mode)

    if input.ndim != 5:
        raise ValueError(f"bilinear_sampler expected 4D or 5D input, got {tuple(input.shape)}")

    # input: (B, C, T, H, W); coords last dim is (t, x, y) in pixel units
    batch, channels, time, height, width = (
        as_int(input.shape[0]),
        as_int(input.shape[1]),
        as_int(input.shape[2]),
        as_int(input.shape[3]),
        as_int(input.shape[4]),
    )
    t_coord = coords[..., 0]
    xy = torch.stack([coords[..., 1], coords[..., 2]], dim=-1)
    xy_grid = xy.reshape(batch, -1, 1, 2)
    frames = []
    for index in range(time):
        sampled = grid_sample_2d(
            input[:, :, index],
            xy_grid,
            align_corners=align_corners,
            padding_mode=padding_mode,
        )
        frames.append(sampled)
    stacked = torch.stack(frames, dim=-1)
    t_flat = t_coord.reshape(batch, 1, -1, 1, 1).to(dtype=input.dtype)
    frame_idx = torch.arange(time, device=input.device, dtype=input.dtype).view(1, 1, 1, 1, time)
    weights = (1.0 - (t_flat - frame_idx).abs()).clamp(min=0.0)
    sampled = (stacked * weights).sum(dim=-1)
    return sampled.reshape(batch, channels, *coords.shape[1:-1])


def map_coordinates_3d_onnx(feats: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """LocoTrack map_coordinates_3d using 4D grid_sample + linear time lerp.

    feats: (B, T, H, W, C). coordinates: (B, N, 3) in (t, y, x), matching tyx queries.
    Each frame is sampled once (output is T by N by C). Floor and ceil are then
    gathered along that time axis. The previous gather expanded an index to
    (N, C, H, W) and materialized a full feature map per point (~30 GB).
    Time stays a data dimension: do not loop ``range(T)`` or read T with ``as_int``.
    """
    height = as_int(feats.shape[2])
    width = as_int(feats.shape[3])
    batch = as_int(feats.shape[0])
    channels = as_int(feats.shape[4])
    n_points = as_int(coordinates.shape[1])
    frames = feats.permute(0, 1, 4, 2, 3).contiguous().flatten(0, 1)

    coord_t = coordinates[..., 0].to(dtype=feats.dtype)
    coord_y = coordinates[..., 1]
    coord_x = coordinates[..., 2]
    xy = torch.stack(
        [
            2.0 * coord_x / max(width, 1) - 1.0,
            2.0 * coord_y / max(height, 1) - 1.0,
        ],
        dim=-1,
    )
    # ones_like keeps T symbolic. A Python expand/arange product would bake the trace length.
    time_ones = torch.ones_like(feats[:, :, :1, :1, :1])
    xy_bt = xy.view(batch, 1, n_points, 2) * time_ones.reshape(batch, -1, 1, 1)
    grid = xy_bt.flatten(0, 1).unsqueeze(2)
    sampled = F.grid_sample(
        frames,
        grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    sampled = sampled.reshape(-1, channels, n_points)
    sampled = sampled.permute(0, 2, 1).reshape(batch, -1, n_points, channels)

    t0 = coord_t.floor()
    t1 = t0 + 1.0
    w1 = (coord_t - t0).clamp(min=0.0, max=1.0)
    w0 = 1.0 - w1
    t_last = torch.arange(feats.shape[1], device=feats.device)[-1]
    t0i = torch.minimum(t0.long().clamp(min=0), t_last)
    t1i = torch.minimum(t1.long().clamp(min=0), t_last)

    def _gather_time(t_idx: torch.Tensor) -> torch.Tensor:
        index = t_idx.reshape(batch, n_points, 1).expand(-1, -1, channels).unsqueeze(1)
        return torch.gather(sampled, 1, index).squeeze(1)

    feat0 = _gather_time(t0i)
    feat1 = _gather_time(t1i)
    return w0.unsqueeze(-1) * feat0 + w1.unsqueeze(-1) * feat1


def generate_resolutions_onnx(full_size, train_size, num_levels=None):
    import numpy as np

    full_size = (as_int(full_size[0]), as_int(full_size[1]))
    train_size = (as_int(train_size[0]), as_int(train_size[1]))
    if full_size == train_size:
        return [train_size]
    if num_levels is None:
        size_ratio = np.array(full_size, dtype=np.float64) / np.array(train_size, dtype=np.float64)
        num_levels = int(np.ceil(np.max(np.log2(np.maximum(size_ratio, 1e-6)))) + 1)
    if num_levels <= 1:
        return [train_size]
    height, width = full_size
    ll_h, ll_w = train_size
    sizes = []
    for i in range(num_levels):
        sizes.append(
            (
                int(round((ll_h * (height / ll_h) ** (i / (num_levels - 1))) // 8)) * 8,
                int(round((ll_w * (width / ll_w) ** (i / (num_levels - 1))) // 8)) * 8,
            )
        )
    return sizes


def _swap_instance_norm(module: nn.Module, device: torch.device, dtype: torch.dtype) -> None:
    """Replace InstanceNorm* with per-channel GroupNorm (static C for ORT / TRT)."""
    instance_types = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    for name, child in list(module.named_children()):
        if isinstance(child, instance_types):
            group_norm = nn.GroupNorm(
                num_groups=child.num_features,
                num_channels=child.num_features,
                affine=bool(child.affine),
                eps=child.eps,
            ).to(device=device, dtype=dtype)
            if child.affine and child.weight is not None:
                group_norm.weight.data.copy_(child.weight.data)
                group_norm.bias.data.copy_(child.bias.data)
            setattr(module, name, group_norm)
        else:
            _swap_instance_norm(child, device, dtype)


def patch_pips_for_onnx(model: nn.Module) -> None:
    import nets.pips2 as pips2

    def conv1d_forward(self, x):
        pad = (int(self.kernel_size) - 1) // 2
        return self.conv(F.pad(x, (pad, pad)))

    def residual_forward(self, x):
        identity = x
        out = x
        if not self.is_first_block:
            if self.use_norm:
                out = self.norm1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        if self.use_norm:
            out = self.norm2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        if self.out_channels != self.in_channels:
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (0, 0, ch1, ch2))
        return out + identity

    def encoder_forward(self, x):
        height = as_int(x.shape[2])
        width = as_int(x.shape[3])
        target = (height // self.stride, width // self.stride)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)
        a = F.interpolate(a, target, mode="bilinear", align_corners=True)
        b = F.interpolate(b, target, mode="bilinear", align_corners=True)
        c = F.interpolate(c, target, mode="bilinear", align_corners=True)
        d = F.interpolate(d, target, mode="bilinear", align_corners=True)
        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return x

    pips2.Conv1dPad.forward = conv1d_forward
    pips2.ResidualBlock1d.forward = residual_forward
    pips2.BasicEncoder.forward = encoder_forward
    param = next(model.parameters())
    _swap_instance_norm(model, param.device, param.dtype)


def patch_alltracker_for_onnx(model: nn.Module) -> None:
    import nets.alltracker as at
    import nets.blocks as blocks

    _padder_init = blocks.InputPadder.__init__

    def padder_init(self, dims, mode="sintel"):
        height, width = as_int(dims[-2]), as_int(dims[-1])
        _padder_init(self, (None, None, height, width), mode=mode)

    def fetch_time_embed(self, t, dtype, is_training=False):
        t = as_int(t)
        seqlen = as_int(self.time_emb.shape[1])
        if t == seqlen:
            return self.time_emb.to(dtype)
        if t == 1:
            return self.time_emb[:, 1:2].to(dtype)
        time_emb = self.time_emb.float()
        time_emb = F.interpolate(time_emb.permute(0, 2, 1), size=t, mode="linear").permute(0, 2, 1)
        return time_emb.to(dtype)

    def coords_grid(self, batch, ht, wd, device, dtype):
        batch, ht, wd = as_int(batch), as_int(ht), as_int(wd)
        coords = torch.meshgrid(
            torch.arange(ht, device=device, dtype=dtype),
            torch.arange(wd, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack(coords[::-1], dim=0)
        return coords[None].repeat(batch, 1, 1, 1)

    _orig_forward = at.Net.forward

    def forward(self, images, iters=4, sw=None, is_training=False, stride=None):
        batch, time, channels, height, width = (as_int(x) for x in images.shape)
        images = images.reshape(batch, time, channels, height, width)
        return _orig_forward(self, images, iters=iters, sw=sw, is_training=is_training, stride=stride)

    _orig_get_fmaps = at.Net.get_fmaps

    def get_fmaps(self, images_, B, T, sw, is_training):
        # Skip the eval-time .cuda() / chunk path; the tensors are already on device.
        return _orig_get_fmaps(self, images_, B, T, sw, True)

    def corr(fmap1, fmap2, num_head):
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]
        dim_i = as_int(dim)
        fmap1 = fmap1.view(batch, num_head, dim_i // num_head, h1 * w1)
        fmap2 = fmap2.view(batch, num_head, dim_i // num_head, h2 * w2)
        corr = fmap1.transpose(2, 3) @ fmap2
        corr = corr.reshape(batch, num_head, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)
        scale = torch.sqrt(fmap1.new_tensor(float(dim_i)))
        return corr / scale

    def corr_init(self, fmap1, fmap2, corr_levels, corr_radius):
        self.num_levels = corr_levels
        self.radius = corr_radius
        self.corr_pyramid = []
        for _ in range(self.num_levels):
            corr = blocks.CorrBlock.corr(fmap1, fmap2, 1)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
            fh, fw = as_int(fmap2.shape[-2]), as_int(fmap2.shape[-1])
            fmap2 = F.interpolate(
                fmap2,
                size=(max(fh // 2, 1), max(fw // 2, 1)),
                mode="bilinear",
                align_corners=False,
            )
            self.corr_pyramid.append(corr)

    blocks.CorrBlock.corr = staticmethod(corr)
    blocks.CorrBlock.__init__ = corr_init
    blocks.InputPadder.__init__ = padder_init
    at.Net.fetch_time_embed = fetch_time_embed
    at.Net.coords_grid = coords_grid
    at.Net.forward = forward
    at.Net.get_fmaps = get_fmaps


def patch_locotrack_for_onnx() -> None:
    import models.locotrack_model as loco_model
    import models.utils as loco_utils

    loco_utils.map_coordinates_3d = map_coordinates_3d_onnx
    loco_utils.generate_default_resolutions = generate_resolutions_onnx

    def bilinear_onnx(x: torch.Tensor, resolution):
        channels = as_int(x.shape[-1])
        x = x.permute(0, 1, 4, 2, 3).flatten(1, 2)
        x = F.interpolate(x, size=tuple(resolution), mode="bilinear", align_corners=False)
        x = x.reshape(x.shape[0], -1, channels, x.shape[2], x.shape[3])
        return x.permute(0, 1, 3, 4, 2)

    loco_utils.bilinear = bilinear_onnx

    _heat = loco_utils.heatmaps_to_points

    def heatmaps_to_points_onnx(all_pairs_softmax, image_shape, threshold=5, query_points=None):
        out_points = loco_utils.soft_argmax_heatmap_batched(all_pairs_softmax, threshold)
        feature_grid_shape = all_pairs_softmax.shape[1:]
        out_points = loco_utils.convert_grid_coordinates(
            out_points,
            feature_grid_shape[3:1:-1],
            image_shape[3:1:-1],
        )
        if query_points is None:
            return out_points
        query_frame = loco_utils.convert_grid_coordinates(
            query_points.detach(),
            image_shape[1:4],
            feature_grid_shape[1:4],
            coordinate_format="tyx",
        )[..., 0:1]
        query_frame = torch.round(query_frame)
        n_frames = all_pairs_softmax.shape[2]
        frame_indices = torch.arange(n_frames, device=query_frame.device)[None, None, :]
        is_query_point = (query_frame == frame_indices)[:, :, :, None]
        return (
            out_points * ~is_query_point
            + torch.flip(query_points[:, :, None], dims=(-1,))[..., 0:2] * is_query_point
        )

    loco_utils.heatmaps_to_points = heatmaps_to_points_onnx
    del _heat

    _orig_same = loco_utils.Conv2dSamePadding.forward

    def same_pad_forward(self, x):
        ih, iw = as_int(x.size(-2)), as_int(x.size(-1))
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)

    loco_utils.Conv2dSamePadding.forward = same_pad_forward
    del _orig_same

    _orig_grids = loco_model.LocoTrack.get_feature_grids

    def get_feature_grids_onnx(self, video, is_training=False, refinement_resolutions=None):
        del is_training
        if refinement_resolutions is None:
            refinement_resolutions = loco_utils.generate_default_resolutions(
                video.shape[2:4], self.initial_resolution
            )
        feature_grid = []
        hires_feats = []
        highest_feats = []
        resize_im_shape = []
        curr_resolution = (-1, -1)
        latent = hires = highest = None
        video_resize = None
        batch = video.shape[0]
        for resolution in refinement_resolutions:
            if resolution[0] % 8 != 0 or resolution[1] % 8 != 0:
                raise ValueError("Image resolution must be a multiple of 8.")
            if not loco_utils.is_same_res(curr_resolution, resolution):
                if loco_utils.is_same_res(curr_resolution, video.shape[-3:-1]):
                    video_resize = video
                else:
                    video_resize = loco_utils.bilinear(video, resolution)
                curr_resolution = resolution
                nchw = video_resize.flatten(0, 1).permute(0, 3, 1, 2)
                resnet_out = self.resnet_torch(nchw)
                latent = resnet_out["resnet_unit_3"].permute(0, 2, 3, 1)
                hires = resnet_out["resnet_unit_1"].permute(0, 2, 3, 1)
                highest = resnet_out["resnet_unit_0"].permute(0, 2, 3, 1)
                if self.extra_convs:
                    latent = self.extra_convs(latent)

                def _l2(feats):
                    return feats / torch.sqrt(
                        torch.maximum(
                            torch.sum(torch.square(feats), axis=-1, keepdims=True),
                            torch.tensor(1e-12, device=feats.device),
                        )
                    )

                latent = _l2(latent).reshape(batch, -1, *latent.shape[1:])
                hires = _l2(hires).reshape(batch, -1, *hires.shape[1:])
                highest = _l2(highest).reshape(batch, -1, *highest.shape[1:])
            feature_grid.append(latent)
            hires_feats.append(hires)
            highest_feats.append(highest)
            resize_im_shape.append((as_int(video_resize.shape[2]), as_int(video_resize.shape[3])))
        return loco_model.FeatureGrids(
            tuple(feature_grid), tuple(hires_feats), tuple(highest_feats), tuple(resize_im_shape)
        )

    loco_model.LocoTrack.get_feature_grids = get_feature_grids_onnx
    del _orig_grids

    def relative_positions(seq_len, reverse=False, device="cuda"):
        idx = torch.arange(seq_len, device=device)
        x = idx[None, :]
        y = idx[:, None]
        return torch.tril(x - y) if not reverse else torch.triu(y - x)

    def attention_forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        query_heads = self._linear_projection(query, self.key_size, self.query_proj)
        key_heads = self._linear_projection(key, self.key_size, self.key_proj)
        value_heads = self._linear_projection(value, self.value_size, self.value_proj)
        seq_len = query.shape[1]
        device = query.device
        bias_forward = loco_model.get_alibi_slope(self.num_heads // 2, device=device) * relative_positions(
            seq_len, device=device
        )
        bias_forward = bias_forward + torch.triu(torch.full_like(bias_forward, -1e9), diagonal=1)
        bias_backward = loco_model.get_alibi_slope(self.num_heads // 2, device=device) * relative_positions(
            seq_len, reverse=True, device=device
        )
        bias_backward = bias_backward + torch.tril(torch.full_like(bias_backward, -1e9), diagonal=-1)
        attn_bias = torch.cat([bias_forward, bias_backward], dim=0)
        scale = query_heads.new_tensor(self.key_size ** -0.5)
        logits = torch.matmul(query_heads, key_heads.transpose(-2, -1)) * scale
        logits = logits + attn_bias.to(dtype=logits.dtype)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.matmul(attn, value_heads)
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, -1, self.num_heads * self.value_size)
        return self.final_proj(attn)

    def linear_projection(self, x, head_size, proj_layer):
        y = proj_layer(x)
        return y.reshape(x.shape[0], -1, self.num_heads, head_size).permute(0, 2, 1, 3)

    loco_model.MultiHeadAttention.forward = attention_forward
    loco_model.MultiHeadAttention._linear_projection = linear_projection


def patch_cotracker_for_onnx() -> None:
    import cotracker.cotracker.models.core.model_utils as model_utils

    model_utils.bilinear_sampler = bilinear_sampler_onnx
    try:
        import cotracker.cotracker.models.core.cotracker.blocks as blocks

        blocks.bilinear_sampler = bilinear_sampler_onnx
    except ImportError:
        pass
    try:
        import cotracker.cotracker.models.core.cotracker.cotracker3_online as online

        online.bilinear_sampler = bilinear_sampler_onnx
        _orig_time = online.CoTrackerThreeOnline.interpolate_time_embed

        def interpolate_time_embed(self, x, t):
            t = as_int(t)
            return _orig_time(self, x, t)

        online.CoTrackerThreeOnline.interpolate_time_embed = interpolate_time_embed
    except ImportError:
        pass
    try:
        import cotracker.cotracker.models.core.cotracker.cotracker as ct

        if hasattr(ct, "bilinear_sampler"):
            ct.bilinear_sampler = bilinear_sampler_onnx
    except ImportError:
        pass
