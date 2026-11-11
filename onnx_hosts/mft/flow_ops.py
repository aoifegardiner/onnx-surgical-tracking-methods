"""Sampling helpers used by the MFT delta chain.

Adapted from MFT (Neoral, Serych, Matas, WACV 2024), CC BY-NC-SA 4.0.
See LICENSE in this directory.
"""
import einops
import numpy as np
import torch
import torch.nn.functional as F


def ensure_numpy(xs):
    if isinstance(xs, torch.Tensor):
        return xs.detach().cpu().numpy()
    else:
        return xs


def ensure_torch(xs):
    if not isinstance(xs, torch.Tensor):
        return torch.from_numpy(xs)
    else:
        return xs


def unravel_indices(indices, shape, stack_dim=-1, np_order=False):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).

    From https://github.com/pytorch/pytorch/issues/35674#issuecomment-739560051
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')

    if np_order:  # row, column (y, x)
        coord = torch.stack(coord[::-1], dim=stack_dim)
    else:  # column, row (x, y)
        coord = torch.stack(coord, dim=stack_dim)
    assert coord.device == indices.device

    return coord


def torch_get_featuremap_coords(feature_map, device=None,
                                keep_shape=False):
    """ get coordinate map corresponding to a feature map

    args:
        feature_map: (..., H, W) tensor
        keep_shape: boolean (default False). Setting it to True does not flatten the output coordinates

    returns:
        xy: (2, H*W) tensor with x, y coordinates.  (2, H, W) tensor if keep_shape == True
    """
    if type(feature_map) is tuple and len(feature_map) == 2:
        H, W = feature_map
        assert device is not None
    else:
        H, W = feature_map.shape[-2:]
        if device is None:
            device = feature_map.device
    xy = unravel_indices(torch.arange(H * W, device=device), (H, W), stack_dim=0)

    if keep_shape:
        xy = einops.rearrange(xy, 'xy (H W) -> xy H W',
                              H=H, W=W, xy=2)
    return xy


def normalize_coords(coordinates, H, W):
    """Normalize coordinates to be in [-1, 1] range as usedi in F.grid_sample.
    args:
        coordinates: (N H W xy) coordinates
    """
    device = coordinates.device
    scales = np.array([2 / (W - 1), 2 / (H - 1)]).astype(np.float32)  # maps to [0, 2]
    scales = torch.from_numpy(scales).to(device)
    scales = einops.rearrange(scales, '(N H W xy) -> N H W xy', N=1, H=1, W=1)
    coordinates_normed = coordinates * scales - 1      # maps to [-1, 1]
    return coordinates_normed


def bilinear_sample(data, coords, device=None):
    """
    args:
        data: (batch, C, H, W) tensor
        coords: (batch, ...outshape..., xy) tensor of coordinates
    returns:
        sampled: (batch, ...outshape..., C) tensor
    """
    assert len(coords.shape) >= 3
    assert coords.shape[-1] == 2
    if device is None:
        device = coords.device
    data_shape = einops.parse_shape(data, 'batch C H W')
    norm_coords = normalize_coords(coords.to(device), data_shape['H'], data_shape['W'])
    flat_coords = norm_coords.view(coords.shape[0], -1, coords.shape[-1])  # flatten all the outshape
    grid_coords = einops.rearrange(flat_coords, 'batch N xy -> batch 1 N xy', xy=2)  # simulate HxW coord grid (but only 1xN)
    sampled_flat = F.grid_sample(data.to(device), grid_coords, align_corners=True)
    sampled = sampled_flat.view(*coords.shape[:-1], data_shape['C'])
    return sampled
