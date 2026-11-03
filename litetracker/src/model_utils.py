from typing import Optional, Tuple, Union
import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F


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


def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    """
    Generate a grid of points covering a rectangular region.

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    ::math:: 
    P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): Grid size (number of points along each dimension).
        extent (Tuple[float, ...]): Height and width of the grid extent (H, W).
        center (Optional[Tuple[float, ...]], optional): Center of the grid (c_y, c_x). Defaults to the center of the extent.
        device (Optional[torch.device], optional): Device for the output tensor. Defaults to CPU.

    Returns:
        torch.Tensor: Grid of points of shape (1, size*size, 2), in row-major order.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


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
