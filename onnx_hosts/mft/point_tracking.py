"""Query sampling on an MFT flow field.

Adapted from MFT (Neoral, Serych, Matas, WACV 2024), CC BY-NC-SA 4.0.
See LICENSE in this directory.
"""
import numpy as np
import einops

from .flow_ops import bilinear_sample, ensure_numpy


def convert_to_point_tracking(MFT_result, queries):
    """Convert MFT results to point-tracking results.

    args:
      MFT_result: FlowOUTrackingResult
      queries: (N xy) tensor with query coordinates on the init frame

    returns:
      current_coords: numpy array with coordinates in the current frame, shape (N, xy)
      current_occlusions: numpy array with occlusions in the current frame, shape (N, )
    """
    current_coords = MFT_result.warp_forward_points(queries)
    current_occlusions = einops.rearrange(
        bilinear_sample(
            einops.rearrange(MFT_result.occlusion, 'C H W -> 1 C H W', C=1),
            einops.rearrange(queries, 'N_pts xy -> 1 N_pts xy', xy=2)),
        'batch N_pts C -> (batch N_pts C)', batch=1, C=1)

    current_coords = ensure_numpy(current_coords)
    current_occlusions = np.float32(ensure_numpy(current_occlusions))

    return current_coords, current_occlusions
