"""StereoMIS 2D scores, matching PointTracker's definitions.

``mte`` is the mean, over points, of the median L2 error on valid frames.
``delta_2d`` and ``surv_2d`` scale pixels so that the frame is 256 units tall
and wide, then score thresholds in that space. Timestep 0 is excluded from
``delta_2d``.
"""

from __future__ import annotations

import numpy as np


def mte(predicted, gt, valid):
    error = gt - predicted
    l2_2ds = []
    for i in range(error.shape[0]):
        l2_2ds.append(np.nanmedian(np.linalg.norm(error[i][valid[i]], ord=2, axis=-1)))
    return np.mean(np.asarray(l2_2ds))


def delta_2d(predicted, gt, valid, height, width, thrs=(1, 2, 4, 8, 16)):
    d_sum = 0.0
    sx_ = width / 256.0
    sy_ = height / 256.0
    sc_pt = np.array([sx_, sy_]).reshape([1, 1, 2])
    for thr in thrs:
        d_ = np.linalg.norm(predicted[:, 1:] / sc_pt - gt[:, 1:] / sc_pt, axis=-1, ord=2) < thr
        d_sum += np.mean(d_[valid[:, 1:]])
    return d_sum / len(thrs)


def surv_2d(predicted, gt, valid, height, width, sur_thr=16):
    sx_ = width / 256.0
    sy_ = height / 256.0
    sc_pt = np.array([sx_, sy_]).reshape([1, 1, 2])
    dists = np.linalg.norm(predicted / sc_pt - gt / sc_pt, axis=-1, ord=2)
    dist_ok = 1 - (dists > sur_thr) * valid
    survival = np.cumprod(dist_ok, axis=1)
    return np.mean(survival)
