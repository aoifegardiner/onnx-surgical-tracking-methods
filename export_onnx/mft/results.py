"""Flow, occlusion, and uncertainty fields for the MFT delta chain.

Adapted from MFT (Neoral, Serych, Matas, WACV 2024), CC BY-NC-SA 4.0.
See LICENSE in this directory.
"""
import einops
import torch
import torch.nn.functional as F
from .flow_ops import ensure_torch, normalize_coords, torch_get_featuremap_coords


class FlowOUTrackingResult(object):
    def __init__(self, flow, occlusion=None, sigma=None):
        """Stores optical flow, occlusion map and flow sigma map.

        flow: (xy-delta, H, W) tensor
        occlusion: (1, H, W) tensor
        sigma: (1, H, W) tensor
        """
        assert len(flow.shape) == 3
        assert flow.shape[0] == 2
        self.H, self.W = flow.shape[1:]
        if occlusion is None:
            occlusion = torch.zeros((1, self.H, self.W), dtype=torch.float32)
        if sigma is None:
            sigma = torch.zeros((1, self.H, self.W), dtype=torch.float32)

        assert flow.shape == (2, self.H, self.W)
        assert occlusion.shape == (1, self.H, self.W)
        assert sigma.shape == (1, self.H, self.W)

        assert torch.all(occlusion >= 0)
        assert torch.all(occlusion <= 1.000001)
        assert torch.all(sigma >= 0)

        self.flow = flow  # (xy-delta, H, W) tensor
        self.occlusion = occlusion  # (1, H, W) tensor
        self.sigma = sigma  # (1, H, W) tensor

    def __repr__(self):
        cls = self.__class__.__name__
        txt = f'<{cls} ({self.H} x {self.W}) has flow, occlusion, sigma>'
        return txt

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        if torch.cuda.is_available():
            return self.to("cuda")
        return self

    def to(self, device):
        self.flow = self.flow.to(device)
        self.occlusion = self.occlusion.to(device)
        self.sigma = self.sigma.to(device)
        return self

    def clone(self):
        return FlowOUTrackingResult(self.flow.clone(),
                                    self.occlusion.clone(),
                                    self.sigma.clone())

    @classmethod
    def identity(cls, flow_shape, device=None):
        """Create a zero-flow, zero-sigma, zero-occlusion result

        args:
            flow_shape: (H, W) tuple"""
        xy = 2
        flow = torch.zeros((xy, flow_shape[0], flow_shape[1]), dtype=torch.float32, device=device)
        occlusion = torch.zeros((1, flow_shape[0], flow_shape[1]), dtype=torch.float32, device=device)
        sigma = torch.zeros((1, flow_shape[0], flow_shape[1]), dtype=torch.float32, device=device)

        return FlowOUTrackingResult(flow, occlusion=occlusion, sigma=sigma)

    def chain(self, flow):
        """Chain 'flow' after our result.

        With 'self.flow' from A to B, and 'flow' from B to C, this
        function produces a flow from A to C by bilinear
        interpolation.

        args:
            flow: (xy H W) tensor with flow from B to C
        return:
            chained_flow: (xy H W) tensor with flow from A to C
        """
        device = flow.device
        flow_shape = einops.parse_shape(flow, 'xy H W')
        assert flow_shape['xy'] == 2
        coords_A = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']),
                                               device=device, keep_shape=True)
        coords_B = coords_A + self.flow.to(device).to(torch.float32)
        coords_B_normed = normalize_coords(einops.rearrange(coords_B, 'xy H W -> 1 H W xy', xy=2),
                                                         flow_shape['H'], flow_shape['W'])

        flow_BC = flow.to(torch.float32)
        flow_BC_sampled = F.grid_sample(einops.rearrange(flow_BC, 'xy H W -> 1 xy H W', xy=2),
                                        coords_B_normed,
                                        align_corners=True)
        flow_AC = coords_B + flow_BC_sampled - coords_A
        chained_flow = einops.rearrange(flow_AC, '1 xy H W -> xy H W', xy=2)
        return chained_flow

    def warp_backward(self, img):
        """Sample the img data using the right end of 'self.flow'

        args:
            img: (C, H, W) tensor

        return:
            img_sampled: (C, flow_H, flow_W) tensor
        """
        assert len(img.shape) == 3
        assert img.shape[1:] == (self.H, self.W)
        device = img.device
        coords_A = torch_get_featuremap_coords((self.H, self.W), device=device, keep_shape=True)
        coords_B = coords_A + self.flow.to(device).to(torch.float32)
        coords_B_normed = normalize_coords(einops.rearrange(coords_B, 'xy H W -> 1 H W xy', xy=2),
                                                         self.H, self.W)

        img_sampled = F.grid_sample(einops.rearrange(img, 'C H W -> 1 C H W'),
                                    coords_B_normed,
                                    align_corners=True)
        return einops.rearrange(img_sampled, '1 C H W -> C H W', H=self.H, W=self.W)

    def warp_forward_points(self, points):
        """Warp the points using the stored optical flow.

        args:
            points: (N xy) tensor with source coordinates.
        return:
            warped_points: (N xy) tensor with warped coordinates
        """
        points = ensure_torch(points).to(torch.float32)
        N = points.shape[0]
        device = points.device
        # (we will keep N points in the usual Width dimension and keep Height = 1)
        points_normed = normalize_coords(einops.rearrange(points, 'N xy -> 1 1 N xy', xy=2),
                                                       H=self.H, W=self.W)

        flow = self.flow.to(device).to(torch.float32)
        flow_sampled = F.grid_sample(einops.rearrange(flow, 'xy H W -> 1 xy H W', xy=2),
                                     points_normed, align_corners=True)
        warped_points = points + einops.rearrange(flow_sampled, '1 xy 1 N -> N xy', xy=2, N=N)
        return warped_points

    def sample(self, points):
        """Sample the results at the given points.

        args:
            points: (N xy) tensor with query coordinates
        return:
            sampled_flow: (xy N)
            sampled_occlusions: (1 N)
            sampled_sigmas: (1 N)
        """
        points = ensure_torch(points).to(torch.float32)
        N = points.shape[0]
        device = points.device
        # (we will keep N points in the usual Width dimension and keep Height = 1)
        points_normed = normalize_coords(einops.rearrange(points, 'N xy -> 1 1 N xy', xy=2),
                                                       H=self.H, W=self.W)

        flow = self.flow.to(device).to(torch.float32)
        sampled_flow = F.grid_sample(einops.rearrange(flow, 'xy H W -> 1 xy H W', xy=2),
                                     points_normed, align_corners=True)
        sampled_flow = einops.rearrange(sampled_flow, '1 xy 1 N -> xy N', xy=2, N=N)
        sampled_occlusions = F.grid_sample(
            einops.rearrange(self.occlusion.to(device).to(torch.float32), '1 H W -> 1 1 H W'),
            points_normed, align_corners=True)
        sampled_occlusions = einops.rearrange(sampled_occlusions, '1 c 1 N -> c N', c=1, N=N)
        sampled_sigmas = F.grid_sample(
            einops.rearrange(self.sigma.to(device).to(torch.float32), '1 H W -> 1 1 H W'),
            points_normed, align_corners=True)
        sampled_sigmas = einops.rearrange(sampled_sigmas, '1 c 1 N -> c N', c=1, N=N)
        return sampled_flow, sampled_occlusions, sampled_sigmas

    def invalid_mask(self):
        """Compute a mask of invalid self.flows, i.e. pointing outside of the image.

        returns:
            invalid_mask: (H, W) bool tensor, with True meaning the flow is invalid
        """
        device = self.flow.device
        coords_A = torch_get_featuremap_coords((self.H, self.W), device=device, keep_shape=True)
        coords_B = coords_A + self.flow.to(torch.float32)  # xy, H, W

        invalid_mask = torch.logical_or(
            torch.any(coords_B < 0, dim=0),
            torch.logical_or(
                coords_B[0, :, :] >= self.W,
                coords_B[1, :, :] >= self.H))
        return invalid_mask
