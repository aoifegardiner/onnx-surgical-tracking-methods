import numpy as np
import torch
from src.lite_tracker import LiteTracker
from pathlib import Path


class LiteTrackerWrapper:

    def __init__(
        self,
        weights_path: Path,
        return_vis: bool,
    ):
        self.modeltype = "LiteTracker"
        self.return_vis = return_vis
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        self.model = LiteTracker()
        self.model = self.model.to(device=self.device)

        assert weights_path.exists(), f"weights_path {weights_path} does not exist"
        with open(weights_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.init_video_online_processing()

        self.queries = None
        self.is_first_frame = True

    @torch.no_grad()
    def __preprocess_frame(self, im: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a single video frame for input to the LiteTracker model.

        Converts the input image from (1, H, W, 3) uint8 format to (1, 3, H, W) float32 format,
        moves it to the correct device and dtype.

        Args:
            im (torch.Tensor): Input image tensor of shape (1, H, W, 3), values in [0, 255], dtype uint8.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape (1, 3, H, W), values in [0, 255], dtype float32.
        """
        im = im.permute(0, 3, 1, 2).to(dtype=self.dtype, device=self.device)
        return im

    @torch.no_grad()
    def trackpoints2D(
        self,
        pointlist: np.ndarray,
        impair: list,
    ):
        """
        Track 2D points between two consecutive video frames using LiteTracker.

        On the first call, initializes the tracker with the given pointlist and first frame.
        On subsequent calls, tracks the points from the previous frame to the next frame.

        Args:
            pointlist (np.ndarray): Array of shape (N, 2) with point coordinates (x, y) in original image size, dtype int64.
                Can be None if not the first frame.
            impair (list): List of two tensors [im0, im1], each of shape (1, H, W, 3), values in [0, 255], dtype uint8.

        Returns:
            np.ndarray: Tracked point coordinates of shape (N, 2), dtype float32.
            np.ndarray (optional): Visibility mask of shape (N,) if self.return_vis is True.
        """
        with torch.autocast(
            device_type=self.device,
            dtype=self.dtype,
            enabled=True,
        ):

            if self.is_first_frame:
                first_im = impair[0]
                first_im = self.__preprocess_frame(first_im)

                # Convert the pointlist to torch
                self.queries = (
                    torch.from_numpy(pointlist)
                    .float()
                    .unsqueeze(0)
                    .to(device=self.device)
                )
                # Append the index of the query's initial frame
                self.queries = torch.cat(
                    (torch.zeros_like(self.queries[:, :, :1]), self.queries), dim=-1
                )
                self.is_first_frame = False
                self.model(first_im, queries=self.queries)

            second_im = impair[1]
            second_im = self.__preprocess_frame(second_im)
            coords, vis, *_ = self.model(second_im, queries=self.queries)
            pointlist = coords[0, 0].cpu().numpy()
            vis = vis[0].cpu().numpy()
            if self.return_vis:
                return pointlist, vis
            else:
                return pointlist
