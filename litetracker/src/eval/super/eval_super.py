import torch
import json
import numpy as np
import os
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
from src.eval.lite_tracker_wrapper import LiteTrackerWrapper


def prepare_gt_data(data_dir: Path):
    """
    Prepare ground truth data from the SuPerDataset annotation file.

    Loads the metadata and extracts ground truth point locations and visibilities for each image.

    Args:
        data_dir (Path): Path to the directory containing 'SuPerDataset.json'.

    Returns:
        labeled_filenames (list): List of image filenames, shape (num_images,).
        location_gt (np.ndarray): Ground truth locations, shape (num_images, num_points, 2).
        visibility_gt (np.ndarray): Ground truth visibilities, shape (num_images, num_points).
    """
    meta_data = json.load(open(data_dir / "SuPerDataset.json"))

    tracking_data = meta_data["_via_img_metadata"]
    num_points = len(list(tracking_data.values())[0]["regions"])

    location_gt_map = {}  # key: filename, value: dict with key: point_id, value: (x, y)
    visibility_gt_map = {}  # key: filename, value: dict with key: point_id, value: bool

    for k, v in tracking_data.items():
        regions = v["regions"]
        filename = v["filename"]

        locations = {}
        visibility = {}

        for point_id in range(num_points):
            locations[point_id] = (-1.0, -1.0)
            visibility[point_id] = False

        for region in regions:
            point_id = int(region["region_attributes"]["id"])
            location = (
                float(region["shape_attributes"]["cx"]),
                float(region["shape_attributes"]["cy"]),
            )
            locations[point_id] = location
            visibility[point_id] = True

        location_gt_map[filename] = locations
        visibility_gt_map[filename] = visibility

    # Convert the gt data to np.arrays with:
    # location - shape: (num_images, num_points, 2)
    # visibility - shape: (num_images, num_points)
    location_gt = []
    visibility_gt = []
    labeled_filenames = []

    for k in location_gt_map.keys():
        locations = []
        visibility = []
        for point_id in range(num_points):
            locations.append(location_gt_map[k][point_id])
            visibility.append(visibility_gt_map[k][point_id])
        location_gt.append(locations)
        visibility_gt.append(visibility)
        labeled_filenames.append(k)

    location_gt = np.array(location_gt)
    visibility_gt = np.array(visibility_gt)

    return labeled_filenames, location_gt, visibility_gt


def compute_evaluation_metrics(
    location_gt: np.ndarray,
    visibility_gt: np.ndarray,
    location_pred: np.ndarray,
    visibility_pred: np.ndarray,
    original_shape: tuple = (640, 480),
    target_shape: tuple = (256, 256),
    thresholds: tuple = (1, 2, 4, 8, 16),
):
    """
    Compute evaluation metrics for predicted and ground truth point locations and visibilities.

    Calculates the Average Jaccard (AJ) at multiple distance thresholds and occlusion accuracy (OA).
    Rescales coordinates from original image shape to target shape before computing metrics.

    Args:
        location_gt (np.ndarray): Ground truth locations, shape (num_images, num_points, 2).
        visibility_gt (np.ndarray): Ground truth visibilities, shape (num_images, num_points).
        location_pred (np.ndarray): Predicted locations, shape (num_images, num_points, 2).
        visibility_pred (np.ndarray): Predicted visibilities, shape (num_images, num_points).
        original_shape (tuple, optional): Original image shape (width, height). Default: (640, 480).
        target_shape (tuple, optional): Target image shape (width, height). Default: (256, 256).
        thresholds (tuple, optional): Distance thresholds for AJ metric. Default: (1, 2, 4, 8, 16).

    Returns:
        None. Prints evaluation results to stdout.
    """
    assert location_gt.shape == location_pred.shape
    assert visibility_gt.shape == visibility_pred.shape
    assert location_gt.shape[:2] == visibility_gt.shape

    shape_scale = np.array(
        [
            (target_shape[0]) / (original_shape[0]),
            (target_shape[1]) / (original_shape[1]),
        ]
    )

    ajs = []
    for th in thresholds:
        num_true_positive = 0
        num_false_positive = 0
        num_false_negative = 0
        num_correct_visibility = 0
        total_items = 0
        for frame_id in range(location_gt.shape[0]):
            for point_id in range(location_gt.shape[1]):
                current_location_gt = location_gt[frame_id][point_id] * shape_scale
                current_visibility_gt = visibility_gt[frame_id][point_id]
                current_location_pred = location_pred[frame_id][point_id] * shape_scale
                current_visibility_pred = visibility_pred[frame_id][point_id]

                distance = np.linalg.norm(
                    np.array(current_location_gt) - np.array(current_location_pred)
                )
                is_within_distance = distance < th
                is_true_positive = (
                    is_within_distance
                    and current_visibility_gt
                    and current_visibility_pred
                )
                is_false_positive = current_visibility_pred and (
                    not current_visibility_gt or not is_within_distance
                )
                is_false_negative = current_visibility_gt and (
                    not current_visibility_pred or not is_within_distance
                )

                if is_true_positive:
                    num_true_positive += 1
                if is_false_positive:
                    num_false_positive += 1
                if is_false_negative:
                    num_false_negative += 1

                is_visibility_correct = current_visibility_gt == current_visibility_pred
                if is_visibility_correct:
                    num_correct_visibility += 1
                total_items += 1

        aj = num_true_positive / (
            num_true_positive + num_false_positive + num_false_negative
        )
        occlusion_accuracy = num_correct_visibility / total_items
        ajs.append(aj)

    for th, aj in zip(thresholds, ajs):
        print(f"AJ - {th}px:\t{(aj * 100):.2f}")

    print(f"AJ - Avg:\t{np.mean(ajs) * 100:.2f}")

    print(f"OA:\t{(occlusion_accuracy * 100):.2f}")


def process_filenames_regex(filenames):
    import re

    result = []
    for filename in filenames:
        match = re.search(r"(\d{3})(\d{3})-left\.png", filename)
        if match:
            last_three = match.group(2)

            num = int(last_three) - 10

            result.append(str(num))

    return result


def main(args: argparse.Namespace):
    labeled_filenames, location_gt, visibility_gt = prepare_gt_data(
        data_dir=args.data_dir
    )
    model = LiteTrackerWrapper(weights_path=args.weights_path, return_vis=True)

    filenames = list(os.listdir(args.data_dir / "img"))
    filenames.sort()
    first_filename = labeled_filenames[0]

    # Remove all the filenames that is before the first_filename
    while filenames[0] != first_filename:
        filenames.pop(0)

    # Deepcopy the location_gt and visibility_gt to initialize predictions
    location_control = location_gt.copy()
    visibility_control = visibility_gt.copy()
    location_control[:] = location_gt[0]
    visibility_control[:] = visibility_gt[0]
    location_pred = location_control.copy()
    visibility_pred = visibility_control.copy()

    pointlist = location_gt[0].copy()

    for i in tqdm(range(len(filenames) - 1)):
        img0 = cv2.imread(str(args.data_dir / "img" / filenames[i]))
        img1 = cv2.imread(str(args.data_dir / "img" / filenames[i + 1]))

        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        img0 = torch.from_numpy(img0).unsqueeze(0)
        img1 = torch.from_numpy(img1).unsqueeze(0)

        current_location_pred, current_visibility_pred = model.trackpoints2D(
            pointlist, [img0, img1]
        )
        if filenames[i + 1] in labeled_filenames:
            filename_index = labeled_filenames.index(filenames[i + 1])
            location_pred[filename_index] = current_location_pred
            visibility_pred[filename_index] = current_visibility_pred

    # Calculate the error
    # Drop the initial frame from the evaluation because that is the first frame anyways
    location_gt = location_gt[1:]
    visibility_gt = visibility_gt[1:]
    location_pred = location_pred[1:]
    visibility_pred = visibility_pred[1:]
    location_control = location_control[1:]
    visibility_control = visibility_control[1:]

    compute_evaluation_metrics(
        location_gt, visibility_gt, location_pred, visibility_pred
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        help="Path to the SuPerDataset directory containing the images and metadata",
    )
    parser.add_argument(
        "-w",
        "--weights_path",
        type=Path,
        help="Path to the .pth file containing the model weights",
    )

    args = parser.parse_args()
    main(args)
