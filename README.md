# ONNX point tracking on STIR and StereoMIS

This repository evaluates published ONNX point trackers. It downloads the graphs, runs them on [STIR 2024](https://zenodo.org/records/14803158) and [StereoMIS tracking](https://doi.org/10.5281/zenodo.10867949), and writes accuracy and latency JSON.

The ONNX files are not in git. They are downloaded from the Hugging Face dataset [aoifegardiner/ONNX_Tracking_Models](https://huggingface.co/datasets/aoifegardiner/ONNX_Tracking_Models).

## What one run does

A method is an ONNX graph plus a short host that feeds frames and reads tracks. Queries come from the dataset labels. The host starts those points on frame 0 and follows them to the end of the clip.

| Method | ONNX graph | What the host does |
| --- | --- | --- |
| RAFT, WAFT | One dense-flow graph per image pair | Samples the flow at the query points |
| MFT | RAFT-OU flow graph | Runs the MFT delta chain around that flow |
| AllTracker | A 16-frame window graph, plus a 2-frame pair graph | Slides the window the same way as `forward_sliding` |
| PIPs++ | One online step graph | Feeds the official growing window |
| LocoTrack | A tracker graph and a feature graph | Keeps the first frame and a growing clip, capped at 300 frames |
| CoTracker3 | One online step graph (`T=16`, step 8) | Carries the previous window state into the next call |
| LiteTracker | `fnet`, `corr_mlp`, and `updateformer` | Runs the online buffers around those three networks |
| TAPNext++ | One step graph per query count | Carries the recurrent state from frame to frame |

STIR frames are 1024×1280 at 26 fps. StereoMIS left frames are 512×640 at 15 fps.

STIR reports end-point error in pixels against the labelled end points, plus the fraction of points inside 4, 8, 16, 32, and 64 px. StereoMIS reports mean trajectory error, delta, and survival, using the same formulas as the StereoMIS point-tracking evaluation.

## Why some methods have one graph per scale

Scale 1, 2, and 4 is the same control as in the PyTorch evaluation. The scale divides the camera resolution by that integer, the tracker runs on the smaller image, and the predicted points are mapped back to the original pixels before the error is computed. A coarser scale is faster and less accurate.

| Dataset | Scale 1 | Scale 2 | Scale 4 |
| --- | --- | --- | --- |
| STIR | 1024×1280 | 512×640 | 256×320 |
| StereoMIS | 512×640 | 256×320 | 128×160 |

RAFT, WAFT, MFT, AllTracker, PIPs++, and LocoTrack track at that downsampled size. Each ONNX graph was traced with height and width fixed, so the 1024×1280 graph cannot accept a 256×320 tensor. Each dataset and each scale is a separate file. AllTracker also needs its pair graph at that same size, and LocoTrack also needs its feature graph at that same size.

CoTracker3, LiteTracker, and TAPNext++ resize every frame to the resolution they were trained at, then map the tracks back to the original pixels. Those canvases are 384×512, 384×512, and 256×256. The tensor that enters the graph does not change when `--scale` changes, so one graph covers every scale. The sweep runs those three methods at scale 1 only.

## Layout

```
export_onnx/     evaluation entry points and ONNX hosts
datasets/        STIR is read in the STIR script; StereoMIS is read here
models/          one JSON sidecar per graph, plus the downloaded .onnx files
MFT_original/    MFT delta chain used around the MFT ONNX flow graph
litetracker/     LiteTracker host used around its three ONNX networks
holoscan_agx/    optional Holoscan operators for an NVIDIA AGX
scripts/         download and sweep helpers
```

Each sidecar (`models/<METHOD>/<name>.json`) records the input names, output names, and resolution. The matching `.onnx` file is downloaded beside it.

## Setup

Python 3.10 or newer. A CUDA GPU is what the latency numbers were measured on. ONNX Runtime falls back to CPU when CUDA is unavailable.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`onnxruntime-gpu` 1.23 expects CUDA 12 and cuDNN 9. Installing the `nvidia-*` wheels with that package is enough; the STIR script adds those libraries to the process before it creates a session.

## Download the models

```bash
python scripts/download_models.py
```

That downloads only the graphs the evaluation opens: STIR and StereoMIS scales 1, 2, and 4 for RAFT, WAFT, MFT, AllTracker (window and pair), PIPs, and LocoTrack (tracker and feature graph), plus `cotracker_online_step.onnx` and the three LiteTracker graphs. Other files in the dataset are not downloaded. To fetch one family:

```bash
python scripts/download_models.py --only raft_stir
```

`python scripts/download_models.py --list` prints the paths without downloading. Files already in `models/` are skipped. `--force` downloads them again. `--fp16` adds `cotracker_online_step_fp16.onnx`, which the eval uses only when passed `--fp16`.

The script reports two groups it will not fetch, because the evaluation needs them and the dataset does not have them:

* StereoMIS RAFT and WAFT. Those runs exit with status 2 and the sweep marks them skipped. The STIR RAFT and WAFT graphs are published.
* TAPNext++ per query count, `tapnextpp_online_step_n<N>.onnx`. The dataset has a single `tapnextpp_online_step.onnx`, which this evaluation does not load. TAPNext++ is skipped.

## Data

STIR 2024, extracted so this path exists:

```
$STIR_DATA/<case>/left*/seq*/frames/*.mp4
$STIR_DATA/<case>/left*/seq*/segmentation/icgstartseg.png
```

The challenge release is [STIR Challenge 2024 on Zenodo](https://zenodo.org/records/14803158). The dataset paper is [Surgical Tattoos in Infrared](https://arxiv.org/abs/2309.16782) and the challenge paper is [arXiv:2503.24306](https://arxiv.org/abs/2503.24306).

StereoMIS tracking, extracted so each sequence looks like:

```
$STEREOMIS_DATA/P1_1/track_pts.pckl
$STEREOMIS_DATA/P1_1/video_frames/*l.png
```

The annotated sequences used here are `P1_1`, `P2_0`, `P2_1`, `P2_6`, `P3_1`, and `P3_2`. The tracking release is [10.5281/zenodo.10867949](https://doi.org/10.5281/zenodo.10867949).

## Run

One STIR sequence with RAFT at native resolution:

```bash
export STIR_DATA=/path/to/STIRChallenge_2024
bash scripts/run_stir.sh --gpu 0 --methods RAFT --num-seq 1
```

`--num-seq` without a value of `-1` keeps the first N sequences in sorted path order. Omit it, or pass `-1`, to run all 60 left-camera STIR clips.

StereoMIS, all annotated sequences, MFT at every scale:

```bash
export STEREOMIS_DATA=/path/to/StereoMIS-tracking
bash scripts/run_stereomis.sh --gpu 0 --methods MFT
```

A single method can also be launched directly:

```bash
python export_onnx/run_all_onnx_stir.py \
  --datadir "$STIR_DATA" \
  --method COTRACKER \
  --scale 1 \
  --num_seq 1 \
  --outdir outputs/stir/COTRACKER/scale1
```

Set `PYTHONPATH` to the repository root and to `litetracker/` before that command. The shell scripts do this. For CoTracker, LiteTracker, and TAPNext++, `--scale` does not select a different graph; see [Why some methods have one graph per scale](#why-some-methods-have-one-graph-per-scale).

## Outputs

Results land in `outputs/stir/<METHOD>/scale<N>/` or `outputs/stereomis/<METHOD>/scale<N>/`.

STIR writes:

* `positions_*.json` — predicted end points
* `gt_positions_start_*.json`, `gt_positions_end_*.json` — labelled start and end points
* `endpointerror_*.json` — mean end-point error, including a static-point control
* `accuracy_*.json` — fraction of points within each pixel threshold
* `latencies_*.json`, `latency_stats_*.json` — per-call milliseconds
* `pipeline_*.json` — wall time from the first pair through the last, in ms/pair and fps

The last line of a STIR run is the summary. A full 60-sequence RAFT scale-1 run on the published graph ends with mean end-point error 44.43 px and average accuracy 0.6113. The same line also prints pipeline frames per second, milliseconds per pair, and whether that rate meets the camera period (1000/26 ms on STIR, 1000/15 ms on StereoMIS). Those timing fields depend on the GPU. The latency median, P95, and P99 drop the first 10 calls. The pipeline figure keeps every frame, including the first.

StereoMIS writes `metrics_*.json` with `MTE_2D`, `delta_2D`, and `survival_2D` per sequence, plus the same latency files.

A run that cannot find its ONNX file prints `SKIP (no STIR ONNX)` or the StereoMIS equivalent and returns exit code 2. The sweep continues.

## AGX

`holoscan_agx/` is the Holoscan path used on an NVIDIA AGX. Dense flow can go through Holoscan Inference. CoTracker, PIPs, LocoTrack, AllTracker, and MFT stay in the Python hosts in that folder, because those graphs use GridSample and a variable clip length. Holoscan itself is not installed by `requirements.txt`.

## Third-party code

`MFT_original/` is the MFT tracker used to chain flow, under the license in `MFT_original/LICENSE`. `litetracker/` is the LiteTracker host, under `litetracker/LICENSE.md`. The ONNX graphs replace the original network weights; the PyTorch checkpoints those projects ship are not included.
