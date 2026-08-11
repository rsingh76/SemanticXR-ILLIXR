<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Semantic SLAM Server

A real-time semantic SLAM system that combines object detection, segmentation, and CLIP-based understanding for interactive 3D scene mapping and querying.

## Overview

This semantic SLAM server provides:
- **Real-time 3D mapping** with object-level semantic understanding
- **Multi-modal AI pipeline** (detection → segmentation → CLIP encoding → mapping)
- **Interactive querying** using natural language descriptions
- **Configurable architecture** supporting multiple datasets and hardware setups
- **Performance monitoring** with detailed logging and analytics

## Clean Python API (`semantic_slam`)

A small, importable façade over the full pipeline. Push RGB/depth/pose frames
and query the resulting semantic map with natural language — no gRPC, no
multiprocess server.

```python
import numpy as np
from semantic_slam import SemanticSLAM

slam = SemanticSLAM(dataset_type="replica", scene_name="room0")  # builds models
slam.push(rgb, depth, pose)          # rgb HxWx3 uint8, depth HxW, pose 4x4 (or flat-16)
hits = slam.query("a brown chair", top_k=5)
for h in hits:
    print(h["score"], h["centroid"], h["class_name"])
slam.save_map("map.pkl.gz")          # also: load_map(), reset()
```

### Install with uv (replaces the old conda env)

Dependencies are managed by [uv](https://docs.astral.sh/uv/). The core stack is
locked in `pyproject.toml` / `uv.lock`; the heavy model stack (pytorch3d,
gradslam, segment-anything — all build CUDA extensions / come from git) plus the
SAM weights are installed by a **one-time, idempotent setup script** that runs
automatically on first launch:

```bash
uv sync                       # core deps into an isolated .venv
scripts/setup_env.sh          # model stack + SAM weights (idempotent; --with-dataset adds Replica)
```

You normally don't need to call the script yourself — the first
`SemanticSLAM(...)` runs it if anything is missing (disable with
`SEMANTIC_SLAM_AUTO_SETUP=0`). The script installs the CUDA build toolchain
(`nvcc`, `gcc-10`/`g++-10`) via `apt` only if it is absent, generates the gRPC
protobuf stubs, and downloads the SAM weights.

> **Note:** the source-built model stack lives in the venv but not the lockfile.
> Use `uv sync --inexact` (as the script does) — a plain `uv sync` would remove
> it. If it ever gets removed, just re-run `scripts/setup_env.sh` (cached builds
> make it fast).

### End-to-end test

```bash
REPLICA_ROOT=/data/replica/Replica .venv/bin/python -m pytest tests/test_replica_e2e.py -q -s
```

The CPU-only smoke test (`tests/test_semantic_slam_smoke.py`) needs none of the
above and runs anywhere.

### Visual validation

To eyeball that it works, `scripts/visualize_e2e.py` pushes a few Replica frames
and writes images to `outputs/semantic_slam_viz/`:

```bash
REPLICA_ROOT=/data/replica/Replica GSA_PATH=$PWD/external/Grounded-Segment-Anything \
  HF_HOME=$HOME/.cache/huggingface SEMANTIC_SLAM_AUTO_SETUP=0 \
  .venv/bin/python scripts/visualize_e2e.py
```

- `frame_XXXXXX_seg.png` — SAM masks overlaid on each input RGB frame.
- `map_bev.png` — bird's-eye scatter of all mapped object centroids.
- `query_<text>.png` — the BEV map with the top-k hits for a text query ringed
  and scored, so you can see where a query localizes in the scene.

### Use as a module in `nvidia/xr-ai`

The package is shaped to drop into `xr-ai`'s `ai-services/` layout (hatchling,
Apache-2.0 SPDX, pinned PyTorch CUDA index, `[project.scripts]` entry point),
matching the existing `vlm-server` / `stt-server`. To consume it from xr-ai,
add it as a path/git dependency and import `semantic_slam`.

## Quick Start

**Prerequisites:** Complete the [Installation](#installation) section first.

```bash
# 1. Install the package in development mode (from repo root)
pip install -e .

# 2. Run replay against a captured Quest scene (writes outputs into the
#    same dataset directory; see "Capture, replay, and run-output layout").
python server/main.py --dataset_type quest --localDataset \
    --sceneName dataset_0 --config config/debug/quest_debug.yaml --save_map

# 3. Monitor performance (the run dir is printed at startup; tail the latest)
tail -f $(ls -td datasets/*/*/logs_performance/*/frame_timing.csv \
                 live_output/*/*/logs_performance/*/frame_timing.csv \
                 output/*/*/logs_performance/*/frame_timing.csv 2>/dev/null | head -1)
```

For live (gRPC) Quest sessions and the broader directory contract, see
[Capture, replay, and run-output layout](#capture-replay-and-run-output-layout).

## Project Structure

```
semantic-slam-server/
├── server/                          # Entry point + gRPC server
│   ├── main.py                      # Main entry point (live + replay)
│   ├── components/                  # gRPC handlers, video decoders, inference service
│   ├── signal_handlers.py
│   └── xr_service.proto             # gRPC service definition
├── slam/                            # Core SLAM functionality
│   ├── models/                      # AI models (detection, segmentation, CLIP, captioning)
│   ├── services/                    # Inference + visualization pipelines
│   ├── core/                        # SLAM data structures
│   ├── datasets/                    # Dataset loaders (Quest, Replica, ScanNet)
│   └── utils/                       # Perf manager, debug dumps, mapping helpers
├── config/                          # YAML configuration profiles
│   ├── defaults.yaml                # Default profile
│   ├── baseline.yaml                # Single-GPU baseline
│   ├── parallelization_mobileclip*.yaml   # Faster pipelines (MobileCLIP variants)
│   └── debug/                       # Debug-enabled variants (quest_debug.yaml, ...)
├── datasets/<type>/dataset_<N>/     # Captured scenes + replay outputs (runtime-created)
│   ├── intrinsics.json
│   ├── decoded_jpg/, depth/, meta/  # per-frame inputs
│   └── pcd_saves/, debug_dumps/, logs_performance/   # per-replay-run outputs
├── live_output/<type>/run_<N>/      # Live-stream outputs (runtime-created)
│   └── pcd_saves/, debug_dumps/, logs_performance/
├── output/<type>/<scene>/           # Replay outputs for read-only datasets (Replica, ScanNet)
└── external/                        # Vendored dependencies
    ├── gradslam/, chamferdist/, Grounded-Segment-Anything/
```

## Installation

### Dependencies Setup

1. Create a new conda environment with Python 3.10:
    ```
    conda create -n sceneGraphDemo python=3.10
    ```

2. Activate the sceneGraphDemo environment:
    ```
    conda activate sceneGraphDemo
    ```

3. Install the required packages:
    ```
    pip install tyro open_clip_torch wandb h5py openai hydra-core distinctipy timm==0.4.12
    pip install pynvvideocodec
    python -m pip install grpcio
    python -m pip install grpcio-tools
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    ```

    H.264/H.265 video decode is handled by NVIDIA's PyNvVideoCodec (Python bindings, MIT-licensed) backed by the NVIDIA Video Codec SDK. Download the SDK from https://developer.nvidia.com/video-codec-sdk (requires NVIDIA driver / CUDA).

4. Set the CUDA_HOME environment variable to the CUDA installation inside the conda environment:
    ```
    export CUDA_HOME=/home/<USERNAME>/miniconda3/envs/sceneGraphDemo/
    ```

5. Install additional packages:
    ```
    conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install -c fvcore -c conda-forge fvcore
    ```
    Optional
    ```
    conda install pytorch3d -c pytorch3d
    ```

6. Chamferdist - Module to compute Chamfer distance between two pointclouds. Clone the chamferdist repository and install it:
    ```
    git clone https://github.com/krrish94/chamferdist.git
    cd chamferdist
    Commit ID: ee75389 (TODO: need verification)
    pip install .
    ```

7. GradSLAM - Module used in mapping the point clouds. Clone the gradslam repository and switch to the conceptfusion branch:
    ```
    git clone https://github.com/gradslam/gradslam.git
    cd gradslam
    git checkout conceptfusion
    pip install .
    ```

8. Install the supervision package. This package has some datastructures that we use for object based point clouds:
    ```
    pip install supervision==0.14.0
    ```

9. Install the line_profiler package from conda-forge:
    ```
    conda install conda-forge::line_profiler
    ```

10. Clone the Grounded-Segment-Anything repository and switch to a specific commit:
     ```
     git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
     cd Grounded-Segment-Anything
     git checkout a4d76a2
     ```

11. Download the following files from [here](https://github.com/IDEA-Research/Grounded-Segment-Anything#install-without-docker) and place them in Grounding-Segment-Anything directory
     - ram_swin_large_14m.pth
     - groundingdino_swint_ogc.pth
     - sam_vit_h_4b8939.pth
     - Download the mobile_sam weights from [here](https://github.com/ChaoningZhang/MobileSAM/tree/master/weights) and put it in Grounded-SAM-Anything/EfficientSAM directory.

     ```
     python -m pip install -e segment_anything
     pip install --no-build-isolation -e GroundingDINO
     pip install --upgrade diffusers[torch]
     git submodule update --init --recursive
     cd grounded-sam-osx && bash install.sh
     cd ..
     git clone https://github.com/xinyu1205/recognize-anything.git
     Commit ID: 88c2b0c (TODO: need verification)
     pip install --upgrade setuptools
     pip install -r ./recognize-anything/requirements.txt
     pip install -e ./recognize-anything/
     ```
     
    

12. Set the GSA_PATH environment variables:
     ```
     export GSA_PATH=<path to Grounded-Segment-Anything directory>
     ```

13. Additional considerations - 
     - Perhaps need to downgrade scipy to 1.10, and opencv-python to opencv-python==4.9.0.80
     - Numpy must be - 1.24.3 or 1.26.4
     - pip install tensorrt==8.6.1, Native tensorrt should also be the same version.
     - pip install pycuda
     - Transformer version must be: pip install transformer==4.25.1

14. **Install the Semantic SLAM package** (REQUIRED - final step after all dependencies):
    ```bash
    # From the semantic-slam-server root directory
    pip install -e .
    ```
    **This is the critical step that:**
    - Installs the package in development mode
    - Makes all imports work properly (`from slam.models import ...`)
    - Enables the server to run (`python server/main.py`)
    - Must be done AFTER all dependencies are installed

15. **Verify Installation** (optional but recommended):
    ```bash
    # Test that core imports work
    python tests/test_basic_imports.py
    ```

16. Speech/Audio options:

    ASR is configured via `model.visualization.asr` in your YAML config (see `ASRConfig` in `config/settings.py`). Pick a backend:

    **Option A — openai-whisper with `medium.en` (recommended, best transcription quality in our testing):**
    ```bash
    pip install openai-whisper
    ```
    ```yaml
    # config/defaults.yaml
    model:
      visualization:
        asr:
          enabled: true
          backend: "openai-whisper"
          model: "medium.en"       # smaller sizes (small.en and below) gave noticeably worse transcripts
          device: "cuda:0"
          warmup: true
    ```

    **Option B — faster-whisper (local, CTranslate2-based, faster in theory but not yet verified end-to-end here — expect to debug before relying on it):**
    ```bash
    pip install faster-whisper
    ```
    ```yaml
    asr:
      enabled: true
      backend: "faster-whisper"
      model: "medium.en"           # tiny.en / base.en / small.en / medium.en / large-v3
      device: "cuda:0"
      compute_type: "float16"      # float16 / int8_float16 / int8
      warmup: true
    ```

    **Option C — OpenAI cloud Whisper:**
    ```bash
    pip install openai
    export OPENAI_API_KEY="xxxxxx"
    ```
    ```yaml
    asr:
      enabled: true
      backend: "openai-api"
    ```

    **Option D — No audio, text queries only:**

    Set `asr.enabled: false`. The client is then expected to send the query as text directly, and the server will skip all audio handling. (For headless testing without a client, the legacy workplace-object iterator path still lives in `clientTextQuery` for the non-`ipad`/`quest` dataset types.)

## Usage:

### **Python Package Imports**
After installation, you can import modules cleanly from anywhere:

```python
# Core SLAM functionality
from slam.models import SegmentationModel, DetectionModel, CLIPModel
from slam.core.slam_classes import MapObjectList
from slam.utils.vis import vis_result_fast

# Configuration system
from config.settings import Config, get_config

# Example usage
config = get_config()
segmentation_model = SegmentationModel(config)
detection_model = DetectionModel(config)
```

### **Configuration System**
The server uses a centralized YAML-based configuration system. Pick a profile
from `config/` and pass it as `--config`; CLI flags override individual values.

```bash
# Default profile, replica replay
python server/main.py --dataset_type replica --localDataset --sceneName room0

# Custom profile
python server/main.py --config config/parallelization_mobileclip.yaml \
    --dataset_type quest --localDataset --sceneName dataset_0

# List available options
python server/main.py --help
```

**Available config profiles** (see `config/`):
- `defaults.yaml`, `baseline.yaml` — single-GPU baselines
- `parallelization_mobileclip.yaml`, `…_objectDownsampling.yaml`, `…_2gpus.yaml` — faster MobileCLIP pipelines
- `parallelization_origCLIP.yaml` — full CLIP, slower
- `no_detection.yaml` — pipeline without the detection stage
- `debug/quest_debug.yaml`, `debug/quest_debug_mobileclip.yaml` — Quest sessions with capture knobs
- `debug/defaults_debug.yaml`, `debug/debug_parallel_*.yaml`, `debug/parallelization_debug.yaml` — inference-dump-enabled variants for reproducibility debugging

### **Performance Logging**
All performance metrics are logged under the run's output dir
(``$SLAM_RUN_OUTPUT_DIR``, see *Capture, replay, and run-output layout* below):
```
<run_output_dir>/logs_performance/<config_name>/
├── frame_timing.csv          # detailed per-frame timing data
└── performance_summary.json  # session statistics and averages
```

**Examples:**
- ``datasets/quest/dataset_3/logs_performance/default/`` — replay of capture 3, default config
- ``live_output/quest/run_7/logs_performance/quest_debug/`` — live Quest run 7 with the quest_debug profile

### **Running the Server**

**Basic usage:**
```bash
# Live Quest streaming (gRPC; client connects to this server)
python server/main.py --dataset_type quest --config config/debug/quest_debug.yaml --save_map

# Live iPad streaming
python server/main.py --dataset_type ipad --clientUpdateMode --clientIP 192.168.1.100

# Replay a captured Quest scene (offline; no client needed)
python server/main.py --dataset_type quest --localDataset \
    --sceneName dataset_0 --config config/debug/quest_debug.yaml --save_map

# Replay a Replica scene (REPLICA_ROOT must be exported)
python server/main.py --dataset_type replica --localDataset --sceneName room0 --save_map
```

**Key command-line options:**
- `--dataset_type <type>` — required, one of `quest`, `ipad`, `replica`, `scannet`
- `--config <path>` — YAML profile (default `config/defaults.yaml`)
- `--localDataset` — replay mode (read from disk instead of gRPC)
- `--sceneName <name>` — scene directory name; for Quest, this is `dataset_<N>` under `datasets/quest/`
- `--save_map` — write the final semantic-map pickle on completion
- `--dataset_stride <N>` — process every Nth frame in replay (default 5)
- `--test_depth_downsampling <N>` — override mapping depth-downsample factor; baked into output filename for sweeps
- `--clientUpdateMode` / `--clientIP` — enable client-side viz updates (live mode)
- `--pipelined_mapping` — use pipelined mapping consumer

**Performance monitoring:**
- Real-time FPS and timing statistics logged to console
- Detailed CSV logs at ``$SLAM_RUN_OUTPUT_DIR/logs_performance/<config>/frame_timing.csv`` (see *Capture, replay, and run-output layout* below)
- Graceful shutdown on Ctrl+C with final performance summary

### **Capture, replay, and run-output layout**

The server organises every run's artifacts under a single per-run directory
so a parameter-sweep workflow (capture once, replay many) keeps inputs and
outputs co-located. ``server/main.py`` resolves this directory at startup
and exports it as ``SLAM_RUN_OUTPUT_DIR``; everything written by the
inference workers — pcd dumps, debug dumps, perf logs — hangs off it.

**Top-level layout:**

```
datasets/                            # capture sinks AND replay sources
  quest/
    dataset_0/                       # one captured scene
      intrinsics.json                # scene-level RGB + depth intrinsics + sizes (cy_yup)
      decoded_jpg/frame_NNNNNN.jpg   # RGB at NATIVE resolution
      depth/depth_NNNNNN.npy         # float32 metric depth at NATIVE depth resolution
      meta/meta_NNNNNN.json          # per-frame poses (as received) + timestamps
      pcd_saves/                     # ← replay semantic-map output
      debug_dumps/{inference,visualizations}/   # ← replay debug artifacts
      logs_performance/<config>/     # ← replay perf logs
    dataset_1/ ...
  ipad/
    dataset_0/{results/, traj.txt}   # iPad capture (no replay path today)
    dataset_1/ ...

live_output/                         # live-stream artifacts only
  quest/
    run_0/{pcd_saves/, debug_dumps/, logs_performance/<config>/}
    run_1/ ...
  ipad/
    run_0/ ...
```

Auto-increment is per-type: ``datasets/quest/dataset_0`` and
``datasets/ipad/dataset_0`` exist independently. ``run_<N>`` under
``live_output/<type>/`` increments separately.

**To capture a live Quest session:**

Set ``dataset.enabled=true`` (off by default). Quest frames land under
``datasets/quest/dataset_<N>/`` automatically. Live-run outputs (semantic
map, debug dumps, perf logs) go to ``live_output/quest/run_<N>/``.

What's saved vs what's used at runtime:
- **Poses**: stored exactly as the proto delivers them (OpenXR right-handed,
  Y-up, camera-to-world). The OpenGL→OpenCV flip lives in
  ``QuestDataset.load_poses`` and runs at consumption time — same call site
  for live and replay, so there's no risk of double-flip.
- **Depth**: stored before ``build_depth_in_rgb_frame``. Replay re-runs the
  alignment with whatever target resolution ``QUEST.yaml`` specifies, so you
  can change processing resolution between runs without re-capturing.
- **RGB**: stored at native resolution (no LANCZOS pre-resize), for the
  same reason.

**To replay a captured scene:**

```bash
python server/main.py --localDataset --dataset_type quest --sceneName dataset_0 --save_map
```

The replay loop in ``server/main.py::_process_quest_dataset`` finds frames
by globbing ``meta_*.json`` under ``datasets/quest/<sceneName>/``, runs
them through the same inference pipeline as live, and emits a
``scene_completion`` signal at the end so the inference worker writes the
final semantic map to:

```
datasets/quest/<sceneName>/pcd_saves/semantic_map_<config>_test_depth_downsampling_<N>.pkl.gz
```

Per-stage timings land at
``datasets/quest/<sceneName>/logs_performance/<config>/frame_timing.csv``.
Note that ``queue_overhead`` and ``total_time`` are not directly comparable
between live and replay (live has backpressure / drops; replay processes
sequentially).

## Configuration Details

### **Environment Variables**
A subset of config fields can be overridden via environment variables (handled
in `Config.from_env`, [config/settings.py](config/settings.py)):

```bash
# Per-stage device overrides (also DEVICE for a global override)
export DETECTION_DEVICE="cuda:1"
export SEGMENTATION_DEVICE="cuda:0"
export CLIP_DEVICE="cuda:0"
export CAPTIONING_DEVICE="cuda:0"
export VISUALIZATION_DEVICE="cuda:0"
export MAPPING_DEVICE="cuda:0"

# Model variant overrides
export SAM_VARIANT="mobilesam"
export CLIP_MODEL="ViT-H-14"

# Server / external paths
export SERVER_PORT=50051
export GSA_PATH=/path/to/Grounded-Segment-Anything

# Replica / ScanNet replay roots (read-only public datasets)
export REPLICA_ROOT=/path/to/Replica
export SCANNET_ROOT=/path/to/ScanNet

# Debug dump toggles
export DEBUG_DUMP_INFERENCE=true
export DEBUG_USE_SLOW_VIS=true
```

These layer in on top of the YAML config. For values not exposed as env vars,
edit the YAML or pass `--<flag>` overrides where wired.

### **Model Configuration**
Each AI model can be configured independently:

```yaml
# config/custom_config.yaml
model:
  detection:
    device: "cuda:0"
    enabled: true
    confidence_threshold: 0.35
  segmentation:
    device: "cuda:1" 
    variant: "mobile_sam"
  clip:
    device: "cuda:0"
    model_name: "ViT-H-14"
    batch_size: 8
```

### **Performance Monitoring**
- **Real-time console output:** FPS, frame timing, queue sizes
- **Detailed CSV logs:** Per-frame breakdown of inference times
- **Summary statistics:** Session averages, percentiles, total frames
- **Graceful shutdown:** Ctrl+C writes final performance summary

### **Troubleshooting**

**Import Issues:**
```bash
# Ensure package is installed
pip install -e .

# Test imports
python -c "from slam.models import SegmentationModel; print('✅ Imports working')"
```

**Performance Issues:**
```bash
# Check GPU utilization
nvidia-smi

# Monitor performance logs (the run dir is printed at startup; tail the latest)
tail -f $(ls -td datasets/*/*/logs_performance/*/frame_timing.csv \
                 live_output/*/*/logs_performance/*/frame_timing.csv \
                 output/*/*/logs_performance/*/frame_timing.csv 2>/dev/null | head -1)
```

**Configuration Issues:**
```bash
# Validate configuration
python -c "from config.settings import get_config; print(get_config())"

# Check environment variables
env | grep SLAM_
```

## Licensing

This repository is licensed under Apache-2.0 (see [`LICENSE`](./LICENSE)). The Python package and every dependency declared in `pyproject.toml` is permissive-licensed (BSD-3, MIT, Apache-2.0, HPND, or NVIDIA proprietary runtime SDK). The repo contains:

- **No AGPL/GPL Python deps** (the previously-declared `ultralytics` has been removed).
- **No LGPL Python deps**, and the default code path no longer links into LGPL native libraries (PyAV → PyNvVideoCodec swap; `imageio` is used only for offline PNG reads via Pillow; the optional MP4 animation export is gated behind an actionable error if `imageio-ffmpeg` isn't installed).
- **NVIDIA proprietary runtime SDKs**: CUDA, cuDNN, NCCL, TensorRT, and the NVIDIA Video Codec SDK are required runtime dependencies but are not redistributed by this repository.

For the full dependency inventory and SPDX identifiers, see [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

## TODOs — Visualization Path

Follow-ups identified while moving point-cloud downsampling from query-time
random-200 to receive-time iterative voxel (in `slam/services/visualization_service.py`):

- **Lift voxel constants into config.** `_voxel_downsample_iterative` currently
  hardcodes `voxel=0.03 m`, `cap=150`, `growth=1.5`, `max_iters=8`. These are
  good defaults but should live under `config.model.visualization` so they can
  be tuned per dataset alongside `similarity_threshold`, `colormap`, etc.
- **Scene-total point cap.** Per-object cap is in place via the iterative
  voxel; the next layer is a scene-total cap on the query response — sort
  matched objects by similarity, then further-downsample (or drop) once the
  total point budget is exceeded. Place: right before `createGRPCResponse`
  returns in `clientTextQuery`.
- **Populate `PointCloud.centroid` on the query response.** It's currently
  hardcoded to `[0, 0, 0]` in `createGRPCResponse` and on every push-path
  callsite (`generateClientUpdate`, `_compute_update_message_size_mbits`).
  The proto field is already there; just compute `pcd.mean(axis=0)` before
  the rebind that sub-samples (so it reflects the full point set).
- **Fix misleading comment in inference output.** `inference_pipeline.py:531`
  and `mapping_server.py:337` describe `fresh_objects` as "index of all
  objects that have been touched/edited/added" — they're actually
  `history_idx` values, not current indices. The visualization cache keyed by
  history_idx depends on this; a future "cleanup" that takes the comment
  literally would silently break invalidation.
- **Optional: Open3D-style averaging in voxel downsample.** Current numpy
  implementation uses "first index wins" per voxel cell, which is fine for
  sparse translucent rendering but slightly biased compared to Open3D's
  per-cell averaging. Worth revisiting only if the bias ever becomes visible.
- **Cache memory bound.** `self._downsampled_cache` grows with object count
  (~3.6 KB per object at 150 points × 3 floats × 8 B). Auto-evicts deleted
  objects each frame, but a hard max-size guard would be defensive if scenes
  ever scale to thousands of objects.
