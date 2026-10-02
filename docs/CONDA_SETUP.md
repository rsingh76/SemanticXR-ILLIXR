<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Conda environment setup (CUDA 12.8 / PyTorch 2.7 / Blackwell-ready)

This documents how to build a working conda environment for SemanticXR **from
scratch**, and every problem hit on the way (with the fix). It was written while
building the environment on the machine described below; the whole flow is
automated by [`scripts/setup_conda_env.sh`](../scripts/setup_conda_env.sh).

```bash
scripts/setup_conda_env.sh            # create env "semanticxr", build everything, verify
conda activate semanticxr
python scripts/verify_conda_env.py    # re-run the checks at any time
```

## Why this differs from the README

The README describes two install paths and **neither works on a current GPU**:

| Source | What it pins | Problem |
|---|---|---|
| README "Installation" (conda) | torch 2.0.1 + CUDA 11.8, gcc-10-era toolchain | torch 2.0.1 has **no kernels for Blackwell (sm_120)**; see [Issue 1](#issue-1) |
| README "Install with uv" / `scripts/setup_env.sh` / `pyproject.toml` | same torch 2.0.1+cu118, `TORCH_CUDA_ARCH_LIST=8.6+PTX` | same; also uv, not conda |
| `sceneGraphDemo_test.yml` | torch **2.3.1**, pytorch3d 0.7.7 for torch 2.0.1 | contradicts the README (2.0.1), mixes conda cu11.8 pytorch3d with pip torch cu12 |

If you are on an Ampere/Ada/Hopper GPU the original torch 2.0.1 stack can still
be made to work, but this guide uses one stack that also runs on Blackwell.

## Target machine (what this was verified on)

| | |
|---|---|
| OS | Ubuntu 22.04, kernel 6.8 (AWS) |
| GPU / driver | NVIDIA RTX PRO 6000 Blackwell (**sm_120**, 96 GB), driver 595.91 |
| CPU / RAM | 8 cores / 62 GB |
| Conda | Miniconda, conda 26.7 |
| System CUDA | 12.8 – 13.2 toolkits exist under `/usr/local`, **not used** (env is self-contained) |

## What gets built

| Component | Version | How |
|---|---|---|
| Python | 3.10 | conda-forge |
| CUDA toolchain | nvcc 12.8.93, gcc 13 (conda), cuBLAS/cuSPARSE/… dev libs | `nvidia/label/cuda-12.8.1` |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 | pip, PyTorch cu128 index |
| numpy | 1.26.4 (pinned, <2) | pip |
| pytorch3d | 0.7.9 | built from source for `sm_120` |
| chamferdist | 1.0.3 | built from source |
| GroundingDINO | repo-vendored copy at GSA `a4d76a2`, **patched** | built from source (editable) |
| segment-anything, RAM (`ram`) | GSA `a4d76a2`, recognize-anything `88c2b0c` | editable |
| gradslam | `conceptfusion` branch | built from source |
| timm | 1.0.11 | pip |
| openai-whisper, faster-whisper | latest | pip, + conda `ffmpeg` |
| PyNvVideoCodec | 2.2.3 | pip |
| semantic-slam (this repo) | 0.1.0 | `pip install -e . --no-deps` |

Everything lives inside the env except `external/` (two git clones + ~8.8 GB of
checkpoints, gitignored) and the generated gRPC stubs (gitignored).

## Files added by this setup

| File | Purpose |
|---|---|
| `scripts/setup_conda_env.sh` | idempotent one-shot builder (see header for flags) |
| `scripts/conda/requirements.txt`, `constraints.txt` | pip deps; constraints keep torch/numpy/timm from drifting |
| `scripts/patches/groundingdino-torch2.5plus.patch` | GroundingDINO CUDA-op fix, applied by the script |
| `scripts/verify_conda_env.py` | GPU/extension/model/decoder/forward-pass verification |
| `scripts/conda/environment.lock.yml`, `pip-freeze.txt` | exact package set that was verified |
| `docs/CONDA_SETUP.md` | this file |

Two tracked source files were changed: `server/video_decoders.py`
([Issue 11](#issue-11)) and `server/main.py` ([Issue 16](#issue-16)).

## Verifying the clone

Checked before building anything:

- `git fsck` clean; working tree clean; 163 tracked files; branch
  `quest/streaming-support` (tracks `origin`, with `upstream` = `rsingh76/SemanticXR-ILLIXR`).
- `.gitattributes` routes `*.mp4` / `*.h264` to Git LFS, but **no such files are
  tracked**, so a missing `git-lfs` is harmless here.
- Not in the clone by design (gitignored): `external/` (Grounded-Segment-Anything,
  recognize-anything), generated `*_pb2*.py` stubs, `datasets/`. The setup script
  creates them.
- Stale references in the docs (files that do not exist in the repo):
  `tests/test_basic_imports.py` (README step 15), `config/debug/defaults_debug.yaml`,
  `debug_parallel_*.yaml`, `parallelization_debug.yaml` (README config list),
  `scripts/install_models.sh` (comment in `pyproject.toml`). Use
  `scripts/verify_conda_env.py` instead of the missing import test.

## Using the environment

```bash
conda activate semanticxr        # sets CUDA_HOME, GSA_PATH, SEMANTIC_SLAM_AUTO_SETUP=0
python server/main.py --help
python server/main.py --dataset_type quest --localDataset --sceneName dataset_0 \
    --config config/debug/quest_debug.yaml --save_map
python -m pytest tests/test_semantic_slam_smoke.py -q                 # CPU smoke tests
REPLICA_ROOT=~/data/replica/Replica python -m pytest tests/test_replica_e2e.py -q -s
```

The activation hook (`$CONDA_PREFIX/etc/conda/activate.d/semanticxr_env.sh`)
exports three variables; set them yourself if you run outside `conda activate`:

| Variable | Value | Why |
|---|---|---|
| `CUDA_HOME` | `$CONDA_PREFIX/cuda_home` | [Issue 4](#issue-4) |
| `GSA_PATH` | `<repo>/external/Grounded-Segment-Anything` | `config/settings.py` raises if it does not exist |
| `SEMANTIC_SLAM_AUTO_SETUP` | `0` | [Issue 13](#issue-13) |

---

## Issue log

Numbering matches the `[Issue N]` comments in the scripts and requirement files.

<a id="issue-1"></a>
### Issue 1 — Blackwell GPU, but the repo pins torch 2.0.1 + CUDA 11.8

- **Symptom (expected, not run):** torch 2.0.1+cu118 ships kernels up to sm_90.
  On an sm_120 GPU every CUDA kernel fails with "no kernel image is available".
  `setup_env.sh` also hard-codes `TORCH_CUDA_ARCH_LIST=8.6+PTX`.
- **Fix:** torch **2.7.1 + cu128** (first release line with `sm_120`),
  CUDA 12.8 toolkit, `TORCH_CUDA_ARCH_LIST` auto-detected from `nvidia-smi`
  (`12.0` here). pytorch3d moves 0.7.7 → **0.7.9** (the 0.7.7 pin matches torch 2.0.1).
  `torch.cuda.get_arch_list()` now includes `sm_120`; `verify_conda_env.py` checks this.
  (pytorch3d 0.7.7 was not tried; 0.7.9 is the latest tag and built and ran cleanly.)
- **Not needed with CUDA 12.x:** the gcc-10 requirement in `setup_env.sh` is for
  CUDA 11.5/11.x only; the env's conda gcc 13 is accepted by nvcc 12.8.

<a id="issue-2"></a>
### Issue 2 — `pip install -e .` would downgrade torch

- **Symptom:** `pyproject.toml` pins `torch==2.0.1`, `torchvision==0.15.2`. The
  README's "critical final step" `pip install -e .` would replace the working
  torch with the old one.
- **Fix:** `pip install -e . --no-deps` (all dependencies are installed
  separately and pinned by `scripts/conda/constraints.txt`). `pip check` then
  reports `semantic-slam requires torch==2.0.1` — **expected and harmless**.
  `pyproject.toml` was deliberately left unchanged (it still describes the uv/cu118 flow).

<a id="issue-3"></a>
### Issue 3 — Anaconda `defaults` channel / ToS

- **Hazard (avoided, not observed):** the machine's conda (26.7) is configured
  with the `defaults` channels only, and recent conda versions require the
  Anaconda Terms of Service to be accepted interactively before using
  `repo.anaconda.com`.
- **Fix:** all conda commands use `--override-channels -c conda-forge`
  (+ `-c nvidia/label/cuda-12.8.1` for the toolkit), which needs no ToS
  acceptance. Nothing is accepted on anyone's behalf.

<a id="issue-4"></a>
### Issue 4 — conda's CUDA 12 layout is invisible to PyTorch extension builds

- **Symptom:** after `conda install cuda-nvcc …`, `$CONDA_PREFIX/include/cuda_runtime.h`
  and `$CONDA_PREFIX/lib64` do **not** exist (everything is under
  `targets/x86_64-linux/`). `torch.utils.cpp_extension` looks in
  `$CUDA_HOME/include` and `$CUDA_HOME/lib64`, so setting `CUDA_HOME=$CONDA_PREFIX`
  fails to find headers/libs when compiling pytorch3d etc.
- **Fix:** a symlink shim `$CONDA_PREFIX/cuda_home/{bin,include,lib,lib64,nvvm}`
  → the real locations, and `CUDA_HOME=$CONDA_PREFIX/cuda_home` exported by the
  env's `activate.d` hook (restored on deactivate). We install the **lean**
  set `cuda-compiler cuda-libraries-dev cuda-profiler-api cuda-nvtx` instead of
  the `cuda-toolkit` meta-package (which adds Nsight, visual tools, etc.).

<a id="issue-5"></a>
### Issue 5 — numpy 2.x sneaks in

- **Symptom:** installing torch from the cu128 index brought numpy **2.2.6**; the
  repo requires numpy < 2 (README: 1.24.3 or 1.26.4).
- **Fix:** `numpy==1.26.4` in `constraints.txt`, passed (`-c`) to every pip call.

<a id="issue-6"></a>
### Issue 6 — `opencv-python` vs `opencv-python-headless`

- **Symptom:** `supervision==0.14.0` hard-requires `opencv-python-headless`;
  the repo declares `opencv-python`. Both ship the same `cv2/` directory, so
  whichever installs last wins; removing one makes `pip check` fail.
- **Fix:** install **both at the same version** (4.10.0.84, as the original
  `sceneGraphDemo_test.yml` did) and re-install `opencv-python` last with
  `--no-deps --force-reinstall` so `cv2` is the GUI-capable build. `pip check` clean.

<a id="issue-7"></a>
### Issue 7 — GroundingDINO CUDA op does not compile on torch ≥ 2.5

- **Symptom:** building `groundingdino` fails in
  `ms_deform_attn_cuda.cu`: `error: no suitable conversion function from
  "const at::DeprecatedTypeProperties" to "c10::ScalarType" exists`.
  Cause: `AT_DISPATCH_FLOATING_TYPES(value.type(), …)`, `x.type().is_cuda()`
  and `x.data<T>()` were removed/deprecated in modern ATen.
- **Fix:** `scripts/patches/groundingdino-torch2.5plus.patch`
  (`value.type()` → `value.scalar_type()`, `.type().is_cuda()` → `.is_cuda()`,
  `.data<T>()` → `.data_ptr<T>()`; 2 files, mechanical). `external/` is a
  separate gitignored clone, so the patch is kept in this repo and applied
  (idempotently) by the setup script. Verified by running the real
  `MultiScaleDeformableAttnFunction` kernel on the GPU and a full GroundingDINO forward.
- The README/`setup_env.sh` build GroundingDINO from upstream `main`; this setup
  uses the copy vendored in the GSA checkout because `config/settings.py` reads
  the model config from `$GSA_PATH/GroundingDINO/groundingdino/config/`.

<a id="issue-8"></a>
### Issue 8 — `--no-build-isolation` needs the build backend installed by hand

- **Symptom:** `pip install --no-build-isolation -e .` →
  `Cannot import 'hatchling.build'`, then `No module named 'editables'`.
- **Fix:** `hatchling` and `editables` are in `requirements.txt` (together with
  `ninja`, `wheel`). `--no-build-isolation` is required so the CUDA extensions
  compile against the env's torch.

<a id="issue-9"></a>
### Issue 9 — README's `timm==0.4.12` breaks every MobileCLIP config

- **Symptom:** `open_clip.create_model("MobileCLIP-B")` →
  `RuntimeError: Please install the latest timm`. All
  `config/parallelization_mobileclip*.yaml` and `debug/quest_debug_mobileclip.yaml`
  use `MobileCLIP-B`. (The default `defaults.yaml`/ViT-H-14 path is unaffected.)
- **Investigation:** timm 0.9.16 → still "Unknown model vit_base_mci_224";
  **1.0.11 and 1.0.20 both work**, and RAM's old imports
  (`timm.models.registry`, `.helpers`, `.hub`, `.layers`) still resolve through
  timm's deprecation shims.
- **Fix:** `timm==1.0.11` (lowest working). Verified with the default profile
  (ViT-H-14 → 1024-d features) **and** `parallelization_mobileclip.yaml`
  (MobileCLIP-B → 512-d features). `pip check` notes `ram requires timm==0.4.12` — harmless.

<a id="issue-10"></a>
### Issue 10 — `openai-whisper` needs an `ffmpeg` binary (undocumented)

- **Symptom:** `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
  at the first transcription (loading the model and warm-up succeed). The README
  recommends this backend and `config/debug/quest_debug*.yaml` selects it.
  `faster-whisper` is unaffected.
- **Fix:** `conda install -c conda-forge ffmpeg`. Both ASR backends then run on the GPU.
- **Licensing note:** conda-forge's ffmpeg is a GPL build used only as an external
  executable; `faster-whisper` also pulls in PyAV (LGPL). The README's "no LGPL deps"
  statement refers to the default (ASR-less) install path.

<a id="issue-11"></a>
### Issue 11 — PyNvVideoCodec ≥ 2.0 API change breaks live video decode (**source fix**)

- **Symptom:** `server/video_decoders.py` assigns `bytes` to
  `PacketData.bsl_data`; with the current PyPI release (2.2.3):
  `TypeError: (): incompatible function arguments … (self, arg0: int)`.
  This would crash the first frame of any live Quest/iPad stream.
  Importing the package succeeds, so nothing flags it until decode time.
- **Investigation:** every 2.x release (2.0.0 … 2.2.3) behaves the same;
  1.0.2 has no `OutputColorType` (which the code also uses), so **no pin fixes it**.
- **Fix (3 lines):** pass a pointer to a numpy view of the bytes:
  `buf = np.frombuffer(data, np.uint8); pkt.bsl_data = buf.ctypes.data; pkt.bsl = buf.size`.
  Verified with a real H.264 stream, both as one buffer and one NAL unit at a
  time (the way the Quest client sends them): RGB `uint8`, 480×640.
  `verify_conda_env.py` includes this decode as a regression check.
  Requires the NVDEC driver library (`libnvcuvid.so`), present with the NVIDIA driver.

<a id="issue-12"></a>
### Issue 12 — `setup_env.sh` never clones Grounded-Segment-Anything or fetches most weights

- **Symptom:** `config/settings.py::PathConfig` raises `ValueError: GSA path not
  found` unless `$GSA_PATH` (default `external/Grounded-Segment-Anything`)
  exists, and the models read checkpoints from fixed paths under it.
  `setup_env.sh` only downloads the SAM weights; the clone and the other
  checkpoints are only described in README steps 10–11.
- **Fix:** the script clones GSA at `a4d76a2` and recognize-anything at `88c2b0c`
  (the commits the README names), and downloads (all idempotent):

  | File | Size | Needed for |
  |---|---|---|
  | `groundingdino_swint_ogc.pth` | 0.7 GB | detection (all configs) |
  | `ram_swin_large_14m.pth` | 5.6 GB | captioning/tagging (all detector configs) |
  | `sam_vit_h_4b8939.pth` | 2.6 GB | `sam_variant: sam`, and the no-detector path |
  | `EfficientSAM/mobile_sam.pt` | 41 MB | `sam_variant: mobilesam` (**the default**) |

  CLIP weights (ViT-H-14 ≈ 4 GB, MobileCLIP-B) are downloaded by `open_clip`
  from Hugging Face on first use into `~/.cache/huggingface`.
- `slam/models/__init__.py` imports detection, segmentation, captioning and CLIP
  eagerly, so GroundingDINO, segment-anything **and** RAM must all be installed
  even for SAM-only use.

<a id="issue-13"></a>
### Issue 13 — `semantic_slam.bootstrap` can re-run the uv installer

- **Hazard (from reading `semantic_slam/bootstrap.py`, not triggered):**
  `SemanticSLAM(...)` calls `ensure_setup()`; if any of
  `chamferdist, pytorch3d, gradslam, segment_anything, groundingdino, ram` is not
  importable it runs `scripts/setup_env.sh` → `uv sync` → a `.venv` with torch
  2.0.1+cu118, silently, on first use.
- **Fix:** the activation hook sets `SEMANTIC_SLAM_AUTO_SETUP=0`, so a broken env
  fails loudly instead. With everything installed `ensure_setup()` is a no-op.

<a id="issue-14"></a>
### Issue 14 — `ram` package metadata drags in packages that are not needed

- **Symptom:** `ram`'s `setup.py` requires `clip @ git+openai/CLIP`,
  `pycocoevalcap`, `fairscale==0.4.4`, `timm==0.4.12`.
- **Investigation:** `clip` is imported only by `ram/utils/openset_utils.py`,
  `pycocoevalcap` only by `ram/data/utils.py` (training/eval), neither on the
  inference path used by this repo.
- **Fix:** `ram` is installed `--no-deps`. `pip check` therefore prints
  `ram requires clip / pycocoevalcap / fairscale==0.4.4 / timm==0.4.12` — harmless
  (fairscale 0.4.13 imports and runs fine).

<a id="issue-15"></a>
### Issue 15 — `segment_anything` is shadowed by `$GSA_PATH` (import-order dependent)

- **Symptom:** `server/main.py` workers die with
  `ImportError: cannot import name 'sam_model_registry' from 'segment_anything' (unknown location)`.
  "unknown location" = a *namespace package*. `scripts/verify_conda_env.py` and
  `tests/test_replica_e2e.py` still passed, because they import `segment_anything`
  *before* `$GSA_PATH` is on `sys.path`.
- **Cause:** `slam/models/{detection,segmentation,captioning}.py` do
  `sys.path.append($GSA_PATH)`. GSA contains an outer `segment_anything/` directory
  *without* `__init__.py` (the real package is `segment_anything/segment_anything/`).
  A default (PEP 660) editable install registers a finder that runs **after**
  Python's normal path scan, so the outer directory wins as a namespace package.
  (The README's legacy `pip install -e segment_anything` was a `.pth` entry and
  did not have this problem.)
- **Fix:** install it as `pip install -e <GSA>/segment_anything --config-settings editable_mode=compat`
  (a plain `.pth` entry). The setup script reproduces the exact failing import
  order to decide whether it needs (re)installing, and `verify_conda_env.py`
  has a regression check for it. Only `segment_anything` collides; the other
  editable packages (`groundingdino`, `ram`) have different directory names.

<a id="issue-16"></a>
### Issue 16 — offline replay crashes: queue item has 8 fields, consumer expects 9 (**source fix**)

- **Symptom:** `python server/main.py --dataset_type {replica,scannet,quest} --localDataset …`
  prints `ValueError: not enough values to unpack (expected 9, got 8)` from
  `inference_consumer` (`slam/services/inference_pipeline.py:335`). The consumer
  process dies, the producer keeps waiting for the queue to drain, and the run
  **hangs at a low frame count** (progress bar stuck, GPU idle). This is the
  README's own headline replay command (`--dataset_type quest --localDataset`).
- **Cause:** the live gRPC path (`inference_service.py`) was extended with a 9th
  field `max_depth_m`, but the three offline replay producers in
  `server/main.py` (`_process_replica_dataset`, `_process_scannet_dataset`,
  `_process_quest_dataset`) still enqueue 8. Not an environment problem.
- **Fix:** append `None` as the 9th field in all three. `None` is the
  documented value for "client did not stamp a cap — fall back to the dataset
  YAML" (`resolve_session_max_depth`), which is the correct meaning for replay.
- **Also note:** the process exited with code 0 and printed
  "All workers completed successfully" even though a worker had crashed, so a
  non-zero exit code or the final banner is not proof that a run worked — check
  for `Traceback` in the log and for the saved map.

### Known limitations / not covered

- **TensorRT / pycuda (README step 13) are not installed.** Only
  `config/no_detection.yaml` enables `trt_sam`/`trt_clip`. That path is also
  stale in the code: `slam/models/clip.py` does `import utils.model_trt_utils`
  but the module lives at `slam/utils/model_trt_utils.py`. TensorRT 8.6.1 (the
  README's pin) predates Blackwell, so it is not expected to work on this GPU anyway.
  All other shipped configs set `trt_*: false`.
- **Live gRPC session with a real Quest/iPad client was not tested** (no client
  available). Covered instead: server CLI loads, protobuf stubs generate, NVDEC
  decodes H.264 as the Quest path does, and the model/mapping pipeline runs
  (see below).
- The conda env `sceneGraphDemo` that already existed on this machine (torch 2.14.1,
  numpy 2.2.6 — matches neither the README nor the yml) was **left untouched**; the
  new env is named `semanticxr`.
- Run-to-run, RAM's class list order changes (it is built from a Python `set`),
  so detections/class lists are not bit-reproducible between runs.

## Verification results

All run inside `conda activate semanticxr` on the machine above.

| Check | Result |
|---|---|
| `scripts/verify_conda_env.py` (10 stages: torch kernel on sm_120, pytorch3d / chamferdist / GroundingDINO CUDA ops, protobuf stubs, NVDEC H.264 decode, `segment_anything` import order, checkpoints, load of all four models, forward pass RAM→GroundingDINO→SAM→CLIP) | 10/10 pass |
| Same forward pass with `config/parallelization_mobileclip.yaml` (MobileCLIP-B) | pass (512-d features; ViT-H-14 gives 1024-d) |
| `pytest tests` — 12 CPU smoke tests + `tests/test_replica_e2e.py` (4 real Replica `room0` frames pushed through `SemanticSLAM`, text query "a chair") | 13/13 pass |
| Real server replay: `python server/main.py --dataset_type replica --localDataset --sceneName room0 --dataset_stride 100 --save_map` (default config: GroundingDINO + RAM + MobileSAM + CLIP ViT-H-14, multiprocess pipeline, mapping, map dump) | completes, 0 tracebacks; saved a semantic map with 39 objects / 28 classes at ~1 FPS (19 frames reported processed) |
| ASR: `faster-whisper` and `openai-whisper` (`small.en`, GPU) through `slam/services/asr.py` | both load, warm up and transcribe (with synthetic noise input, so the transcript text is meaningless) |
| Clean-room rebuild: `scripts/setup_conda_env.sh --env <new name>` on a fresh env, then verify | 9/9 stages passed (the segment_anything stage was added afterwards); package set identical to the hand-built env except two unpinned transitive deps (`filelock`, `fsspec`) |
| Re-run of the script on the existing env | every finished step skipped; verify passes |

Not run: a live gRPC session with a Quest/iPad client, the Quest replay path
(no captured Quest dataset on this machine — it shares the Issue 16 fix and the
NVDEC decoder verified above), the ScanNet path, `evaluation/` scripts, and
anything TensorRT.

Measured build cost on this machine: the scripted clean-room rebuild took about
6 minutes (8 cores, `sm_120` only) with the `external/` clones and checkpoints
already on disk. A first run additionally downloads ~8.8 GB of checkpoints and,
on first model use, ~4 GB of CLIP weights; compile time scales with the number of
architectures in `TORCH_CUDA_ARCH_LIST`.
