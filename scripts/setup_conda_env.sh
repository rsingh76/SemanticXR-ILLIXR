#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the SemanticXR conda environment from scratch (CUDA 12.8 / torch 2.7 /
# Blackwell-capable). Every step is idempotent: re-running skips finished work.
# Background, rationale and the full issue log: docs/CONDA_SETUP.md
#
# Usage:  scripts/setup_conda_env.sh [options]
#   --env NAME         conda env name                       (default: semanticxr, or $ENV_NAME)
#   --recreate         DELETE and rebuild the env if it already exists
#   --skip-weights     do not download model checkpoints (~8.8 GB)
#   --skip-verify      do not run scripts/verify_conda_env.py at the end
#   --with-dataset     also download the Replica dataset (~12 GB) to $REPLICA_ROOT
#
# Environment overrides:
#   TORCH_CUDA_ARCH_LIST   default: detected from nvidia-smi (e.g. "12.0")
#   MAX_JOBS               parallel compile jobs (default: min(nproc, 8))
#   CONDA_EXE              path to conda (default: found on PATH or ~/miniconda3)
#   REPLICA_ROOT           default: $HOME/data/replica/Replica
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$REPO_ROOT/scripts/conda"
ENV_NAME="${ENV_NAME:-semanticxr}"
RECREATE=0; SKIP_WEIGHTS=0; SKIP_VERIFY=0; WITH_DATASET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift ;;
    --recreate) RECREATE=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    --with-dataset) WITH_DATASET=1 ;;
    -h|--help) sed -n 2,22p "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Preflight
# ---------------------------------------------------------------------------
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
[[ -z "$CONDA" && -x "$HOME/miniconda3/bin/conda" ]] && CONDA="$HOME/miniconda3/bin/conda"
[[ -n "$CONDA" ]] || die "conda not found. Install Miniconda first (https://docs.conda.io/en/latest/miniconda.html)."
CONDA_BASE="$("$CONDA" info --base)"
for t in git curl; do command -v "$t" >/dev/null || die "$t is required"; done

# Compute capability -> TORCH_CUDA_ARCH_LIST (e.g. RTX PRO 6000 Blackwell = 12.0).
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd';')"
  fi
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0}"
fi
export TORCH_CUDA_ARCH_LIST
export FORCE_CUDA=1
NPROC="$(nproc 2>/dev/null || echo 4)"
export MAX_JOBS="${MAX_JOBS:-$(( NPROC < 8 ? NPROC : 8 ))}"
log "env=$ENV_NAME  conda=$CONDA  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS"

GSA_PATH="${GSA_PATH:-$REPO_ROOT/external/Grounded-Segment-Anything}"
RAM_PATH="$REPO_ROOT/external/recognize-anything"

# conda-forge only: the `defaults` channel requires accepting Anaconda's ToS
# interactively on conda >= 25 (Issue 3). nvidia/label/cuda-12.8.1 for the toolkit.
CF=(--override-channels -c conda-forge)
CUDA_CH=(--override-channels -c nvidia/label/cuda-12.8.1 -c conda-forge)

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "$RECREATE" -eq 1 ]]; then
    warn "--recreate: removing existing env '$ENV_NAME'"
    "$CONDA" env remove -y -n "$ENV_NAME" >/dev/null
  else
    log "env '$ENV_NAME' exists; reusing (use --recreate to rebuild)."
  fi
fi
if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "creating env '$ENV_NAME' (python 3.10)"
  "$CONDA" create -y -n "$ENV_NAME" "${CF[@]}" python=3.10 pip >/dev/null
fi

# ---------------------------------------------------------------------------
# 2. CUDA 12.8 toolchain + ffmpeg inside the env (nvcc, headers, cuBLAS... , gcc 13)
# ---------------------------------------------------------------------------
PREFIX="$("$CONDA" run -n "$ENV_NAME" printenv CONDA_PREFIX)"
if [[ ! -x "$PREFIX/bin/nvcc" ]]; then
  log "installing CUDA 12.8 compiler + dev libraries into the env"
  "$CONDA" install -y -n "$ENV_NAME" "${CUDA_CH[@]}" "cuda-version=12.8" \
      cuda-compiler cuda-libraries-dev cuda-profiler-api cuda-nvtx >/dev/null
fi
if [[ ! -x "$PREFIX/bin/ffmpeg" ]]; then
  log "installing ffmpeg (needed by openai-whisper) [Issue 10]"
  "$CONDA" install -y -n "$ENV_NAME" "${CF[@]}" ffmpeg >/dev/null
fi

# ---------------------------------------------------------------------------
# 3. CUDA_HOME shim + activation hooks [Issues 4, 12]
#    conda's CUDA 12 packages use targets/x86_64-linux/{include,lib}; torch's
#    cpp_extension wants $CUDA_HOME/{include,lib64}.
# ---------------------------------------------------------------------------
mkdir -p "$PREFIX/cuda_home" "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
( cd "$PREFIX/cuda_home" \
  && ln -sfn ../bin bin \
  && ln -sfn ../targets/x86_64-linux/include include \
  && ln -sfn ../targets/x86_64-linux/lib lib \
  && ln -sfn ../targets/x86_64-linux/lib lib64 \
  && ln -sfn ../nvvm nvvm )
cat > "$PREFIX/etc/conda/activate.d/semanticxr_env.sh" <<EOF
#!/bin/bash
# Generated by scripts/setup_conda_env.sh -- SemanticXR env activation hook.
export _SXR_OLD_CUDA_HOME="\${CUDA_HOME:-}"
export CUDA_HOME="\$CONDA_PREFIX/cuda_home"
# Never let semantic_slam.bootstrap run the uv-based scripts/setup_env.sh (it
# would try to install torch 2.0.1+cu118). [Issue 13]
export SEMANTIC_SLAM_AUTO_SETUP=0
if [ -z "\${GSA_PATH:-}" ] && [ -d "$GSA_PATH" ]; then
    export GSA_PATH="$GSA_PATH"
    export _SXR_SET_GSA_PATH=1
fi
EOF
cat > "$PREFIX/etc/conda/deactivate.d/semanticxr_env.sh" <<'EOF'
#!/bin/bash
export CUDA_HOME="${_SXR_OLD_CUDA_HOME:-}"
[ -z "$CUDA_HOME" ] && unset CUDA_HOME
unset _SXR_OLD_CUDA_HOME SEMANTIC_SLAM_AUTO_SETUP
if [ "${_SXR_SET_GSA_PATH:-}" = "1" ]; then unset GSA_PATH _SXR_SET_GSA_PATH; fi
EOF
# Older revisions of this script / manual builds used this name; drop it so the
# two hooks don't both run.
rm -f "$PREFIX/etc/conda/activate.d/semanticxr_cuda.sh" "$PREFIX/etc/conda/deactivate.d/semanticxr_cuda.sh"

# Activate (conda's own hooks reference unset vars, so relax -u around them).
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
export GSA_PATH
PY="$PREFIX/bin/python"
PIP=("$PY" -m pip)
PIPC=("$PY" -m pip install -c "$HERE/constraints.txt")
[[ "${CUDA_HOME:-}" == "$PREFIX/cuda_home" ]] || die "CUDA_HOME not set by activation hook (got '${CUDA_HOME:-}')"
log "CUDA_HOME=$CUDA_HOME  nvcc: $(nvcc --version | tail -1)"

have() { "$PY" -c "import $1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 4. PyTorch (cu128 wheels; the first with sm_120/Blackwell kernels) [Issue 1]
# ---------------------------------------------------------------------------
if ! "$PY" -c "import torch; assert torch.__version__.startswith('2.7.1') and torch.version.cuda=='12.8'" 2>/dev/null; then
  log "installing torch 2.7.1 + torchvision 0.22.1 (cu128)"
  "${PIPC[@]}" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple
fi

# ---------------------------------------------------------------------------
# 5. Pure-Python dependencies, numpy held at 1.26.4 [Issue 5]
# ---------------------------------------------------------------------------
log "installing python dependencies"
"${PIPC[@]}" -r "$HERE/requirements.txt"
# Issue 6: make cv2 come from the GUI wheel (headless is only there for supervision's metadata).
"${PIPC[@]}" --no-deps --force-reinstall opencv-python==4.10.0.84 >/dev/null

# ---------------------------------------------------------------------------
# 6. External repos (gitignored under external/) [Issue 12]
# ---------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/external"
if [[ ! -d "$GSA_PATH/.git" ]]; then
  log "cloning Grounded-Segment-Anything @ a4d76a2"
  git clone -q https://github.com/IDEA-Research/Grounded-Segment-Anything.git "$GSA_PATH"
  git -C "$GSA_PATH" checkout -q a4d76a2
fi
if [[ ! -d "$RAM_PATH/.git" ]]; then
  log "cloning recognize-anything @ 88c2b0c"
  git clone -q https://github.com/xinyu1205/recognize-anything.git "$RAM_PATH"
  git -C "$RAM_PATH" checkout -q 88c2b0c
fi

# Issue 7: GroundingDINO's CUDA op uses Tensor::type()/data<T>(), which no longer
# compile against torch >= 2.5. Patch lives in the repo; apply once.
PATCH="$REPO_ROOT/scripts/patches/groundingdino-torch2.5plus.patch"
if git -C "$GSA_PATH" apply --check --reverse "$PATCH" >/dev/null 2>&1; then
  log "GroundingDINO torch>=2.5 patch already applied."
else
  log "applying GroundingDINO torch>=2.5 patch"
  git -C "$GSA_PATH" apply "$PATCH"
  # source changed -> force a rebuild of the extension if it was built before
  "${PIP[@]}" uninstall -y groundingdino >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------
# 7. Source-built CUDA extensions + git packages (all --no-deps / --no-build-isolation)
# ---------------------------------------------------------------------------
build() { # module  description  pip-args...
  local mod="$1" what="$2"; shift 2
  if have "$mod"; then log "$what already installed."; return; fi
  log "building $what (sm: $TORCH_CUDA_ARCH_LIST) -- this can take several minutes"
  "${PIPC[@]}" --no-build-isolation --no-deps "$@"
}
build pytorch3d  "pytorch3d 0.7.9"   "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
build chamferdist "chamferdist"      chamferdist
# Issue 15: segment_anything must be a *compat-mode* editable install. The slam/
# models append $GSA_PATH to sys.path, where the outer (init-less)
# segment_anything/ dir would shadow a PEP 660 editable package as a namespace
# package. The check below reproduces that exact import order.
if ! "$PY" -c "import sys; sys.path.append('$GSA_PATH'); from segment_anything import sam_model_registry" >/dev/null 2>&1; then
  log "installing segment-anything (editable, compat mode) [Issue 15]"
  "${PIPC[@]}" --no-build-isolation --no-deps --force-reinstall \
      -e "$GSA_PATH/segment_anything" --config-settings editable_mode=compat
else
  log "segment-anything already installed."
fi
build groundingdino "GroundingDINO"  -e "$GSA_PATH/GroundingDINO"
# Issue 14: ram's setup.py declares clip/pycocoevalcap/fairscale==0.4.4 which
# are not used on the inference path -> --no-deps.
build ram "recognize-anything (ram)" -e "$RAM_PATH"
build gradslam "gradslam (conceptfusion)" "gradslam @ git+https://github.com/gradslam/gradslam.git@conceptfusion"

# Project itself. --no-deps: pyproject.toml pins torch==2.0.1 / torchvision==0.15.2
# and would downgrade torch. [Issue 2]
if ! "${PIP[@]}" show semantic-slam >/dev/null 2>&1; then
  log "installing semantic-slam (editable, --no-deps)"
  ( cd "$REPO_ROOT" && "${PIPC[@]}" --no-deps --no-build-isolation -e . )
fi

# ---------------------------------------------------------------------------
# 8. gRPC stubs (generated, gitignored)
# ---------------------------------------------------------------------------
if [[ ! -f "$REPO_ROOT/slam/protocols/vis_proto/vis_pb2.py" ]]; then
  log "generating vis_proto stubs"
  ( cd "$REPO_ROOT/slam/protocols/vis_proto" && "$PY" -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. vis.proto )
fi
if [[ ! -f "$REPO_ROOT/server/xr_service_pb2.py" ]]; then
  log "generating xr_service stubs"
  ( cd "$REPO_ROOT/server" && "$PY" -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. xr_service.proto )
fi

# ---------------------------------------------------------------------------
# 9. Model checkpoints into $GSA_PATH (paths hard-wired in config/settings.py)
# ---------------------------------------------------------------------------
dl() { # url dest
  if [[ -s "$2" ]]; then log "present: ${2#"$GSA_PATH"/}"; return; fi
  mkdir -p "$(dirname "$2")"
  log "downloading ${2#"$GSA_PATH"/}"
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}
if [[ "$SKIP_WEIGHTS" -eq 0 ]]; then
  dl https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth "$GSA_PATH/groundingdino_swint_ogc.pth"
  dl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth                                     "$GSA_PATH/sam_vit_h_4b8939.pth"
  dl https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth  "$GSA_PATH/ram_swin_large_14m.pth"
  dl https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt                              "$GSA_PATH/EfficientSAM/mobile_sam.pt"
  # CLIP (ViT-H-14 / MobileCLIP-B) weights are fetched from Hugging Face by open_clip on first use.
fi

# ---------------------------------------------------------------------------
# Optional: Replica dataset for tests/test_replica_e2e.py
# ---------------------------------------------------------------------------
if [[ "$WITH_DATASET" -eq 1 ]]; then
  REPLICA_ROOT="${REPLICA_ROOT:-$HOME/data/replica/Replica}"
  if [[ ! -d "$REPLICA_ROOT/room0" ]]; then
    mkdir -p "$(dirname "$REPLICA_ROOT")"
    log "downloading Replica (~12 GB) -> $(dirname "$REPLICA_ROOT")"
    curl -fL --retry 3 -C - -o "$(dirname "$REPLICA_ROOT")/Replica.zip" https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
    ( cd "$(dirname "$REPLICA_ROOT")" && unzip -q -o Replica.zip )
  else
    log "Replica present at $REPLICA_ROOT"
  fi
fi

# ---------------------------------------------------------------------------
# 10. Verify
# ---------------------------------------------------------------------------
if [[ "$SKIP_VERIFY" -eq 0 ]]; then
  log "verifying (scripts/verify_conda_env.py)"
  ( cd "$REPO_ROOT" && "$PY" scripts/verify_conda_env.py $([[ "$SKIP_WEIGHTS" -eq 1 ]] && echo --skip-models) )
fi

log "done.  Use it with:  conda activate $ENV_NAME"
