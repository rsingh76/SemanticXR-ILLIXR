# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SemanticXR images: CUDA 12.8 / PyTorch 2.7.1 (Blackwell-capable). Same stack
# as scripts/setup_conda_env.sh. Normally driven by ./sxr; see docs/DOCKER.md.
#
# This builds the ENVIRONMENT image (target `env`): Python, CUDA extensions,
# Grounded-SAM, RAM. The SemanticXR code is NOT inside: docker compose mounts
# your git checkout at /workspace, so code changes never need a rebuild.
# (docker/Dockerfile.release adds a frozen copy of the code on top, for
# deployments without a checkout.)
#
#   ./sxr build      (= docker build --target env -t semanticxr-env:<fingerprint> .)
#
# Model checkpoints are never baked in: they live on the /models volume and are
# downloaded on first start (docker/fetch_models.sh).

ARG CUDA_VERSION=12.8.1
ARG UBUNTU=ubuntu22.04

###############################################################################
# builder: venv + CUDA extensions (needs nvcc -> devel image)
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv
RUN pip install --upgrade pip setuptools wheel

WORKDIR /opt/build
COPY scripts/conda/requirements.txt scripts/conda/constraints.txt scripts/conda/

# PyTorch first (cu128 wheels: first line with sm_120 kernels), then everything else.
RUN pip install -c scripts/conda/constraints.txt \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
RUN pip install -c scripts/conda/constraints.txt -r scripts/conda/requirements.txt \
    # supervision needs opencv-python-headless metadata; cv2 must come from the GUI wheel (CONDA_SETUP Issue 6)
    && pip install -c scripts/conda/constraints.txt --no-deps --force-reinstall opencv-python==4.10.0.84

# External repos at the commits the README names, plus the GroundingDINO
# torch>=2.5 fix (CONDA_SETUP Issue 7). They live in /opt/external, outside
# /workspace, so mounting a checkout there cannot hide them.
COPY scripts/patches/ scripts/patches/
RUN git clone -q https://github.com/IDEA-Research/Grounded-Segment-Anything.git /opt/external/Grounded-Segment-Anything \
    && git -C /opt/external/Grounded-Segment-Anything checkout -q a4d76a2 \
    && git -C /opt/external/Grounded-Segment-Anything apply /opt/build/scripts/patches/groundingdino-torch2.5plus.patch \
    && git clone -q https://github.com/xinyu1205/recognize-anything.git /opt/external/recognize-anything \
    && git -C /opt/external/recognize-anything checkout -q 88c2b0c

# GPU architectures to compile the CUDA extensions for (pytorch3d, chamferdist,
# GroundingDINO). Default covers Ampere (8.0/8.6), Ada (8.9), Hopper (9.0) and
# Blackwell (10.0 datacenter, 12.0 RTX/workstation). Each extra arch adds build
# time. Declared HERE, not at the top of the stage, so that changing it only
# rebuilds the extensions and reuses the cached torch/pip layers above (DOCKER.md D7).
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
ARG MAX_JOBS=8
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS}

# Source-built CUDA extensions + git packages. All --no-deps / --no-build-isolation
# so they build against the torch above (CONDA_SETUP Issues 7, 8, 14, 15).
ARG PIPC="pip install -c scripts/conda/constraints.txt --no-build-isolation --no-deps"
RUN ${PIPC} "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
RUN ${PIPC} chamferdist
# No GPU is visible during `docker build`, so GroundingDINO's setup.py silently
# builds *no* CUDA op unless told it is in Docker (DOCKER.md D1).
RUN AM_I_DOCKER=true BUILD_WITH_CUDA=true ${PIPC} -e /opt/external/Grounded-Segment-Anything/GroundingDINO
RUN ${PIPC} -e /opt/external/Grounded-Segment-Anything/segment_anything --config-settings editable_mode=compat \
    && ${PIPC} -e /opt/external/recognize-anything \
    && ${PIPC} "gradslam @ git+https://github.com/gradslam/gradslam.git@conceptfusion"

# Fail the build (instead of the first inference) if any CUDA extension came out
# without GPU code for every requested architecture.
RUN set -e; \
    sos="$(python -c 'import importlib.util as u; print(" ".join(u.find_spec(m).origin for m in ("pytorch3d._C", "chamferdist._C", "groundingdino._C")))')"; \
    for so in $sos; do \
      archs="$(cuobjdump --list-elf "$so" 2>/dev/null | grep -o 'sm_[0-9]*a\?' | sort -u | tr '\n' ' ')"; \
      echo "$(basename "$so"): ${archs:-NO GPU CODE}"; \
      for a in $(echo "$TORCH_CUDA_ARCH_LIST" | tr ';' ' '); do \
        a="sm_$(echo "${a%%+*}" | tr -d .)"; \
        echo "$archs" | grep -qw "$a" || { echo "ERROR: $so lacks $a"; exit 1; }; \
      done; \
    done

# Drop git metadata / build trees we don't need at runtime (keeps the editable
# sources + the in-place GroundingDINO _C.so).
RUN rm -rf /opt/external/*/.git /opt/external/Grounded-Segment-Anything/GroundingDINO/build \
    && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} +

###############################################################################
# env: runtime environment (no compiler; CUDA libs come from the torch wheels)
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-base-${UBUNTU} AS env

ENV DEBIAN_FRONTEND=noninteractive
# libgl1/libglib2.0-0/libsm6/libxext6/libxrender1: opencv-python (GUI build) + open3d
# libegl1: open3d's core library links libEGL.so.1 (DOCKER.md D2)
# libgomp1: open3d/faiss; ffmpeg: openai-whisper (CONDA_SETUP Issue 10) and the NVDEC self-test
# git: lets `git` work inside the container on the mounted checkout (optional convenience)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 libpython3.10 ca-certificates curl git \
        libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/external /opt/external

ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    # NVDEC (PyNvVideoCodec) needs the driver's libnvcuvid, which the NVIDIA
    # container runtime only mounts when the "video" capability is requested (D3).
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    GSA_PATH=/opt/external/Grounded-Segment-Anything \
    # Never let semantic_slam.bootstrap run the uv installer (CONDA_SETUP Issue 13).
    SEMANTIC_SLAM_AUTO_SETUP=0 \
    # Every downloaded model lands on the /models volume.
    MODELS_DIR=/models \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache \
    # writable config dir for matplotlib when running as an arbitrary UID
    MPLCONFIGDIR=/tmp/matplotlib

# Checkpoint paths are hard-wired under $GSA_PATH (config/settings.py): point
# them at the /models volume, which the entrypoint fills.
RUN G=/opt/external/Grounded-Segment-Anything \
    && ln -sfn /models/gsa/groundingdino_swint_ogc.pth $G/groundingdino_swint_ogc.pth \
    && ln -sfn /models/gsa/sam_vit_h_4b8939.pth        $G/sam_vit_h_4b8939.pth \
    && ln -sfn /models/gsa/ram_swin_large_14m.pth      $G/ram_swin_large_14m.pth \
    && ln -sfn /models/gsa/mobile_sam.pt               $G/EfficientSAM/mobile_sam.pt \
    # the mounted checkout is owned by the host user, not by root
    && git config --system --add safe.directory /workspace

COPY docker/container-entrypoint.sh /usr/local/bin/sxr-entrypoint

# Environment fingerprint (docker/env_hash.sh over the same files): the
# entrypoint compares it with the mounted checkout and refuses to start when a
# git pull changed the dependencies but the image was not rebuilt (D8).
COPY Dockerfile /opt/sxr-env/src/
COPY scripts/conda/requirements.txt scripts/conda/constraints.txt /opt/sxr-env/src/scripts/conda/
COPY scripts/patches/ /opt/sxr-env/src/scripts/patches/
COPY docker/env_hash.sh /opt/sxr-env/
RUN chmod +x /usr/local/bin/sxr-entrypoint \
    && bash /opt/sxr-env/env_hash.sh /opt/sxr-env/src > /opt/sxr-env/hash \
    && echo "environment fingerprint: $(cat /opt/sxr-env/hash)"

WORKDIR /workspace
VOLUME ["/models"]
# 50051: frame stream (XrService)   50054: queries / visualization (VisualizerServer)
EXPOSE 50051 50054
ENTRYPOINT ["/usr/local/bin/sxr-entrypoint"]
CMD ["quest"]
