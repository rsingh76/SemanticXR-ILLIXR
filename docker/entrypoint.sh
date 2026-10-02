#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint, run from the SemanticXR checkout mounted at /workspace
# (or from the frozen copy in a release image). Prepares the models volume,
# fetches missing models, then runs one of:
#
#   quest [extra server args]          live Quest server (default)
#   replay <scene> [extra args]        offline replay of datasets/quest/<scene>
#   server <server/main.py args>       anything server/main.py accepts
#   verify [verify_conda_env.py args]  GPU / extension / model self-test
#   fetch-models                       only download models, then exit
#   <anything else>                    run as-is (bash, python, ...), as the same user
#
# Environment:
#   SXR_CONFIG        YAML profile for quest/replay  (default config/debug/quest_debug.yaml)
#   SXR_SKIP_FETCH=1  don't download anything at start (offline sites with a pre-filled /models)
#   SXR_PREFETCH=0    only the 4 GSA checkpoints; CLIP/BERT/ASR are fetched lazily on first use
#   SXR_UID/SXR_GID   user to run as (default: owner of the checkout / of /data)
#   SXR_IGNORE_ENV_MISMATCH=1  start even if the image was built for other dependencies
set -euo pipefail
cd /workspace

CONFIG="${SXR_CONFIG:-config/debug/quest_debug.yaml}"
RELEASE=0; [[ -f /workspace/.sxr-release ]] && RELEASE=1
log()  { printf '\033[1;36m[semanticxr]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[semanticxr]\033[0m %s\n' "$*" >&2; }

cmd="${1:-quest}"
[[ $# -gt 0 ]] && shift

# --- user ---------------------------------------------------------------------
# Run as the owner of the mounted checkout (release image: of /data), so every
# file written on the host -- outputs, model downloads, generated stubs -- is
# owned by the host user, exactly as with a native run. Root only if that
# directory is root-owned.
if [[ "$(id -u)" == "0" && -z "${_SXR_DROPPED:-}" ]]; then
  if [[ "$RELEASE" == "1" ]]; then owner_dir=/data; mkdir -p /data; else owner_dir=/workspace; fi
  uid="${SXR_UID:-$(stat -c %u "$owner_dir")}"
  gid="${SXR_GID:-$(stat -c %g "$owner_dir")}"
  if [[ "$uid" != "0" ]]; then
    dirs=("$MODELS_DIR" "$MODELS_DIR/gsa" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME")
    [[ "$RELEASE" == "1" ]] && dirs+=(/data /data/datasets /data/live_output /data/output)
    for d in "${dirs[@]}"; do
      mkdir -p "$d"
      # only the directories themselves (cheap); content we create later is ours anyway
      [[ "$(stat -c %u "$d")" == "$uid" ]] || chown "$uid:$gid" "$d"
    done
    log "running as uid=$uid gid=$gid (owner of $owner_dir)"
    export _SXR_DROPPED=1 HOME=/tmp
    exec setpriv --reuid="$uid" --regid="$gid" --clear-groups -- bash "$0" "$cmd" "$@"
  fi
fi
mkdir -p "$MODELS_DIR/gsa" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
if [[ "$RELEASE" == "1" ]]; then
  mkdir -p /data/datasets /data/live_output /data/output /tmp/main_server/temp_output_dir
fi

# Plain commands (bash, python, git, ...) run as-is -- but as the same user as
# the server, so nothing they create in the checkout is root-owned.
case "$cmd" in
  quest|replay|server|verify|fetch-models) ;;
  *) exec "$cmd" "$@" ;;
esac

# --- image vs checkout --------------------------------------------------------
if [[ "$RELEASE" == "0" ]]; then
  want="$(bash docker/env_hash.sh /workspace)"
  have="$(cat /opt/sxr-env/hash 2>/dev/null || echo unknown)"
  if [[ "$want" != "$have" ]]; then
    warn "This checkout needs environment image $want, but the running image is $have:"
    warn "dependencies (Dockerfile, scripts/conda/*.txt or scripts/patches/) changed since it was built."
    warn "Fix: run './sxr update' on the host (pulls or builds the matching image), then start again."
    if [[ "${SXR_IGNORE_ENV_MISMATCH:-0}" != "1" ]]; then
      warn "(To start anyway, e.g. for a comment-only change: SXR_IGNORE_ENV_MISMATCH=1)"
      exit 3
    fi
    warn "SXR_IGNORE_ENV_MISMATCH=1: starting anyway."
  fi
  # gRPC stubs are generated (gitignored): create/refresh them in the checkout
  # when missing or older than their .proto. Same protobuf version as the conda
  # env, so native runs can use them too.
  gen() { # dir proto stub
    if [[ ! -f "$1/$3" || "$1/$2" -nt "$1/$3" ]]; then
      log "generating gRPC stubs for $1/$2"
      ( cd "$1" && python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. "$2" )
    fi
  }
  gen slam/protocols/vis_proto vis.proto vis_pb2.py
  gen server xr_service.proto xr_service_pb2.py
fi

# --- GPU sanity -------------------------------------------------------------
if [[ "$cmd" != "fetch-models" ]]; then
  if ! nvidia-smi -L >/dev/null 2>&1; then
    warn "no GPU visible in the container. Run with --gpus all (docker run) or 'gpus: all' (compose),"
    warn "and make sure the NVIDIA Container Toolkit is installed on the host."
    exit 1
  fi
  log "GPU: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1)"
  if ! ls /usr/lib/x86_64-linux-gnu/libnvcuvid.so* >/dev/null 2>&1 && ! ldconfig -p | grep -q libnvcuvid; then
    warn "libnvcuvid (NVDEC) not mounted: live video decode will fail."
    warn "Run with -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video (the image sets this; something overrode it)."
  fi
fi

# --- models -------------------------------------------------------------------
if [[ "${SXR_SKIP_FETCH:-0}" != "1" ]]; then
  fetch_args=(--config "$CONFIG")
  [[ "${SXR_PREFETCH:-1}" == "0" ]] && fetch_args+=(--no-prefetch)
  bash docker/fetch_models.sh "${fetch_args[@]}"
else
  log "SXR_SKIP_FETCH=1: not downloading models"
fi

# --- run ----------------------------------------------------------------------
case "$cmd" in
  fetch-models)
    exit 0 ;;
  verify)
    exec python scripts/verify_conda_env.py "$@" ;;
  quest)
    log "live Quest server, config $CONFIG  (ports 50051 stream, 50054 queries)"
    exec python server/main.py --dataset_type quest --config "$CONFIG" --save_map "$@" ;;
  replay)
    [[ $# -ge 1 ]] || { warn "usage: replay <scene, e.g. dataset_0> [extra args]"; exit 2; }
    scene="$1"; shift
    [[ -d "datasets/quest/$scene" ]] || { warn "no capture at datasets/quest/$scene"; exit 2; }
    log "replaying datasets/quest/$scene with $CONFIG"
    exec python server/main.py --dataset_type quest --localDataset --sceneName "$scene" \
         --config "$CONFIG" --save_map "$@" ;;
  server)
    exec python server/main.py "$@" ;;
  *)
    warn "unknown mode '$cmd' (expected quest, replay, server, verify, fetch-models)"; exit 2 ;;
esac
