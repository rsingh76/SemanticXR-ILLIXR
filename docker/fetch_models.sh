#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download every model the server needs onto the /models volume. Idempotent:
# files already present are skipped, so it is cheap to run on every start.
#
#   fetch_models.sh [--config YAML] [--no-prefetch]
#
#   1. GSA checkpoints (fixed paths, see config/settings.py)  -> $MODELS_DIR/gsa/
#   2. Hugging Face / whisper models used by the given config -> $HF_HOME, $XDG_CACHE_HOME
#      (CLIP, bert-base-uncased, ASR). Skipped with --no-prefetch; they would
#      otherwise be downloaded lazily on first use, into the same volume.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
CONFIG="config/debug/quest_debug.yaml"
PREFETCH=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift ;;
    --no-prefetch) PREFETCH=0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

log() { printf '\033[1;32m[fetch]\033[0m %s\n' "$*"; }

dl() { # url dest
  if [[ -s "$2" ]]; then log "present: $2"; return; fi
  mkdir -p "$(dirname "$2")"
  log "downloading $(basename "$2")"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$2.part" "$1"
  mv "$2.part" "$2"
}

dl https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth "$MODELS_DIR/gsa/groundingdino_swint_ogc.pth"
dl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth                                     "$MODELS_DIR/gsa/sam_vit_h_4b8939.pth"
dl https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth  "$MODELS_DIR/gsa/ram_swin_large_14m.pth"
dl https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt                              "$MODELS_DIR/gsa/mobile_sam.pt"

if [[ "$PREFETCH" -eq 1 ]]; then
  log "prefetching Hugging Face / ASR models used by $CONFIG"
  python "$(dirname "$0")/prefetch_models.py" --config "$CONFIG"
fi
log "models ready in $MODELS_DIR"
