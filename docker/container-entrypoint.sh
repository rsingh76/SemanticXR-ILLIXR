#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Baked into the image as /usr/local/bin/sxr-entrypoint. Deliberately dumb: all
# startup logic lives in docker/entrypoint.sh of the SemanticXR checkout mounted
# at /workspace, so it is versioned with the code (git pull updates it, no rebuild).
if [[ -f /workspace/docker/entrypoint.sh ]]; then
  exec bash /workspace/docker/entrypoint.sh "$@"
fi
case "${1:-quest}" in
  quest|replay|server|verify|fetch-models)
    echo "[semanticxr] no SemanticXR checkout at /workspace." >&2
    echo "[semanticxr] Run via 'docker compose' from the checkout (it mounts '.:/workspace')," >&2
    echo "[semanticxr] or add -v /path/to/SemanticXR:/workspace to docker run." >&2
    exit 1 ;;
  *)
    exec "$@" ;;
esac
