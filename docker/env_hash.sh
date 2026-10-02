#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Print the fingerprint of everything that defines the *environment* image
# (dependencies, patches, Dockerfile) -- but not the SemanticXR code, which is
# mounted from the checkout. Used by ./sxr (image tags), the Dockerfile (baked
# into the image) and docker/entrypoint.sh (mismatch check), so all three agree.
#
#   docker/env_hash.sh [ROOT]     ROOT = repo checkout (default: .)
set -euo pipefail
cd "${1:-.}"
export LC_ALL=C
{
  for f in Dockerfile scripts/conda/requirements.txt scripts/conda/constraints.txt \
           $(ls scripts/patches/*.patch | sort); do
    printf '== %s\n' "$f"
    cat "$f"
  done
} | sha256sum | cut -c1-12
