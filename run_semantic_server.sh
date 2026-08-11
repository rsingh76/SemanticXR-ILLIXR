#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Rahul Singh, University of Illinois Urbana-Champaign <rahuls10@illinois.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Launch this SemanticXR semantic-slam-server INSIDE ILLIXR via the
# `semantic_python` plugin, using the `sceneGraphDemo` conda environment.
#
# This is the in-repo copy of the launcher: it lives in the project root and
# locates the ILLIXR build (build-scene/) relative to where this repo is
# vendored (…/ILLIXR/plugins/semantic_python/<this repo>). Override ILLIXR_ROOT
# or BUILD_DIR if your layout differs.
#
# Entry script is server/illixr_relay.py — the lean relay that bridges the
# ILLIXR switchboard to the SLAM worker processes (see
# SEMANTICXR_ILLIXR_INTEGRATION.md). The standalone gRPC path (server/main.py)
# is unaffected.
#
# Usage:
#   ./run_semantic_server.sh                 # run with defaults below
#   DURATION=60 ./run_semantic_server.sh     # override any setting inline
#   BUILD_DIR=/path/to/build ./run_semantic_server.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Settings (override any of these by exporting them before running the script)
# ---------------------------------------------------------------------------
# This script's directory == the SemanticXR project root (this repo).
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# ILLIXR repo root is three levels up: …/ILLIXR/plugins/semantic_python/<this repo>.
ILLIXR_ROOT="${ILLIXR_ROOT:-$(cd "$PROJECT_ROOT/../../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ILLIXR_ROOT/build-scene}"
# Default to the currently-activated conda env ($CONDA_PREFIX). Either
# `conda activate <env>` before running, or set CONDA_ENV explicitly.
CONDA_ENV="${CONDA_ENV:-${CONDA_PREFIX:-}}"
PY_VER="${PY_VER:-3.10}"

PY_SCRIPT="${SEMANTIC_PYTHON_SCRIPT:-$PROJECT_ROOT/server/illixr_relay.py}"
# config path is intentionally RELATIVE — we cd into PROJECT_ROOT below so it resolves.
PY_ARGS="${SEMANTIC_PYTHON_ARGS:-dataset_type=quest,config=config/debug/quest_debug.yaml,save_map}"

# Default to this host's primary IP (the client connects to it). Override
# ILLIXR_TCP_SERVER_IP if autodetect picks the wrong interface.
SERVER_IP="${ILLIXR_TCP_SERVER_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
SERVER_PORT="${ILLIXR_TCP_SERVER_PORT:-50057}"
DURATION="${DURATION:-1000}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
[[ -x "$BUILD_DIR/main.opt.exe" ]]    || { echo "ERROR: $BUILD_DIR/main.opt.exe not found — build ILLIXR first, or set BUILD_DIR."; exit 1; }
[[ -f "$PY_SCRIPT" ]]                 || { echo "ERROR: python script not found: $PY_SCRIPT"; exit 1; }
[[ -n "$CONDA_ENV" && -d "$CONDA_ENV" ]] || { echo "ERROR: conda env not found — 'conda activate <env>' or set CONDA_ENV (got: '${CONDA_ENV}')."; exit 1; }
[[ -d "$PROJECT_ROOT" ]]             || { echo "ERROR: project root not found: $PROJECT_ROOT"; exit 1; }
[[ -n "$SERVER_IP" ]]                || { echo "ERROR: could not determine host IP — set ILLIXR_TCP_SERVER_IP."; exit 1; }

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Must be real shell env vars (read by ld.so / pybind before plugin construction):
export PYTHONHOME="$CONDA_ENV"
# PROJECT_ROOT goes FIRST so Python resolves `server`/`slam`/`config` to THIS clone
# rather than the OLD copy that `pip install -e .` registered in the conda env.
# (Python checks PYTHONPATH top-to-bottom and stops at the first match; the editable
#  install's finder sits below it and is never reached.)
export PYTHONPATH="$PROJECT_ROOT:$CONDA_ENV/lib/python$PY_VER/site-packages"
export VIRTUAL_ENV="$CONDA_ENV"
# Put the conda env's bin FIRST on PATH so whisper's ASR spawns CONDA's ffmpeg,
# not the system /usr/bin/ffmpeg. Our LD_LIBRARY_PATH below forces conda's
# libavutil (56.51) onto every child process; the system ffmpeg binary is built
# against the newer 56.70 and crashes with
#   undefined symbol: av_opt_child_class_iterate, version LIBAVUTIL_56
# Using conda's ffmpeg keeps binary+lib consistent.
export PATH="$CONDA_ENV/bin:${PATH}"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$BUILD_DIR/plugins/semantic_python:$BUILD_DIR/plugins/tcp_network_backend:${LD_LIBRARY_PATH:-}"

# REQUIRED: ILLIXR dlopens plugins with RTLD_LOCAL, so libpython's symbols are not in the
# global scope. Python C-extension modules (e.g. _ctypes) then fail with
# "undefined symbol: PyUnicode_FromFormat". Preloading libpython makes its symbols global
# at process start, which fixes all such extension imports.
export LD_PRELOAD="$CONDA_ENV/lib/libpython$PY_VER.so.1.0:${LD_PRELOAD:-}"

# torch needs CUDA_HOME (README setup step 4). Point it at the env.
export CUDA_HOME="${CUDA_HOME:-$CONDA_ENV}"

# ILLIXR / plugin config:
export SEMANTIC_PYTHON_SCRIPT="$PY_SCRIPT"
export SEMANTIC_PYTHON_ARGS="$PY_ARGS"
export ILLIXR_TCP_SERVER_IP="$SERVER_IP"
export ILLIXR_TCP_SERVER_PORT="$SERVER_PORT"
export ILLIXR_IS_CLIENT=0        # 0 = server
export ILLIXR_DISPLAY_MODE=none

echo "Launching semantic_python with the SemanticXR SLAM server (relay)"
echo "  illixr root: $ILLIXR_ROOT"
echo "  build dir  : $BUILD_DIR"
echo "  python env : $CONDA_ENV (py $PY_VER)"
echo "  script     : $PY_SCRIPT"
echo "  args       : $PY_ARGS"
echo "  cwd        : $PROJECT_ROOT"
echo

# Run from the project root so relative paths (config/, ./datasets, ./output,
# yolov8l-world.pt, run_output_dir) resolve as they do in standalone runs.
# (main.opt.exe is referenced by absolute path; plugins resolve via LD_LIBRARY_PATH.)
cd "$PROJECT_ROOT"
exec "$BUILD_DIR/main.opt.exe" --plugins=tcp_network_backend,semantic_python --duration="$DURATION"
