<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running SemanticXR in Docker

Docker replaces the conda setup ([CONDA_SETUP.md](CONDA_SETUP.md)) with the same
verified stack (CUDA 12.8, PyTorch 2.7.1, patched source-built extensions), so
a new machine needs no environment build at all.

The design separates what changes rarely from what changes daily:

| What | Changes | Lives in |
|---|---|---|
| Python, CUDA extensions, Grounded-SAM / RAM code (the *environment*) | when dependencies change | the **image** `semanticxr-env:<fingerprint>` |
| SemanticXR code (any branch / commit / local edit) | daily (`git pull`, edits) | **your git checkout**, mounted into the container at `/workspace` |
| Model weights (~14 GB) | almost never | a **models folder** on the host (default `~/.cache/semanticxr/models`) |
| Run outputs (`live_output/`, `datasets/`, `output/`) | every run | **your checkout**, exactly where a native run writes them |

So code changes never need a rebuild. Only dependency changes do, and the
container detects those and says so.

## New machine: one command

```bash
git clone https://github.com/ILLIXR/SemanticXR && cd SemanticXR
./sxr setup                                  # or: ./sxr setup --registry <host/namespace>
docker compose up                            # live Quest server on ports 50051 + 50054
```

`./sxr setup`:

1. **Checks the host**: Docker + Compose v2, Docker permissions, NVIDIA driver
   R570+, NVIDIA Container Toolkit, free disk. It explains what to install
   if something is missing; it never uses sudo itself.
2. **Writes `.env`** (per machine, gitignored): models folder, and the registry if given.
3. **Gets the environment image** for this checkout: already local, else pulled
   from the registry (if configured), else **built locally** (~30 min once).
4. **Downloads the model weights** into the models folder (~14 GB, once per machine).
5. **Runs the GPU self-test** (CUDA ops, video decode, all models, a real forward pass).

## Day to day

```bash
git pull                     # or: git checkout <branch>, or edit files
docker compose up            # runs exactly the code in this checkout
```

That's all. Configs, code and compose settings are read from the checkout at
start. If a pull changed the *dependencies* (`Dockerfile`, `scripts/conda/*.txt`,
`scripts/patches/`), the container refuses to start with:

```
This checkout needs environment image <new>, but the running image is <old> ...
Fix: run './sxr update' on the host ...
```

`./sxr update` then pulls the matching image from the registry, or builds it.
Switching back and forth between branches with different dependencies doesn't
rebuild: every image is kept under its fingerprint tag.

## `./sxr` commands

| Command | Does |
|---|---|
| `./sxr setup [--registry R] [--models-dir D] [--arch A] [--skip-verify]` | one-time machine setup (above); safe to re-run |
| `./sxr update` | make `semanticxr-env:local` (what compose runs) match this checkout: local tag, else registry pull, else local build |
| `./sxr build [--arch A] [--no-cache]` | build the environment image locally. `--arch local` = only this machine's GPU (faster) |
| `./sxr push` | publish this checkout's environment image to the registry (see below) |
| `./sxr release [--version V] [--push]` | image with the code baked in, for sites without a checkout |
| `./sxr status` | checkout, fingerprint, which image compose uses, registry, models folder |
| `./sxr up / down / logs / verify / shell` | shortcuts for the docker compose commands |

## Using a registry (optional, recommended for several machines)

Without a registry, every machine builds its own environment image once (~30 min).
With one, only the first machine builds and everyone else pulls.

```bash
# once per machine
docker login <registry host>
./sxr setup --registry <registry host>/<namespace>     # saved in .env, not in the repo

# whenever a commit changes dependencies (on one machine, after pushing the commit)
./sxr build && ./sxr push        # -> <registry>/<namespace>/semanticxr-env:<fingerprint>
```

Images are tagged by the **fingerprint** of the dependency files, so any
checkout knows exactly which image it needs. `./sxr push` refuses two mistakes:

- **A reduced-architecture image** (`--arch local`), which other machines would
  pull and then fail on their GPU.
- **Dependency files with uncommitted changes**: no other checkout could ever
  match that fingerprint.

The registry address is per-machine configuration (`.env`, or `SXR_REGISTRY=...`),
never committed, so private registries stay out of the public repo.

## Settings

Set in `.env` (written by `./sxr setup`, editable) or per command (`SXR_CONFIG=... docker compose up`).
`docker-compose.yml` itself is commented line by line.

| Variable | Default | Meaning |
|---|---|---|
| `SXR_CONFIG` | `config/debug/quest_debug.yaml` | YAML profile for `quest` / `replay` (also decides which CLIP / ASR models are prefetched) |
| `SXR_MODELS_DIR` | `~/.cache/semanticxr/models` (set by setup) | model weights + caches, shared by all checkouts on the machine |
| `SXR_REGISTRY` | none | registry namespace for `./sxr update/push/release` |
| `SXR_ARCH` | all 6 | GPU archs for local builds (`local` = this GPU) |
| `SXR_STREAM_PORT` / `SXR_QUERY_PORT` | `50051` / `50054` | host ports (the headset must use these) |
| `SXR_SKIP_FETCH` | `0` | `1` = never download models (offline site with a pre-filled models folder) |
| `SXR_PREFETCH` | `1` | `0` = only the 4 checkpoints at start; CLIP/BERT/ASR download lazily on first use |
| `SXR_IGNORE_ENV_MISMATCH` | `0` | `1` = start despite a dependency mismatch (e.g. a comment-only change) |
| `SXR_UID` / `SXR_GID` | owner of the checkout | user the server runs as |
| `OPENAI_API_KEY` | – | only for `asr.backend: openai-api` |

## What runs: modes

`command:` in `docker-compose.yml`, or after `docker compose run --rm semanticxr`:

| Mode | Does |
|---|---|
| `quest [args]` (default) | live Quest server with `$SXR_CONFIG` + `--save_map`; extra args are appended |
| `replay <scene> [args]` | replay `datasets/quest/<scene>` (a capture made with `dataset.enabled: true`) |
| `server <args>` | `python server/main.py <args>`, anything it accepts |
| `verify` | GPU self-test (`scripts/verify_conda_env.py`) |
| `fetch-models` | download models, then exit |
| anything else (`bash`, `python …`) | run as-is, without the preparation steps |

Before running a mode, the entrypoint (`docker/entrypoint.sh`, from your
checkout) switches to the checkout owner's user, checks the image fingerprint,
generates the gRPC stubs if missing/outdated, checks the GPU and NVDEC, and
fetches missing models (~10 s when nothing is missing).

## Host requirements

| | |
|---|---|
| GPU | NVIDIA with compute capability 8.0, 8.6, 8.9, 9.0, 10.0 or 12.0 (A100, RTX 30xx/A-series, RTX 40xx/L40, H100, B200, RTX 50xx/RTX PRO Blackwell) |
| Driver | **R570 or newer** (CUDA 12.8) |
| Software | Docker, Docker Compose v2, **NVIDIA Container Toolkit** |
| Disk | ~16 GB image + ~14 GB models (+ ~30 GB build cache when building locally) |
| Network | inbound TCP **50051** (frame stream) and **50054** (voice/text queries) reachable from the headset |

## Other deployment options

- **Frozen release** (no checkout at the site): `./sxr release --version X [--push]`
  builds `semanticxr:X` = environment + code. Run it with
  `docker run --gpus all --shm-size=16g --init -p 50051:50051 -p 50054:50054 -v <models>:/models -v <data>:/data semanticxr:X`;
  outputs go to `<data>`.
- **No registry, no build at the site:** `docker save semanticxr-env:local | gzip > env.tar.gz`,
  copy it, then `docker load < env.tar.gz`, `docker tag <loaded image> semanticxr-env:local`.
- **Offline models:** copy a filled models folder and set `SXR_SKIP_FETCH=1`
  (plus `HF_HUB_OFFLINE=1` if the machine has no internet at all).
- **Native (conda) and Docker on the same checkout** work side by side: both
  write outputs to the same folders, and the generated gRPC stubs are
  compatible (same protobuf version).

---

## Issues found while building the image

Numbering continues separately from [CONDA_SETUP.md](CONDA_SETUP.md) (Issues 1–16),
all of which the Dockerfile also handles.

<a id="issue-d1"></a>
### D1: GroundingDINO silently builds without its CUDA op inside `docker build`

- **Symptom:** the image builds fine; `verify` then fails at
  "groundingdino MultiScaleDeformableAttention CUDA op", and real inference would
  fail at the first detection. The give-away was the build step taking 3 s instead of ~45 s.
- **Cause:** no GPU is visible during `docker build`, and GroundingDINO's
  `setup.py` only compiles the extension if `torch.cuda.is_available()` **or**
  `AM_I_DOCKER` and `BUILD_WITH_CUDA` are set; otherwise it returns no extension and
  prints nothing alarming (upstream GSA issues #53/#84).
- **Fix:** `AM_I_DOCKER=true BUILD_WITH_CUDA=true` on that build step, plus a
  build-time guard that inspects `pytorch3d._C`, `chamferdist._C` and
  `groundingdino._C` with `cuobjdump --list-elf` and fails the build unless each
  contains code for **every** arch in `TORCH_CUDA_ARCH_LIST`.

<a id="issue-d2"></a>
### D2: open3d needs `libEGL.so.1`, which the slim runtime image lacks

- **Symptom:** `ImportError: libEGL.so.1: cannot open shared object file`
  on `import open3d` (so on `import slam`).
- **Investigation:** ran `ldd` over every `.so` in the venv inside the image.
  Apart from wheel-vendored libraries (resolved via RPATH), the only missing system
  libraries were `libEGL.so.1` (open3d core: needed), `libtbb.so.12` (numba's
  optional TBB pool and open3d's TensorFlow ops: unused), and `libibverbs`/`librdmacm`
  (cuFile RDMA transport: unused).
- **Fix:** `libegl1` added to the runtime image.

<a id="issue-d3"></a>
### D3: NVDEC needs the `video` driver capability (prevented)

- The NVIDIA container runtime only mounts the driver's `libnvcuvid` (needed by
  PyNvVideoCodec for the Quest H.264/H.265 stream) when the container asks for the
  `video` capability. On this test host it happened to be mounted anyway, but that
  depends on the host's runtime configuration.
- **Fix:** the image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`; the
  entrypoint warns if `libnvcuvid` is still missing; `verify` decodes a real H.264 stream.

<a id="issue-d4"></a>
### D4: root-owned files on the host, and the code writes into its own source tree

- Containers run as root by default, so every model download and every run
  output in the bind mounts would be root-owned on the host (can't delete
  without sudo). Running as a normal user instead hits two places where the
  code writes into the (root-owned) source tree: ASR's scratch audio
  `slam/services/output_audio.wav` (found by reading `visualization_service.py`)
  and the frame scratch dir `./main_server/temp_output_dir`.
- **Fix:** the entrypoint switches to the owner of the mounted checkout (release
  image: of `/data`) with `setpriv`, so everything it writes on the host is owned
  by that user, exactly like a native run. With a mounted checkout the scratch
  files simply land in the checkout (gitignored); in a release image, where the
  code is root-owned, both scratch paths are symlinked to `/tmp`. matplotlib gets
  a writable `MPLCONFIGDIR`.

<a id="issue-d5"></a>
### D5: shared memory and stop timeout (prevented)

- The server moves frames between processes with `multiprocessing` queues;
  Docker's default `/dev/shm` is 64 MB. Compose sets `shm_size: 16gb` (and the
  plain `docker run` examples use `--shm-size=16g`). A run with the default was not tried.
- `docker stop` sends SIGKILL after 10 s by default. A live session shut down
  cleanly in 5.8 s in testing, but `--save_map` dumps the map on shutdown, which takes
  longer for big scenes, so compose sets `stop_grace_period: 60s`. `init: true` reaps
  the worker processes.

<a id="issue-d6"></a>
### D6: large host folders would be sent as build context

- `docker build` uploads the whole repo folder as context. Model folders inside
  the checkout (the old `./docker-volumes/` default, or a host `external/` with
  ~9 GB of checkpoints) would be uploaded on every build, and a host `external/`
  could shadow the image's patched, compiled copy. Both, plus run outputs,
  `.env` and `internal/`, are excluded in `.dockerignore`.

<a id="issue-d7"></a>
### D7: changing the GPU arch list rebuilt everything

- **Symptom:** a build with a different `TORCH_CUDA_ARCH_LIST` re-ran apt, the
  PyTorch download and every pip install, not just the CUDA compiles.
- **Cause:** the arch list was an `ENV` at the top of the builder stage, so it was
  part of every layer's cache key.
- **Fix:** it is declared right before the extension builds. Base layers are now
  shared between arch choices; a site that rebuilds for its own GPU only recompiles
  the extensions.

<a id="issue-d8"></a>
### D8: a `git pull` can silently outgrow the image (design)

- With the code mounted from the checkout, a commit that changes dependencies
  would otherwise run on an image built for the old ones, failing later with
  confusing import or ABI errors.
- **Fix:** `docker/env_hash.sh` fingerprints `Dockerfile`, `scripts/conda/*.txt` and
  `scripts/patches/`. The image stores the fingerprint it was built from; the
  entrypoint compares it with the checkout and exits with instructions on a
  mismatch (override: `SXR_IGNORE_ENV_MISMATCH=1`). Images are tagged by
  fingerprint, so registry lookups and branch switches are exact.

<a id="issue-d9"></a>
### D9: plain commands (`bash`, `python`) ran as root inside the container

- **Symptom (found in testing):** `docker compose run --rm semanticxr bash`
  ran as uid 0, so any file it created in the mounted checkout (even a `.pyc`)
  was root-owned on the host.
- **Cause:** only the server modes went through the user switch; other commands
  were exec'd directly by the image's entrypoint.
- **Fix:** the baked-in entrypoint is now a pure pass-through to
  `docker/entrypoint.sh` in the checkout, which switches user first and then runs
  any command. Verified: a file touched from `bash` in the container is owned by
  the host user, and no root-owned files remain in the checkout.

<a id="issue-d10"></a>
### D10: `docker compose up` before setup created root-owned folders

- **Symptom (seen on the dev machine):** with no `.env`, compose fell back to
  `./docker-volumes/...`. Docker created those missing folders **as root**, the
  entrypoint therefore stayed root, and 14 GB of models plus a live session were
  written root-owned into the checkout.
- **Fix:** the models mount has no default any more:
  `${SXR_MODELS_DIR:?run ./sxr setup first ...}`. Compose refuses to start with
  that message until `./sxr setup` has created the folder (as the user) and
  written it to `.env`. Outputs no longer need a separate folder: they go to the
  checkout.

<a id="issue-d11"></a>
### D11: pushing the multi-GB environment layer can time out

- **Symptom (found in testing):** `docker push` failed with
  `net/http: timeout awaiting response headers` on the largest layer (the Python
  environment) while the registry committed it. `docker push` does not retry by itself.
- **Fix:** `./sxr push` / `./sxr release --push` retry up to 3 times (layers that
  already arrived are skipped), and the pull in `./sxr update` retries network
  errors the same way. "Not in registry" falls back to a local build immediately;
  authentication errors stop at once with the `docker login <host>` hint.

## Verification (on the RTX PRO 6000 Blackwell host)

Current layout (environment image + mounted checkout):

| Test | Result |
|---|---|
| `./sxr build` (6 archs) | success; every extension contains sm_80/86/89/90/100/120 (build guard); 1238 s with base layers cached; image 15.7 GB; fingerprint baked into the image = fingerprint computed on the host |
| `./sxr setup` on an existing checkout (image + models present) | .env written, 10/10 self-test checks pass |
| Raw command in the container (`bash`): user, code location | runs as the checkout owner (uid 1000); `server/main.py` is the mounted `/workspace` copy; `git` sees the checkout's branch; created files are user-owned (after the D9 fix) |
| Dependency change (comment added to `scripts/conda/constraints.txt`) | `./sxr status` shows MISMATCH; container refuses to start (exit 3) with the fix instructions; `SXR_IGNORE_ENV_MISMATCH=1` starts it; revert -> matches again |
| `docker compose up` without `.env` | refuses: "run ./sxr setup first" (D10) |
| Replica server replay via `docker compose run ... server ...` | 0 tracebacks, 19 frames, 40-object map written to the checkout's `output/replica/room0/`, user-owned |
| Live Quest via `docker compose up -d` | both servers up; host gRPC to 50051 + 50054 READY; real `UploadSyncMessage_quest` accepted; session dir `live_output/quest/run_2` in the checkout (continuing after a native run's `run_1`); `down` in 6 s |
| Native conda env on the same checkout afterwards | 12/12 CPU smoke tests; container-generated gRPC stubs import natively |
| `./sxr push` guards | refused a 12.0-only image; refused with uncommitted dependency files |
| Registry, new machine (local `registry:2` as stand-in; fresh checkout copy; **no** local env images; empty models dir) | first attempt: push timed out (D11) -> setup fell back to a local build, downloaded 14 GB of models, 10/10 pass, no root-owned files. After the retry fix: push OK; fresh `./sxr setup --registry ...` **pulled** the image, fingerprint matched, 10/10 pass |
| `./sxr release` on top of the environment image + `verify` in the release image | built in ~1 s (no environment rebuild); runs as the owner of `/data`; checks pass |

Not tested: a real headset through the container, a second physical machine /
different GPU, a real private registry with authentication (the auth-error path
of `./sxr` is untested), air-gapped operation, and a truly cold build on a
machine with no Docker build cache (~30 min expected for 6 archs).
