# SemanticXR → ILLIXR Integration (as built)

Runs the **SemanticXR semantic-slam-server** *inside ILLIXR* via the `semantic_python` plugin: it
receives RGB-D frames and voice queries from the ILLIXR switchboard and returns query responses over
the switchboard (transmitted to the headset by `tcp_network_backend`) — instead of talking gRPC
directly to the client.

**Status:** working end-to-end — frames flow into the SLAM pipeline, the map builds, and voice
queries return point clouds over the switchboard. See **Known issues** for open items.

The original gRPC path (`server/main.py`, standalone) is **untouched and still works**; the ILLIXR
path is an additive entry point selected by the launcher.

---

## How to run

```bash
# from the ILLIXR repo root
./run_semantic_server.sh                  # DURATION=… to override the default
```

The launcher (`ILLIXR/run_semantic_server.sh`) sets the conda env, `PYTHONHOME`/`PYTHONPATH`
(clone first — see below), `LD_PRELOAD=libpython`, `CUDA_HOME`, `PATH` (conda bin first — see
ffmpeg note), then runs `main.opt.exe --plugins=tcp_network_backend,semantic_python` with
`SEMANTIC_PYTHON_SCRIPT=server/illixr_relay.py` and
`SEMANTIC_PYTHON_ARGS=dataset_type=quest,config=config/debug/quest_debug.yaml,save_map`.

> The server **blocks until the client connects** (`tcp_network_backend`'s constructor waits on
> `accept()`), and only then does `semantic_python` start, spawn the workers, and load models. There
> is currently no server-side-only smoke test (the plugin needs the network backend, which needs the
> peer). See SEMANTIC_PYTHON_BUILD.md §4.

**PYTHONPATH note:** the `sceneGraphDemo` env has an editable (`pip install -e`) install of this
package pointing at a *different* checkout. The launcher puts this clone's root first on `PYTHONPATH`
so `server`/`slam`/`config` resolve here, not to the editable install.

---

## Architecture (as built)

```
                    ILLIXR switchboard
                          │
        ┌─────────────────┴──────────────────┐
        │  relay  (embedded interpreter)       │   lean — no compute
        │   semantic_data → frameQ             │
        │   voice_query   → voiceQ             │
        │   respQ → illixr_response_writer.put │
        └──┬───────────────────────────┬───────┘
        frameQ                      voiceQ / respQ
           ▼                            ▼
   [inference+mapping process] ─visualizationQueue─▶ [visualization process]
                              (per-frame map stream — DIRECT, bypasses relay)
```

- **relay** (`server/illixr_relay.py`, runs in the plugin's embedded interpreter): holds the three
  switchboard proxies; multiplexes them onto `mp.Queue`s and back. No compute, single non-blocking
  loop, no threads.
- **inference+mapping** and **visualization** are OS **processes** (`spawn`), preserving the
  original pipeline's parallelism and data path.
- The per-frame map stream (`visualizationQueue`, inference→viz) stays process→process and never
  passes through the relay.

### Proxies injected by the plugin (`plugin.cpp`, schemas in `switchboard_bindings.hpp`)

| Proxy global | Topic | API |
|---|---|---|
| `illixr_semantic_reader` | `semantic_data` | `.get() -> dict \| None` |
| `illixr_voice_reader` | `semantic_query` | `.get() -> dict \| None` (dedups by `query_id`; carries full `pcm_data`) |
| `illixr_response_writer` | `semantic_response` (**network** topic) | `.put(query_id, point_clouds, colors, server_latency, text_query)` |

The proxies are only valid in the embedded interpreter's process — not picklable, never used from a
spawned child. That is why all switchboard I/O funnels through the relay.

---

## Files

**New (this integration):**
- `server/illixr_relay.py` — the relay entry point (spawn fixes, worker orchestration, relay loop,
  graceful-shutdown trigger, FPS gate).
- `server/illixr_ingress.py` — `IllixrFrameConverter`: `semantic_data` dict → inference 9-tuple,
  reusing the gRPC path's decoders.

**Modified (dual-mode — gRPC path unchanged when the new params/env are absent):**
- `slam/services/visualization_service.py` — `vis_server(..., voiceQ=None, respQ=None)`: ILLIXR mode
  reuses `updateMap` (map state) + `getClipComparison` (query) with no gRPC. Added query point-count
  logging.
- `slam/services/inference_pipeline.py` — skips the gRPC viz "wake" RPC when `ILLIXR_RELAY=1`
  (relay starts the drain loop directly).
- `slam/services/mapping_server.py` — same wake-RPC skip (pipelined mode).

**Not committed:** generated proto stubs (`*_pb2*.py`, gitignored — regenerate with
`grpc_tools.protoc`), runtime output (`live_output/`, `logs_performance/`, `*.wav`).

---

## Data contracts

### Ingress — semantic_data dict → 9-tuple (`server/illixr_ingress.py`)

Mirrors the live Quest gRPC path (`server/components/grpc_server.py` is_quest), reusing
`VideoProcessor` (H.265), `_decode_quest_depth_bytes` (R16_SFloat), and
`slam/datasets/quest.py:build_depth_in_rgb_frame` so the result is identical-by-construction. The
9-tuple matches `server/components/inference_service.py:129-139`:

```
(image_pil, image_array, depth_array, [frame]+16_rowmajor_c2w, frame, ts_ns, ts_ns, {}, max_depth_m)
```

- RGB H.265 → decode → resize to QUEST.yaml processing res (960×960, LANCZOS).
- depth R16 → metric float32 → `build_depth_in_rgb_frame` (aligned into RGB camera, 960×960).
- pose: `[frame_number] + rgb_camera_pose.flatten()` (row-major c2w, **unflipped** — matches gRPC;
  the OpenGL→OpenCV flip lives downstream in `quest.py:load_poses`, not used on this path).
- `max_depth_m`: 0.0/unset → `None` (per-dataset default applies downstream).

Conversion runs in the **inference worker** (not the relay): keeps the relay lean and avoids shipping
the ~6 MB decoded arrays over `mp.Queue`. The worker accepts either a raw dict (relay) or a 9-tuple
(gRPC) — `inference_pipeline.py` converts dicts that contain `'image'`.

### Egress — query_response (`_answer_illixr_voice_query` → `respQ` → relay → writer)

Reuses `asr.transcribe_chunks` + `getClipComparison`. `point_clouds` are the **voxel-downsampled**
object clouds (downsampled in `updateMap` at receive; `getClipComparison` returns those). Per cloud:
`points` = flat XYZ, `centroid` = real `points.mean` (the gRPC `createGRPCResponse` hardcoded
`[0,0,0]`; the client uses the centroid, so we compute it). `server_latency` in seconds.

Wire path (verified in C++): `illixr_response_writer.put` → `query_response` (full points) →
`network_writer::put` boost-serializes the whole event → `tcp_network_backend::topic_send` →
`send_to_peer` over the client socket. Topic: **`semantic_response`**. Nothing truncates the cloud.

---

## Key implementation decisions

- **`spawn` under the embedded interpreter.** `sys.executable` is `main.opt.exe`, not Python, so
  `spawn` re-execs the wrong binary. Fix: `mp.set_executable(<conda python>)`. `Process.start()` also
  works from the plugin's worker thread. `__main__.__file__` is **not** needed: `spawn` only
  re-imports `__main__` when it has a `__file__` (it doesn't under `eval_file`), so as long as every
  `Process` target lives in an importable module (`slam.services`), children never touch `__main__`.
- **Worker-side ingress conversion** (above) keeps the relay a thread-free, compute-free loop.
- **Egress reuse:** `updateMap` runs verbatim in a thread via a fake gRPC context; queries reuse
  `getClipComparison`. Only the I/O edges change (queues instead of gRPC).
- **Lifecycle parity:** on teardown the relay emits `scene_completion` then `shutdown` onto `frameQ`,
  which drives the *existing* worker handlers — `dump_semantic_map` (`--save_map`),
  `write_scene_summary_and_reset` (perf), and viz finalize. No logic duplicated.
- **FPS pacing:** relay reuses `InferenceService.should_process_frame` (no-op at current config where
  `target_fps ≥ client_fps`, but correct when it isn't).

---

## Bring-up findings & fixes

- **Viz "wake" RPC blocked startup 120s.** The inference process pinged viz's gRPC server at
  `localhost:50054` (which doesn't exist in ILLIXR mode) with a 120 s connect timeout, stalling the
  mapping loop and dropping frames meanwhile. Fixed: skip the wake RPC when `ILLIXR_RELAY=1` (the
  relay starts viz's drain loop directly).
- **Voice queries crashed on ffmpeg.** Whisper shells out to the `ffmpeg` binary; the launcher's
  `LD_LIBRARY_PATH` (conda libs, needed for embedding) forced the *system* ffmpeg to load conda's
  older `libavutil` → `undefined symbol: av_opt_child_class_iterate`. Fixed by putting conda's `bin`
  first on `PATH` so conda's ffmpeg (matching libavutil) is used.
- **Wire format confirmed.** Real frames are H.265 RGB (~266 KB) + R16 depth (320×320×2 = 204800 B),
  exactly what `IllixrFrameConverter` expects.
- **Downsampling + full-cloud transmission confirmed** (see Egress).

---

## Known issues / follow-ups

- **Client renders one point per object (under investigation).** The server transmits the full,
  downsampled cloud (verified through serialization → socket). The `📦 [VISUALIZATION]` /
  `📤 [relay]` logs report the point counts leaving the server; if those are large, the cause is
  client-side rendering/parsing (e.g. it draws only the per-object `centroid`), not ILLIXR transport.
- **Mid-stream H.265 join.** Because the pipeline starts after model load, the decoder can miss the
  initial keyframe/SPS-PPS (`PPS id out of range`) until the next IDR. Failed decodes are dropped
  gracefully. A readiness handshake (warm the pipeline before the client streams) would remove the
  warm-up frame loss; see next item.
- **No readiness handshake / warm-up gap.** `tcp_network_backend`'s constructor blocks on `accept()`
  before `semantic_python` is even constructed, so model load (1–3 min) happens *after* the client
  connects — initial frames stream into a latest-value topic and are dropped until the pipeline is
  ready. A clean fix needs a C++ change (lazy network writer + start `semantic_python` first, or a
  ready signal to the client) and an ILLIXR rebuild.
- **Teardown.** The relay emits `scene_completion`/`shutdown` in its `finally`; whether that runs at
  ILLIXR `--duration` teardown (vs the process being killed) still needs confirmation. The viz
  process's `pdeathsig` covers perf finalize but not the map dump.
- **ASR durability.** The ffmpeg `PATH` fix works but is environment-fragile; the robust fix is to
  feed Whisper a numpy PCM array directly (resample in numpy, no ffmpeg subprocess) — an `asr.py`
  change, not yet done.
