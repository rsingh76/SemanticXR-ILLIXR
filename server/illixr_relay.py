#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Rahul Singh, University of Illinois Urbana-Champaign <rahuls10@illinois.edu>
# SPDX-License-Identifier: Apache-2.0
"""ILLIXR relay entry point for the SemanticXR semantic-slam-server.

This REPLACES ``server/main.py``'s ``__main__`` orchestration + the gRPC
``serve()`` producer when the server runs *inside ILLIXR* via the
``semantic_python`` plugin (run with ``eval_file`` in the plugin's worker
thread).  It implements the chosen "Option 1" topology
(see SEMANTICXR_ILLIXR_INTEGRATION.md — Decision):

                       ILLIXR switchboard
                             │
        ┌────────────────────┴─────────────────────┐
        │  THIS relay (embedded interpreter)         │  lean — no compute
        │    semantic_data → frameQ                  │
        │    voice_query   → voiceQ                  │
        │    respQ → illixr_response_writer.put      │
        └──┬──────────────────────────────┬──────────┘
        frameQ                         voiceQ / respQ
           ▼                               ▼
   [inference+mapping process] ─visualizationQueue─▶ [visualization process]
                              (per-frame map stream — DIRECT, bypasses relay)

The two heavy stages stay OS **processes** (full parallelism, no GIL question).
``spawn`` is made to work under ILLIXR by pointing multiprocessing at the real
conda Python (``mp.set_executable``); see the validated notes in the doc.

Proxies injected into this script's globals by the plugin (plugin.cpp:111-118):
  illixr_semantic_reader  — .get() -> semantic_data dict | None
  illixr_voice_reader     — .get() -> voice_query dict   | None  (dedup by query_id)
  illixr_response_writer  — .put(query_id, point_clouds, colors, server_latency, text_query)

Status: SPINE — process model + relay loop wired. Ingress 9-tuple conversion
(milestone 4) and viz egress voiceQ/respQ wiring (milestone 5) are marked TODO.
"""
import os
import queue
import sys
import time
import multiprocessing as mp
from pathlib import Path

# Worker entry points — MUST be importable module-level functions (NOT defined
# in this file), so spawned children never need to re-import __main__.
from slam.services import (
    inference_pipeline_service as inference_consumer_mapping,
    vis_server,
)
from config.settings import Config, get_config, set_config
from server.main import get_parser, apply_config_overrides
from server.components.inference_service import InferenceService

# Proxies are injected into globals() by the plugin. Use globals().get so this
# file stays import-safe (e.g. if imported outside ILLIXR for linting).
_sem_reader = globals().get("illixr_semantic_reader")
_voice_reader = globals().get("illixr_voice_reader")
_resp_writer = globals().get("illixr_response_writer")

# Real conda interpreter for spawned workers (NOT main.opt.exe == sys.executable).
CONDA_PY = os.path.join(os.environ.get("PYTHONHOME", ""), "bin", "python")


def _prepare_runtime(args, config):
    """Replicate server/main.py's streaming-mode scene/output/env setup so the
    spawned workers (which inherit os.environ) write to the right dirs.

    Mirrors main.py:517-551 for the live (non --localDataset) Quest path only.
    """
    scene_name = args.sceneName if args.sceneName else args.dataset_type
    config_name = Path(args.config).stem if args.config else "default"

    os.environ["SLAM_CONFIG_NAME"] = config_name
    os.environ["SLAM_SCENE_NAME"] = scene_name
    os.environ["SLAM_DATASET_TYPE"] = args.dataset_type.lower()

    dataset_type_l = args.dataset_type.lower()
    live_root = Path(config.dataset.live_output_directory) / dataset_type_l
    live_root.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in live_root.iterdir() if d.is_dir() and d.name.startswith("run_")]
    next_n = max((int(d.split("_", 1)[1]) for d in existing if d.split("_", 1)[1].isdigit()), default=-1) + 1
    run_output_dir = live_root / f"run_{next_n}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SLAM_RUN_OUTPUT_DIR"] = str(run_output_dir.resolve())

    print(f"🎯 [relay] scene={scene_name} config={config_name} out={run_output_dir}", flush=True)
    return scene_name, run_output_dir


def run():
    args = get_parser().parse_args(sys.argv[1:])

    if args.config:
        print(f"📋 [relay] loading config: {args.config}", flush=True)
        config = Config.from_yaml(args.config)
    else:
        config = get_config()
    config = apply_config_overrides(config, args)
    set_config(config)

    scene_name, _ = _prepare_runtime(args, config)

    # Mark relay mode for spawned workers (inherited via os.environ). Workers use
    # this to skip the gRPC viz wake RPC — in relay mode the viz drain loop is
    # started directly, so the wake RPC has no server and would block 120s.
    os.environ["ILLIXR_RELAY"] = "1"

    # ---- spawn fix: real python + spawn start method (before any Process) ----
    mp.set_start_method("spawn", force=True)
    if os.path.exists(CONDA_PY):
        mp.set_executable(CONDA_PY)
        print(f"🐍 [relay] mp.set_executable -> {CONDA_PY}", flush=True)
    else:
        print(f"⚠️  [relay] conda python not found at {CONDA_PY}; spawn will use {sys.executable}", flush=True)

    # NOTE: we deliberately do NOT call install_signal_handlers() — signal.signal
    # only works on the main thread, and the relay runs on the plugin's worker
    # thread. ILLIXR --duration / process teardown drives shutdown.

    frameQ = mp.Queue(maxsize=config.server.inference_queue_size)        # relay -> inference
    visualizationQueue = mp.Queue(maxsize=config.server.visualization_queue_size)  # inference -> viz (direct)
    voiceQ = mp.Queue(maxsize=8)                                          # relay -> viz   (TODO m5)
    respQ = mp.Queue()                                                    # viz   -> relay (TODO m5)

    # inference+mapping in one process (non-pipelined), matching main.py:595-602.
    inference_proc = mp.Process(
        target=inference_consumer_mapping,
        args=(frameQ, visualizationQueue, config.model.detection.enabled, config,
              args.dataset_type, False, args.save_map, scene_name),
        name="inference_mapping",
    )
    # viz process in ILLIXR mode: voiceQ/respQ replace the gRPC query edges.
    # (vis_server runs the existing updateMap drain loop for map state + the
    #  reused getClipComparison engine for queries — see visualization_service.py.)
    vis_proc = mp.Process(
        target=vis_server,
        args=(visualizationQueue, config, args.dataset_type, args.clientUpdateMode,
              args.obejct_update_frequency, args.clientIP, args, voiceQ, respQ),
        name="visualization",
    )

    inference_proc.start()
    vis_proc.start()
    print(f"🚀 [relay] workers up: inference pid={inference_proc.pid} viz pid={vis_proc.pid}", flush=True)

    # Reuse the gRPC path's FPS gate for parity. semantic_data carries no client
    # 'fps' field, so note_client_fps stays unset and should_process_frame falls
    # back to config.frame_processing.default_client_fps (matches gRPC fallback).
    pacer = InferenceService(frameQ, config)

    last_frame = None
    frames_seen = 0
    try:
        while True:
            # ---- ingress: ship RAW semantic_data dict -> frameQ ----
            # The inference worker decodes/reprojects it (server/illixr_ingress.py).
            # Relay stays lean: dedup by frame_number, non-blocking put, drop-on-full
            # (== switchboard latest-value semantics under load).
            if _sem_reader is not None:
                f = _sem_reader.get()
                if f is not None and f.get("frame_number") != last_frame:
                    last_frame = f["frame_number"]
                    if pacer.should_process_frame(last_frame):
                        pacer.last_frame_index = last_frame  # advance gate cursor
                        frames_seen += 1
                        if frames_seen % 30 == 1:
                            print(f"📥 [relay] frame #{last_frame} (seen={frames_seen})", flush=True)
                        try:
                            frameQ.put_nowait(f)
                        except queue.Full:
                            pass  # drop stale frame

            # ---- voice query -> voiceQ (proxy dedups by query_id) ----
            if _voice_reader is not None:
                q = _voice_reader.get()
                if q is not None:
                    print(f"🎤 [relay] voice query #{q.get('query_id')}", flush=True)
                    try:
                        voiceQ.put_nowait(q)
                    except queue.Full:
                        pass

            # ---- drain responses -> ILLIXR switchboard ----
            if _resp_writer is not None:
                while True:
                    try:
                        r = respQ.get_nowait()
                    except queue.Empty:
                        break
                    _resp_writer.put(r["query_id"], r["point_clouds"], r["colors"],
                                     r["server_latency"], r["text_query"])
                    total_pts = sum(len(pc["points"]) // 3 for pc in r["point_clouds"])
                    print(f"📤 [relay] response sent #{r['query_id']} "
                          f"({len(r['point_clouds'])} clouds, {total_pts} pts) -> switchboard", flush=True)

            if not inference_proc.is_alive():
                print("❌ [relay] inference process died; exiting relay loop", flush=True)
                break

            time.sleep(0.001)
    finally:
        # Graceful shutdown == feature parity. Emitting scene_completion then
        # shutdown onto frameQ drives the SAME handlers the gRPC path uses:
        #   * inference_pipeline scene_completion -> dump_semantic_map (--save_map)
        #     + write_scene_summary_and_reset (perf), then forwards both signals
        #     to visualizationQueue (viz perf finalize + clean exit).
        # No logic is duplicated here — we just supply the trigger the gRPC
        # disconnect / Ctrl+C handlers would otherwise supply.
        print(f"🧹 [relay] graceful shutdown: scene_completion('{scene_name}') + shutdown", flush=True)
        try:
            frameQ.put({"type": "scene_completion", "scene_name": scene_name,
                        "timestamp": time.perf_counter_ns()}, timeout=5)
            frameQ.put({"type": "shutdown"}, timeout=5)
        except Exception as e:
            print(f"⚠️  [relay] could not enqueue shutdown signals: {e}", flush=True)

        # Give workers time to dump the map + finalize perf, then exit on the
        # shutdown signal. Only force-kill if they overrun the grace period.
        GRACE_S = 120
        inference_proc.join(timeout=GRACE_S)
        vis_proc.join(timeout=GRACE_S)
        for p in (inference_proc, vis_proc):
            if p.is_alive():
                print(f"⚠️  [relay] {p.name} did not exit in {GRACE_S}s; terminating", flush=True)
                p.terminate()
                p.join(timeout=10)


if __name__ == "__main__":
    run()
