#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Rahul Singh, University of Illinois Urbana-Champaign <rahuls10@illinois.edu>
# SPDX-License-Identifier: Apache-2.0
"""ILLIXR ingress: convert a switchboard ``semantic_data`` dict to the inference
9-tuple, reusing the EXACT decoders of the live Quest gRPC path so the result is
identical-by-construction to ``grpc_server._process_frame_request`` (is_quest).

This runs inside the **inference worker process** (not the relay), so:
  * the relay stays lean — it just ships the raw dict over the mp.Queue;
  * the ~6 MB decoded RGB+depth arrays are never pickled across the boundary;
  * the H.265 decoder (which is inter-frame / stateful) lives in one place and
    persists across frames for the whole session.

ASSUMPTION (verify at bring-up): ILLIXR's ``semantic_data.image`` is H.265/HEVC
and ``.depth`` is raw R16_SFloat — the same wire format the Quest client sends
over gRPC. The struct (include/illixr/data_format/semantics.hpp) stores opaque
byte vectors; the producer is the upstream Quest client/bridge. If the format
differs, decode fails loudly here (None return → frame dropped) rather than
silently corrupting the map.

semantic_data dict schema (plugins/semantic_python/switchboard_bindings.hpp:91-103):
  image, depth            numpy uint8 (flat)  — encoded bytes
  frame_number            int
  image_width/height      int                 — native RGB resolution
  depth_width/height      int                 — native depth resolution
  depth_near_z            float               — ZBufferParams.x (== -2*sensor_near)
  intrinsics              numpy float32 [fx,fy,cx,cy]   (RGB)
  depth_intrinsics        numpy float32 [fx,fy,cx,cy]
  rgb_camera_pose         numpy float32 (4,4) row-major c2w (OpenXR, unflipped)
  depth_pose              numpy float32 (4,4) row-major c2w
  max_depth_m             float               — 0.0/unset => use YAML default
"""
import os
import time

import numpy as np
import yaml
from PIL import Image

from server.components.video_processor import VideoProcessor
from server.components.grpc_server import _decode_quest_depth_bytes
from slam.datasets.quest import build_depth_in_rgb_frame


def _quest_target_resolution():
    """(width, height) processing resolution from QUEST.yaml camera_params —
    mirrors grpc_server._quest_target_resolution()."""
    quest_yaml = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # project root
        "slam", "datasets", "QUEST.yaml",
    )
    with open(quest_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    cp = cfg["camera_params"]
    return int(cp["image_width"]), int(cp["image_height"])


class IllixrFrameConverter:
    """Stateful (persistent H.265 decoder) converter — one per worker session."""

    def __init__(self, config=None):
        self._vp = VideoProcessor(config)
        self._target_w, self._target_h = _quest_target_resolution()
        print(f"🔁 [ingress] IllixrFrameConverter ready (target {self._target_w}x{self._target_h})", flush=True)

    def convert(self, f):
        """semantic_data dict -> inference 9-tuple, or None to drop the frame.

        Mirrors grpc_server._process_frame_request is_quest path:
          - H.265 decode (grpc_server.py:314-329)
          - R16 depth decode (grpc_server.py:343-346)
          - meta + build_depth_in_rgb_frame + LANCZOS resize (grpc_server.py:401-423)
          - pose = [frame] + 16 row-major c2w, unflipped (grpc_server.py:366-372)
          - 9-tuple layout (inference_service.py:129-139)
        """
        # ---- RGB: H.265 -> PIL (native res) ----
        video_data = bytes(f["image"])
        if not video_data:
            return None
        processed_frame, is_valid = self._vp.process_video_frame(video_data, codec="h265")
        if not is_valid or processed_frame is None:
            return None
        if isinstance(processed_frame, Image.Image):
            image_pil = processed_frame
        else:  # np.ndarray (rgb24)
            image_pil = Image.fromarray(processed_frame)

        # ---- depth: R16_SFloat -> metric float32 (native res) ----
        depth_native = _decode_quest_depth_bytes(
            bytes(f["depth"]), int(f["depth_width"]), int(f["depth_height"]),
            float(f["depth_near_z"]),
        )
        if depth_native is None:
            return None

        # ---- meta (scalar split intrinsics + 4x4 poses, unflipped) ----
        intr = np.asarray(f["intrinsics"], dtype=np.float32).reshape(-1)
        dintr = np.asarray(f["depth_intrinsics"], dtype=np.float32).reshape(-1)
        rgb_pose = np.asarray(f["rgb_camera_pose"], dtype=np.float32).reshape(4, 4)
        depth_pose = np.asarray(f["depth_pose"], dtype=np.float32).reshape(4, 4)
        meta = {
            "rgb_camera_pose": rgb_pose,
            "depth_pose": depth_pose,
            "fx": float(intr[0]), "fy": float(intr[1]), "cx": float(intr[2]), "cy": float(intr[3]),
            "depth_fx": float(dintr[0]), "depth_fy": float(dintr[1]),
            "depth_cx": float(dintr[2]), "depth_cy": float(dintr[3]),
            "image_width": int(f["image_width"]), "image_height": int(f["image_height"]),
        }

        # ---- align depth into RGB frame + downsample RGB to processing res ----
        depth_array = build_depth_in_rgb_frame(meta, depth_native, self._target_h, self._target_w)
        image_pil = image_pil.convert("RGB").resize((self._target_w, self._target_h), Image.LANCZOS)
        image_array = np.array(image_pil)

        # ---- pose: [frame] + 16 row-major c2w (NOT flipped; matches gRPC) ----
        frame_number = int(f["frame_number"])
        pose_data = [frame_number] + rgb_pose.reshape(-1).tolist()

        # ---- per-frame depth cap (0.0/unset -> None -> YAML default downstream) ----
        max_depth = float(f.get("max_depth_m", 0.0) or 0.0)
        max_depth_m = None if max_depth == 0.0 else max_depth

        # semantic_data carries no client timestamp; stamp both with server time.
        ts = time.perf_counter_ns()
        return (
            image_pil,       # 0
            image_array,     # 1
            depth_array,     # 2
            pose_data,       # 3
            frame_number,    # 4
            ts,              # 5  client_timestamp (unavailable -> server time)
            ts,              # 6  server_timestamp
            {},              # 7  per-frame time_dict (filled by consumer)
            max_depth_m,     # 8
        )
