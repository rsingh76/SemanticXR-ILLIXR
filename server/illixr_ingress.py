#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Rahul Singh, University of Illinois Urbana-Champaign <rahuls10@illinois.edu>
# SPDX-License-Identifier: Apache-2.0
"""ILLIXR ingress: convert a switchboard ``semantic_data`` dict to the inference
9-tuple, reusing the EXACT depth decode and reprojection of the live Quest gRPC
path so the result is identical-by-construction to
``grpc_server._process_frame_request`` (is_quest).

This runs inside the **inference worker process** (not the relay), so the
converted payload is never pickled across the mp.Queue boundary: the raw dict is
~5 MB/frame (1280x1280x3 RGB + 320x320x2 depth) while the 9-tuple is ~9 MB
(two RGB copies at the processing resolution + a float32 depth-in-RGB map).
Converting here keeps the smaller form on the queue and keeps the relay loop —
which also services voice queries and drains responses — free of per-frame work.

RGB IS ALREADY DECODED. As of ILLIXR commit 12a3f448 the ``semantic_python``
plugin GPU-decodes the H.265 stream with NVDEC and hands Python an (H, W, 3)
uint8 array (``entry_to_numpy`` in switchboard_bindings.hpp). Earlier revisions
of the plugin (through 9256b074, 2026-06-11) passed the encoded HEVC bytes and
this file decoded them itself; that path is gone. Depth is still delivered raw.
If ``cmake/GetSemanticXR.cmake``'s pin is ever moved back to a pre-12a3f448
plugin, this file must be reverted in lockstep.

The plugin returns an EMPTY ``image`` array when its decoded-frame cache has no
entry for the frame (decode not finished, decoder still buffering, decode error,
or eviction), so an empty/ill-shaped array here means "drop this frame", not
"corrupt input".

semantic_data dict schema (plugins/semantic_python/switchboard_bindings.hpp:124-144):
  image                   numpy uint8 (H, W, 3) — DECODED RGB, may be empty
  depth                   numpy uint8 (flat)    — raw R16_UNORM bytes
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
    """semantic_data dict -> inference 9-tuple. One per worker session."""

    def __init__(self, config=None):
        # No video decoder here: ILLIXR decodes on the GPU (see module docstring).
        self._target_w, self._target_h = _quest_target_resolution()
        self._dims_warned = False
        print(f"🔁 [ingress] IllixrFrameConverter ready (ILLIXR-decoded RGB, target {self._target_w}x{self._target_h})", flush=True)

    def convert(self, f):
        """semantic_data dict -> inference 9-tuple, or None to drop the frame.

        Mirrors grpc_server._process_frame_request is_quest path:
          - RGB arrives decoded from ILLIXR (no H.265 step; see module docstring)
          - R16 depth decode (grpc_server.py:343-346)
          - meta + build_depth_in_rgb_frame + LANCZOS resize (grpc_server.py:401-423)
          - pose = [frame] + 16 row-major c2w, unflipped (grpc_server.py:366-372)
          - 9-tuple layout (inference_service.py:129-139)
        """
        # ---- RGB: already decoded by ILLIXR as (H, W, 3) uint8 ----
        rgb = np.asarray(f["image"])
        if rgb.size == 0 or rgb.ndim != 3 or rgb.shape[2] != 3:
            return None  # cache miss / decoder still buffering -> drop frame
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8, copy=False)
        image_pil = Image.fromarray(rgb, mode="RGB")
        native_h, native_w = int(rgb.shape[0]), int(rgb.shape[1])

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
            # Use the decoded array's own dimensions: build_depth_in_rgb_frame
            # scales projected depth coords by target/native, and the RGB below is
            # resized from this same shape. If the plugin's reported intrinsics
            # resolution disagrees, fx/fy/cx/cy are inconsistent with the pixels
            # too — that is a producer bug we can only surface, not repair.
            "image_width": native_w, "image_height": native_h,
        }
        rep_w, rep_h = int(f["image_width"]), int(f["image_height"])
        if (rep_w, rep_h) != (native_w, native_h) and not self._dims_warned:
            self._dims_warned = True
            print(f"⚠️  [ingress] decoded RGB is {native_w}x{native_h} but semantic_data "
                  f"reports {rep_w}x{rep_h}; intrinsics do not match the pixels", flush=True)

        # ---- align depth into RGB frame + downsample RGB to processing res ----
        depth_array = build_depth_in_rgb_frame(meta, depth_native, self._target_h, self._target_w)
        image_pil = image_pil.resize((self._target_w, self._target_h), Image.LANCZOS)
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
