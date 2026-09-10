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
  depth_near_z            float               — OpenXR XrEnvironmentDepthImageMETA.nearZ,
                                                POSITIVE metres (0.1 observed). NOT Unity's
                                                ZBufferParams.x (-2*sensor_near, negative),
                                                which is what the gRPC path carries.
  intrinsics              numpy float32 [fx,fy,cx,cy]   (RGB)
  depth_intrinsics        numpy float32 [fx,fy,cx,cy]
  rgb_camera_pose         numpy float32 (4,4) row-major c2w (OpenXR, unflipped)
  depth_pose              numpy float32 (4,4) row-major c2w
  rgb_timestamp_ns        int                 CLOCK_BOOTTIME ns (encoder PTS)
  depth_timestamp_ns      int                 OVRPlugin predicted-display time ns --
                                              NOT the same base as rgb_timestamp_ns
  max_depth_m             float               — 0.0/unset => use YAML default
"""
import os
import time

import numpy as np
import yaml
from PIL import Image

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


def _rgb_intrinsics_resolution():
    """(width, height) the wire RGB intrinsics were computed for.

    android_sensors publishes the device intrinsics unscaled, at the RGB
    sensor's native resolution, while the encoder emits a downscaled image --
    so cx/cy land far from the delivered image centre.  The Unity path does
    this rescale itself (StreamingOrchestrator.cs:487-503) before sending.
    Anchor: QUEST.yaml::streaming_defaults.rgb, overridable per-device.
    """
    env = os.environ.get("ILLIXR_RGB_INTRINSICS_RES")
    if env:
        w, _, h = env.partition("x")
        return int(w), int(h or w)
    quest_yaml = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "slam", "datasets", "QUEST.yaml",
    )
    with open(quest_yaml, "r") as fh:
        cfg = yaml.safe_load(fh)
    rgb = cfg["streaming_defaults"]["rgb"]
    return int(rgb["image_width"]), int(rgb["image_height"])


class _ReplayRequestShim:
    """Duck-typed stand-in for the gRPC request object.

    ``SLAMGRPCServer._save_quest_replay_frame`` reads a handful of fields off
    the proto; semantic_data carries all of them. Rather than duplicate that
    writer (and drift from the replay layout ``QuestDataset`` expects), hand it
    an object exposing the same attribute names.

    Sizes come from the arrays we are actually writing, not from the reported
    ``image_width/height``, because the intrinsics passed in have already been
    rescaled to the decoded resolution above -- ``intrinsics.json`` has to agree
    with the pixels in ``decoded_jpg/``.

    TEMPORARY: the proper fix is to lift ``_save_quest_replay_frame`` and
    ``_get_or_create_capture_dir`` out of the servicer into a shared writer that
    both transports call.
    """

    class _Intr:
        __slots__ = ("fx", "fy", "cx", "cy")

        def __init__(self, a):
            self.fx, self.fy, self.cx, self.cy = (float(v) for v in np.asarray(a).reshape(-1)[:4])

    def __init__(self, f, intr, dintr, rgb_wh, depth_wh):
        self.rgb_camera_pose    = np.asarray(f["rgb_camera_pose"], dtype=np.float32).reshape(-1).tolist()
        self.depth_pose         = np.asarray(f["depth_pose"], dtype=np.float32).reshape(-1).tolist()
        self.intrinsics         = self._Intr(intr)
        self.depth_intrinsics   = self._Intr(dintr)
        self.image_width,  self.image_height  = int(rgb_wh[0]), int(rgb_wh[1])
        self.depth_width,  self.depth_height  = int(depth_wh[0]), int(depth_wh[1])
        self.depth_near_z       = float(f["depth_near_z"])
        # Recorded as-is for diagnostics. They are NOT in the same time base --
        # rgb is CLOCK_BOOTTIME (encoder PTS), depth is the OVRPlugin predicted-display
        # time -- so do not subtract them without converting (see semantics.hpp).
        # Plugin builds before the stamps were exposed omit the keys; 0 then means
        # "unknown", which load_quest_meta already tolerates.
        self.rgb_timestamp_ns   = int(f.get("rgb_timestamp_ns", 0) or 0)
        self.depth_timestamp_ns = int(f.get("depth_timestamp_ns", 0) or 0)


def _make_frame_dumper(config):
    """Replay-frame writer, or None when ``dataset.enabled`` is false.

    Deliberately bypasses ``SLAMGRPCServer.__init__``: that constructor builds a
    ``VideoProcessor`` (two NVDEC decoders) and an ``InferenceService``, none of
    which the dump path touches. ``_save_quest_replay_frame`` only reaches
    ``self.config`` and ``self._capture_roots``, so bind exactly those.
    """
    if config is None or not getattr(getattr(config, "dataset", None), "enabled", False):
        return None
    try:
        from server.components.grpc_server import SLAMGRPCServer
        writer = SLAMGRPCServer.__new__(SLAMGRPCServer)
        writer.config = config
        writer._capture_roots = {}
        writer.data_dump_enabled = True
        print("💾 [ingress] dataset.enabled -> writing replay frames via "
              "SLAMGRPCServer._save_quest_replay_frame", flush=True)
        return writer
    except Exception as exc:
        print(f"⚠️  [ingress] replay frame dumping unavailable: {exc}", flush=True)
        return None


class IllixrFrameConverter:
    """semantic_data dict -> inference 9-tuple. One per worker session."""

    def __init__(self, config=None):
        # No video decoder here: ILLIXR decodes on the GPU (see module docstring).
        self._target_w, self._target_h = _quest_target_resolution()
        self._dims_warned = False
        self._dumper = _make_frame_dumper(config)
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

        # ---- depth: uint16 inverse-NDC -> metric float32 (native res) ----
        # ILLIXR's Quest producer (plugins/android_sensors) encodes the
        # environment-depth swapchain as
        #     u16_norm = 1.0 - (near_z / depth_m)
        # so the inverse is  depth_m = near_z / (1.0 - u16_norm).
        # This is NOT the Unity/gRPC convention that
        # grpc_server._decode_quest_depth_bytes implements (there, depth_near_z
        # carries Unity's ZBufferParams.x = -2*sensor_near and the R channel of a
        # preprocessed RGBA16F texture is used).  Applying that formula to this
        # stream yields negative depths that clip to an all-zero map.
        dw_i, dh_i = int(f["depth_width"]), int(f["depth_height"])
        d_bytes = np.asarray(f["depth"])
        if d_bytes.size != dw_i * dh_i * 2:
            return None
        near_z = float(f["depth_near_z"])
        raw_u16 = d_bytes.view(np.uint16).reshape(dh_i, dw_i)
        u16_norm = raw_u16.astype(np.float32) / 65535.0
        denom = 1.0 - u16_norm
        # BOTH ends of the range are "no return", not geometry:
        #   raw == 65535 -> denom <= 0     -> depth = +inf  (far / nothing hit)
        #   raw == 0     -> depth = near_z -> a point sitting ON the near plane
        # Only the first was being rejected, so every cleared/no-return pixel
        # decoded to exactly near_z and unprojected to ~near_z in front of the
        # camera.  DEPTH_MIN_VALID_M (0.05 m, quest.py:46) does not catch that
        # because near_z > 0.05.  Those points then pile up AT the camera
        # position, which is what produced run_9's blob ~1 m above the tracking
        # origin (see the 'wood floor' objects at Y=0.9-1.1).  doug_test.py:305
        # already treats raw == 0 as invalid; match it.
        valid_raw = (raw_u16 > 0) & (denom > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_native = np.where(valid_raw, near_z / denom, 0.0).astype(np.float32)
        depth_native = np.clip(
            np.nan_to_num(depth_native, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 100.0)
        if not np.any(depth_native > 0):
            return None

        # ---- meta (scalar split intrinsics + 4x4 poses, unflipped) ----
        intr = np.asarray(f["intrinsics"], dtype=np.float32).reshape(-1)
        # Rescale the RGB intrinsics from the resolution they were computed for
        # to the resolution actually delivered (mirrors the Unity path).  Square
        # source and target here, so this is a uniform scale with no crop term;
        # a non-square change would additionally need the crop offsets that
        # StreamingOrchestrator.cs:493-500 applies.
        _ir_w, _ir_h = _rgb_intrinsics_resolution()
        if _ir_w and _ir_h and (_ir_w != native_w or _ir_h != native_h):
            _sx, _sy = native_w / float(_ir_w), native_h / float(_ir_h)
            if not getattr(self, "_scale_logged", False):
                self._scale_logged = True
                print(f"🔧 [ingress] rescaling RGB intrinsics {_ir_w}x{_ir_h} -> "
                      f"{native_w}x{native_h} (x{_sx:.4f}): "
                      f"fx {intr[0]:.2f}->{intr[0]*_sx:.2f}  cx {intr[2]:.2f}->{intr[2]*_sx:.2f}"
                      f"  cy {intr[3]:.2f}->{intr[3]*_sy:.2f}", flush=True)
            intr = intr * np.array([_sx, _sy, _sx, _sy], dtype=np.float32)
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

        # ---- replay dump: same writer the gRPC path uses ------------------
        # Called here deliberately, with NATIVE-resolution RGB and PRE-alignment
        # depth, matching grpc_server.py:397-400 so a dump replays identically.
        if self._dumper is not None:
            self._dumper._save_quest_replay_frame(
                int(f["frame_number"]), image_pil, depth_native,
                _ReplayRequestShim(f, intr, dintr,
                                   (native_w, native_h),
                                   (depth_native.shape[1], depth_native.shape[0])),
                time.perf_counter_ns())

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
