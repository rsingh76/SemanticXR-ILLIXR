# SPDX-FileCopyrightText: Copyright (c) 2026 Rahul Singh, University of Illinois Urbana-Champaign <rahuls10@illinois.edu>
# SPDX-License-Identifier: Apache-2.0

"""Meta Quest dataset implementation for semantic SLAM operations.

Local-file mode (replay): a scene directory written by
``server/components/grpc_server.py::SLAMGRPCServer._save_quest_replay_frame``
during a live Quest session with ``dataset.enabled=true``. Layout:

    <scene>/
        intrinsics.json                  # scene-level RGB+depth intrinsics + sizes
        decoded_jpg/frame_NNNNNN.jpg     # RGB at NATIVE resolution
        depth/depth_NNNNNN.npy           # float32 metric depth at NATIVE depth resolution
        meta/meta_NNNNNN.json            # per-frame poses (as received) + timestamps

Streaming mode (server/components/grpc_server.py → UpstreamSyncMessage_quest)
constructs the (RGB, aligned-depth, rgb_camera_pose) tuple in-process and
reuses ``build_depth_in_rgb_frame`` here for the depth→RGB-frame resample.

Coordinate / convention notes (these are what's saved on disk):
  * Poses (``rgb_camera_pose``, ``depth_pose``) are stored *as received* from
    the proto: OpenXR right-handed, Y-up, -Z forward, camera-to-world. The
    OpenGL→OpenCV flip (used by the rest of the pipeline) is applied at
    consumption time in ``QuestDataset.load_poses``, not on disk.
  * Intrinsics ``cy_yup`` is the on-device viewport principal point measured
    from the image bottom. Callers that want OpenCV-style cy must flip with
    ``cy_top = image_height - cy_yup``.
Depth format: float32 metric depth in meters, native (depth-camera) resolution.
"""

import json
import os

import numpy as np
import torch
from typing import Optional

from .base import BaseDataset


# --- Tunable thresholds for the Quest RGB/depth alignment math ---------------
#
# Stricter than the pipeline's generic ``depth > 0`` check because Quest
# IR-depth emits sub-cm garbage values, and the RGB reprojection step needs a
# safety margin away from the lens to avoid division blow-ups.
DEPTH_MIN_VALID_M = 0.05       # reject depth pixels nearer than this (m)
PROJECTION_MIN_Z_M = 0.2       # in RGB camera frame, treat z_fwd <= this as invalid (m)


def load_quest_intrinsics(scene_dir):
    """Load ``intrinsics.json`` from a Quest replay scene directory.

    Returns the parsed dict; structure is documented in
    ``SLAMGRPCServer._save_quest_replay_frame``.
    """
    path = os.path.join(scene_dir, 'intrinsics.json')
    with open(path, 'r') as f:
        return json.load(f)


def load_quest_meta(meta_json_path):
    """Load a per-frame ``meta_NNNNNN.json`` plus the scene's ``intrinsics.json``
    into a single flat dict matching the contract of ``build_depth_in_rgb_frame``.

    ``meta_json_path`` is expected to live at ``<scene>/meta/meta_NNNNNN.json``;
    ``intrinsics.json`` is read from the scene root (one level up from
    ``meta/``).

    Returned keys: ``frame_number``, ``rgb_camera_pose`` (4×4 float32, OpenXR
    Y-up c2w), ``depth_pose`` (same), RGB intrinsics ``fx/fy/cx/cy``
    (cy is *cy_yup*, viewport Y-up — matches the on-disk convention), depth
    intrinsics ``depth_fx/fy/cx/cy``, ``image_width/height`` (RGB native),
    ``depth_width/height``, and the three timestamp fields.
    """
    with open(meta_json_path, 'r') as f:
        frame = json.load(f)
    scene_dir = os.path.dirname(os.path.dirname(meta_json_path))
    scene = load_quest_intrinsics(scene_dir)
    rgb = scene['rgb']
    depth = scene['depth']
    return {
        'frame_number': frame['frame_number'],
        'rgb_camera_pose': np.array(frame['rgb_camera_pose'], dtype=np.float32),
        'depth_pose': np.array(frame['depth_pose'], dtype=np.float32),
        'fx': rgb['fx'], 'fy': rgb['fy'], 'cx': rgb['cx'], 'cy': rgb['cy_yup'],
        'depth_fx': depth['fx'], 'depth_fy': depth['fy'],
        'depth_cx': depth['cx'], 'depth_cy': depth['cy_yup'],
        'image_width': rgb['image_width'], 'image_height': rgb['image_height'],
        'depth_width': depth['image_width'], 'depth_height': depth['image_height'],
        'rgb_timestamp_ns': frame.get('rgb_timestamp_ns', 0),
        'depth_timestamp_ns': frame.get('depth_timestamp_ns', 0),
        'server_timestamp_ns': frame.get('server_timestamp_ns', 0),
    }


def _unproject_depth_to_world(depth, d_pose_gl, d_fx, d_fy, d_cx, d_cy_yup):
    """Unproject a Quest depth image to world points in OpenGL space.

    Mirrors ``unproject_depth_to_world`` in unity_rgbd_reconstruct.py: camera
    convention is X-right, Y-up, -Z-forward, and ``d_cy_yup`` is the viewport-Y
    principal point (from the bottom of the image).
    """
    dh, dw = depth.shape
    uu, vv = np.meshgrid(np.arange(dw), np.arange(dh))
    d = depth.astype(np.float32)
    valid = d > DEPTH_MIN_VALID_M
    x_l = (uu - d_cx) / d_fx * d
    y_l = (d_cy_yup - vv) / d_fy * d
    z_l = -d
    pts_cam = np.stack([x_l, y_l, z_l, np.ones_like(d)], axis=-1)
    pts_world = (d_pose_gl @ pts_cam.reshape(-1, 4).T).T[:, :3].reshape(dh, dw, 3)
    return pts_world, valid


def _project_world_to_rgb(pts_world, rgb_pose_gl, fx, fy, cx, cy_yup, img_h):
    """Project world points into a Quest RGB camera, returning JPEG pixel coords.

    Mirrors ``project_world_to_rgb`` in unity_rgbd_reconstruct.py. Returns
    (u, v_jpeg, z_fwd) where (u, v_jpeg) are in stored-JPEG Y-down pixel space
    and z_fwd > 0 means "in front of the camera".
    """
    R = rgb_pose_gl[:3, :3]
    t = rgb_pose_gl[:3, 3]
    flat = pts_world.reshape(-1, 3)
    p_local = (R.T @ (flat - t).T).T
    z_fwd = -p_local[:, 2]
    safe_z = np.where(z_fwd > PROJECTION_MIN_Z_M, z_fwd, 1e-6)
    u = fx * p_local[:, 0] / safe_z + cx
    v_sensor = fy * p_local[:, 1] / safe_z + cy_yup
    v_jpeg = img_h - v_sensor
    shape = pts_world.shape[:2]
    return u.reshape(shape), v_jpeg.reshape(shape), z_fwd.reshape(shape)


def build_depth_in_rgb_frame(meta, depth, target_h, target_w):
    """Resample a Quest depth frame onto the RGB camera at the target resolution.

    For every valid depth pixel we unproject with the *depth* camera's pose +
    intrinsics, then project back into the *RGB* camera with the RGB pose +
    intrinsics (all values come straight from the meta file). The projected
    pixel coordinates are scaled to the target RGB resolution and the RGB-frame
    forward distance (z_fwd, OpenCV forward convention) is scattered into the
    output array. Pixels without coverage are left at 0.

    Args:
        meta: dict returned by load_quest_meta() (or an in-process equivalent
            constructed in the live-streaming gRPC path)
        depth: native-resolution depth image (H_d, W_d), metric, float32
        target_h, target_w: output depth-in-RGB-frame resolution (typically the
            resolution the rest of the pipeline will consume the RGB at)

    Returns:
        depth_rgb: (target_h, target_w) float32 array of forward distances in
            the RGB camera frame (OpenCV convention: larger = further forward).
    """
    rgb_pose = meta['rgb_camera_pose']
    d_pose = meta['depth_pose']
    rgb_w, rgb_h = meta['image_width'], meta['image_height']

    pts_world, valid = _unproject_depth_to_world(
        depth, d_pose,
        meta['depth_fx'], meta['depth_fy'], meta['depth_cx'], meta['depth_cy'],
    )
    u, v, z_fwd = _project_world_to_rgb(
        pts_world, rgb_pose,
        meta['fx'], meta['fy'], meta['cx'], meta['cy'], rgb_h,
    )

    # Rescale from native RGB resolution to target resolution.
    sx = target_w / float(rgb_w)
    sy = target_h / float(rgb_h)
    iu = np.round(u * sx).astype(np.int32)
    iv = np.round(v * sy).astype(np.int32)
    in_bounds = (iu >= 0) & (iu < target_w) & (iv >= 0) & (iv < target_h)
    m = valid & (z_fwd > PROJECTION_MIN_Z_M) & in_bounds

    depth_rgb = np.zeros((target_h, target_w), dtype=np.float32)
    # Z-buffer: when multiple depth pixels land on the same RGB pixel, keep the
    # closest one so near surfaces occlude far ones.
    if m.any():
        flat_idx = iv[m].astype(np.int64) * target_w + iu[m].astype(np.int64)
        z_vals = z_fwd[m].astype(np.float32)
        order = np.argsort(-z_vals)  # far first so nearer overwrites
        flat_idx = flat_idx[order]
        z_vals = z_vals[order]
        depth_rgb.reshape(-1)[flat_idx] = z_vals
    return depth_rgb


class QuestDataset(BaseDataset):
    """Quest dataset for real-time streaming or local file processing.

    Two modes:
      * Local-file: ``basedir`` points at a scene directory of meta_XXXXXX.txt
        / depth_XXXXXX.npy / decoded_jpg/*.jpg tuples. Intrinsics come from
        the first meta file.
      * Streaming: ``basedir`` is None (get_dataset sets this when
        scene_name == 'quest'). Intrinsics come from QUEST.yaml's
        ``streaming_defaults.rgb`` section — Quest 3S sensor intrinsics are
        factory-fixed per device, so the yaml defaults match what the proto
        would report frame-by-frame. No per-frame handoff required.

    In both cases the loaded intrinsics are scaled to the
    ``desired_width x desired_height`` processing target (QUEST.yaml's
    camera_params.image_width/height, typically 640x640). cy is flipped from
    viewport Y-up to image Y-down so the rest of the OpenCV-convention
    pipeline Just Works.
    """

    def __init__(
        self,
        config_dict,
        basedir=None,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = None,
        desired_width: Optional[int] = None,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cpu",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = False,
        **kwargs,
    ):
        # Processing-target resolution must come from the caller (which passes
        # cfg.image_width/height, ultimately QUEST.yaml's camera_params). We
        # don't carry our own defaults to avoid drift with the yaml.
        if desired_height is None or desired_width is None:
            raise ValueError(
                "QuestDataset requires desired_height and desired_width "
                "(populated from QUEST.yaml via mapping_utils.setup())."
            )
        # Load intrinsics from either (a) the first meta file in basedir for
        # local-file mode, or (b) streaming_defaults in QUEST.yaml when no
        # basedir is provided (streaming Quest, instantiated via get_dataset
        # with scene_name='quest'). Meta + yaml both store cy in viewport
        # Y-up convention (from image bottom); SemanticXR expects cy Y-down
        # (from image top), so we flip it here before scaling.
        self.input_folder = basedir
        if basedir and os.path.isdir(basedir):
            scene_intr = load_quest_intrinsics(basedir)
            rgb_intr = scene_intr['rgb']
            fx = rgb_intr['fx']
            fy = rgb_intr['fy']
            cx = rgb_intr['cx']
            cy_yup = rgb_intr['cy_yup']
            image_width = rgb_intr['image_width']
            image_height = rgb_intr['image_height']
            intrinsics_source = os.path.join(basedir, 'intrinsics.json')
        else:
            # Streaming mode: QuestDataset is created before any frames have
            # arrived, so fall back to the device-constant values in
            # QUEST.yaml's streaming_defaults. Quest 3S sensor intrinsics are
            # fixed per device, so this is equivalent to using the actual
            # first-frame intrinsics without needing a handoff.
            defaults = (config_dict.get('streaming_defaults') or {}).get('rgb')
            if not defaults or float(defaults.get('fx', 0.0)) == 0.0:
                raise ValueError(
                    "QuestDataset called with basedir=None (streaming mode) but "
                    "QUEST.yaml has no streaming_defaults.rgb or its fx is still "
                    "0.0. Fill in device intrinsics or pass a basedir."
                )
            fx = float(defaults['fx'])
            fy = float(defaults['fy'])
            cx = float(defaults['cx'])
            cy_yup = float(defaults['cy_yup'])
            image_width = int(defaults['image_width'])
            image_height = int(defaults['image_height'])
            intrinsics_source = 'QUEST.yaml::streaming_defaults.rgb'

        cy_top = image_height - cy_yup
        config_dict['camera_params']['fx'] = fx
        config_dict['camera_params']['fy'] = fy
        config_dict['camera_params']['cx'] = cx
        config_dict['camera_params']['cy'] = cy_top
        config_dict['camera_params']['image_width'] = image_width
        config_dict['camera_params']['image_height'] = image_height
        print(f"Quest intrinsics loaded from {intrinsics_source}: "
              f"fx={fx:.2f} fy={fy:.2f} "
              f"cx={cx:.2f} cy_yup={cy_yup:.2f} -> cy_top={cy_top:.2f} "
              f"({image_width}x{image_height})")

        super().__init__(
            config_dict=config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            channels_first=channels_first,
            normalize_color=normalize_color,
            device=device,
            dtype=dtype,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            relative_pose=relative_pose,
            **kwargs,
        )
    def __len__(self):
        raise NotImplementedError("Quest is a streaming dataset")

    def load_poses(self, pose):
        """Load a Quest rgb_camera_pose (OpenGL RH, c2w) and flip Y/Z to OpenCV.

        The rest of the pipeline expects an OpenCV-convention camera-to-world
        matrix (X-right, Y-down, Z-forward). Quest poses arrive in OpenGL/OpenXR
        convention (X-right, Y-up, -Z-forward); negating the Y and Z columns of
        the rotation block converts between them. Same trick the iPad dataset
        uses (ARKit is also OpenGL-style RH Y-up).

        Args:
            pose: list of 17 values [frame_number, 4x4 matrix values]
                  or list of 16 values [4x4 matrix values]
        """
        if isinstance(pose, (list, np.ndarray)):
            pose_array = np.array(pose)
            if pose_array.size == 17:
                pose_array = pose_array[1:]  # Strip frame number
            elif pose_array.size != 16:
                raise ValueError(f"Expected 16 or 17 pose values, got {pose_array.size}")
            c2w = pose_array.reshape(4, 4)
        else:
            raise ValueError(f"Unsupported pose type: {type(pose)}")

        # OpenXR (Quest) is right-handed Y-up, same as OpenGL.
        # Negate Y and Z axes of the rotation to convert to the
        # left-handed convention used by the rest of the pipeline.
        # This matches the iPad convention (ARKit is also Y-up right-handed).
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1

        c2w = torch.from_numpy(c2w).float()
        return c2w
