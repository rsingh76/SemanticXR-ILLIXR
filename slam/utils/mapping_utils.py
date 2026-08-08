# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mapping support utilities for semantic SLAM operations."""

import os
import time
from pathlib import Path

import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from slam.datasets import ipad, replica, scannet, quest


from slam.core.utils import (
    BG_CLASSES,
    configure_ignore_clip_filter,
    gobs_to_detection_list,
    gobs_to_detection_list_optimized,
)

ASYNC_IO=True
DEBUG_PRINT = False
def debug_print(*args, **kwargs):
    if DEBUG_PRINT:
        print(*args, **kwargs)


def init_clip_ignore_filter(cfg: DictConfig, experiment_config) -> None:
    """Pre-encode cfg.ignore_classes via CLIP text encoder and arm filter_gobs().

    Loads CLIP once (matching inference's ViT-H-14 / pretrained weights so
    embeddings live in the same space as gobs['image_feats']), encodes the
    ignore-class strings, L2-normalizes them, hands them to filter_gobs's
    module-level cache via configure_ignore_clip_filter(), then frees the
    full CLIP model — we only ever needed the text encoder once.

    Safe to call when skip_ignored_clip is False or ignore_classes is empty
    (returns immediately). Safe to call from both mapping_server and
    inference_pipeline startup paths — module state is per-process.
    """
    if not bool(cfg.get('skip_ignored_clip', False)):
        return
    classes = list(cfg.get('ignore_classes', []) or [])
    if not classes:
        print("[init_clip_ignore_filter] skip_ignored_clip=True but "
              "ignore_classes is empty; CLIP ignore filter NOT armed.")
        return

    # Imported lazily so processes that don't need CLIP-ignore don't pay the
    # open_clip import cost.
    import open_clip

    clip_name = experiment_config.model.clip.model_name
    pretrained = experiment_config.model.clip.pretrained
    device = cfg.get('device', 'cuda:0')
    threshold = float(cfg.get('ignore_clip_threshold', 0.28))

    print(f"[init_clip_ignore_filter] Loading CLIP {clip_name}/{pretrained} on "
          f"{device} to encode {len(classes)} ignore class(es)...")
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(clip_name)
    tokens = tokenizer(classes).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens).float()
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    configure_ignore_clip_filter(text_feats.detach(), threshold)

    # Free the full CLIP model — we have the [N, D] text features cached.
    del clip_model, tokenizer, tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[init_clip_ignore_filter] CLIP ignore filter armed: "
          f"{len(classes)} classes, thresh={threshold}, classes={classes}")


def get_dataset(datasetClass, config_dict, desired_height, desired_width, device, dtype, scene_name=None, test_depth_downsampling=1):
    if datasetClass.lower() == "ipad":
        dataset = ipad.IPADDataset(
            config_dict = config_dict,
            desired_height = desired_height,
            desired_width = desired_width,
            device = device,
            dtype=dtype,
            test_depth_downsampling=test_depth_downsampling,
        )
    elif datasetClass.lower() == "replica":
        dataset = replica.ReplicaDataset(
            config_dict = config_dict,
            desired_height = desired_height,
            desired_width = desired_width,
            device = device,
            dtype=dtype,
            test_depth_downsampling=test_depth_downsampling,
        )
    elif datasetClass.lower() == "quest":
        # Replay (local-file) mode: scene_name names a captured directory under
        # ``<dataset.output_directory>/quest/``. Streaming (live) mode is
        # signalled by scene_name == 'quest' (or no captured dir) and leaves
        # basedir=None so QuestDataset.__init__ falls back to QUEST.yaml's
        # streaming_defaults.rgb (device-fixed Quest 3S intrinsics).
        from config.settings import get_config as _get_config
        basedir = None
        if scene_name and scene_name not in ['quest']:
            candidate = os.path.join(_get_config().dataset.output_directory, 'quest', scene_name)
            if os.path.isdir(candidate):
                basedir = candidate
        dataset = quest.QuestDataset(
            config_dict = config_dict,
            basedir = basedir,
            desired_height = desired_height,
            desired_width = desired_width,
            device = device,
            dtype=dtype,
            test_depth_downsampling=test_depth_downsampling,
        )
    elif datasetClass.lower() == "scannet":
        # Get ScanNet root directory from environment
        scannet_root = os.environ.get('SCANNET_ROOT')

        # If we have both scene name and root, and scene name looks like a real scene,
        # then set up for file-based loading to get proper intrinsics
        if (scene_name and scannet_root and
            scene_name not in ['scannet', 'replica', 'ipad', 'quest']):
            # For ScanNet scenes, the structure is: SCANNET_ROOT/scene_name
            # This matches the logic in factory.py _get_scannet_paths
            sequence = f"{scene_name}"

            dataset = scannet.ScanNetDataset(
                config_dict = config_dict,
                basedir = scannet_root,
                sequence = sequence,
                desired_height = desired_height,
                desired_width = desired_width,
                device = device,
                dtype=dtype,
            )
        else:
            # For cases without proper scene name, create dataset without file paths
            dataset = scannet.ScanNetDataset(
                config_dict = config_dict,
                basedir = None,
                sequence = None,
                desired_height = desired_height,
                desired_width = desired_width,
                device = device,
                dtype=dtype,
            )
    else:
        raise NotImplementedError(f"Dataset class {datasetClass} not supported")
    return dataset


def process_cfg(cfg: DictConfig, useDetector, datasetClass):
    # cfg.dataset_root = Path(cfg.dataset_root)
    if datasetClass.lower() == "replica":
        cfg.dataset_config = "./REPLICA.yaml"
    elif datasetClass.lower() == "scannet":
        cfg.dataset_config = "./ScanNet.yaml"
    elif datasetClass.lower() == "quest":
        cfg.dataset_config = "./QUEST.yaml"
        
    cfg.dataset_config = Path(cfg.dataset_config)
    
    if cfg.dataset_config.name != "multiscan.yaml":
        # For datasets whose depth and RGB have the same resolution
        # Set the desired image heights and width from the dataset config
        # Get path to datasets directory relative to this file
        datasets_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
        cfg.dataset_config = omegaconf.OmegaConf.load(os.path.join(datasets_path, cfg.dataset_config))
        if cfg.image_height is None:
            cfg.image_height = cfg.dataset_config.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = cfg.dataset_config.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
        if useDetector:
            cfg.mask_conf_threshold = 0.25
            cfg.skip_bg = False

        # Promote dataset-level ignore list (e.g. QUEST.yaml's human classes)
        # onto the root cfg so filter_gobs() can read it directly.
        dataset_ignores = list(cfg.dataset_config.get("ignore_classes", []) or [])
        if dataset_ignores:
            cfg.ignore_classes = dataset_ignores
            cfg.skip_ignored = True
            print(f"[process_cfg] skip_ignored=True, ignore_classes={dataset_ignores}")

        # Promote CLIP-similarity ignore flag + threshold (independent of the
        # string-match variant). filter_gobs() will use both if both are on.
        if cfg.dataset_config.get("skip_ignored_clip", False):
            cfg.skip_ignored_clip = True
            cfg.ignore_clip_threshold = float(
                cfg.dataset_config.get("ignore_clip_threshold",
                                       cfg.get("ignore_clip_threshold", 0.28))
            )
            print(f"[process_cfg] skip_ignored_clip=True, "
                  f"ignore_clip_threshold={cfg.ignore_clip_threshold}")

        # Promote the debug flag so filter_gobs / filter_detections_clip_ignore
        # see it on root cfg (they read cfg.get('debug_ignore_drops', False)).
        # Without this, debug_ignore_drops stays at base.yaml's default (False)
        # even when QUEST.yaml sets it True — and drops would happen silently.
        if cfg.dataset_config.get("debug_ignore_drops", False):
            cfg.debug_ignore_drops = True
            print("[process_cfg] debug_ignore_drops=True")

        # Promote per-dataset post-process cadence + merge thresholds onto root
        # cfg. mapping_server / inference_pipeline read these as cfg.<key> in
        # their per-frame loop (see {denoise,filter,merge}_interval gates and
        # merge_overlap_objects). Defaults live in base.yaml; QUEST.yaml may
        # override any subset.
        for _k in (
            "denoise_interval", "filter_interval", "merge_interval",
            "merge_overlap_thresh", "merge_visual_sim_thresh", "merge_text_sim_thresh",
            "max_depth_m",
        ):
            if _k in cfg.dataset_config:
                cfg[_k] = cfg.dataset_config[_k]
                print(f"[process_cfg] {_k}={cfg[_k]} (from {cfg.dataset_config.get('dataset_name', 'dataset')}.yaml)")

    else:
        # For dataset whose depth and RGB have different resolutions
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg
    
# @hydra.main(version_base=None, config_path="../utilsSLAM/", config_name="base")
def setup(useDetector, datasetClass):
    # Get path to utilsSLAM directory relative to this file  
    datasets_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
    cfg = omegaconf.OmegaConf.load(os.path.join(datasets_path, "base.yaml"))
    cfg = process_cfg(cfg, useDetector, datasetClass)
    return cfg


def resolve_session_max_depth(client_value, cfg):
    """Resolve the per-session far-depth cap from client stamp + dataset YAML.

    Called once per session (on the first frame after start / scene reset);
    the resolved value is then frozen on the consumer for the rest of the
    session — see inference_pipeline.inference_consumer / mapping_server.

    Wire-vs-YAML semantics (single source of truth, kept in sync with the
    proto comment on UpstreamSyncMessage.max_depth_m):
      ``client_value`` (Optional[float], from grpc_server._resolve_max_depth_m):
          * None — client did not stamp the field. Fall back to YAML.
          * <0   — client explicitly requested no cap.
          * >0   — client-supplied cap in meters.
      ``cfg.max_depth_m`` (per-dataset YAML, e.g. QUEST.yaml; promoted to root
      cfg by process_cfg):
          * None / null / <=0 — no cap.
          * >0                — cap in meters.
    Returns Optional[float]; ``None`` downstream means "no cap".
    """
    if client_value is None:
        yaml_value = getattr(cfg, "max_depth_m", None)
        if yaml_value is None:
            return None
        try:
            yaml_value = float(yaml_value)
        except (TypeError, ValueError):
            return None
        return yaml_value if yaml_value > 0 else None
    if client_value < 0:
        return None
    return float(client_value)


def create_pcd_parallel(image_np, depth_array, pose, frameNumber, dataset, cfg, classes, gobs, output_receiver_list, pipelined_mapping, datasetClass, time_dict, max_depth_m=None):
    start = time.perf_counter_ns()  
    color_tensor, depth_tensor, intrinsics, unt_pose = dataset.getItems(image_np, depth_array, pose)
    assert not pipelined_mapping, "pipelined_mapping must be False to reach here"
        
    color_np = color_tensor.cpu().numpy() # (H, W, 3)
    image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
    
    # Get the depth image
    depth_tensor = depth_tensor[..., 0]
    depth_array = depth_tensor.cpu().numpy()

    # Get the intrinsics matrix
    cam_K = intrinsics.cpu().numpy()[:3, :3]
    
    unt_pose = unt_pose.cpu().numpy()
    # Don't apply any transformation otherwise
    adjusted_pose = unt_pose
    
    
    fg_detection_list, bg_detection_list, idx_to_keep = gobs_to_detection_list_optimized(
        cfg = cfg,
        image = image_rgb,
        depth_array = depth_array,
        cam_K = cam_K,
        idx = frameNumber,
        gobs = gobs,
        trans_pose = adjusted_pose,
        class_names = classes,
        BG_CLASSES = BG_CLASSES,
        color_path = None,
        pipelined_mapping=pipelined_mapping,
        dataset_type=datasetClass,
        time_dict=time_dict,
        max_depth_m=max_depth_m,
    )
    output_receiver_list.append(fg_detection_list)
    output_receiver_list.append(bg_detection_list)
    output_receiver_list.append(idx_to_keep)
    time_to_get_data = time.perf_counter_ns() 
    # print(f"Create GOBS: {(time_to_get_data - start)/1e6} ms")
    time_dict['mapping_time'] = (time.perf_counter_ns() - start)/1e6
    time_dict['gobs_creation_time'] = (time.perf_counter_ns() - start)/1e6

