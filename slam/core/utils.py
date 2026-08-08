# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLAM utility functions for object detection and mapping."""

import copy
import cv2
import json
import time
from collections import Counter

import faiss
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from slam.utils.general_utils import to_tensor, to_numpy, Timer
from .slam_classes import MapObjectList, DetectionList

from slam.utils.ious import (
    compute_3d_iou,
    compute_3d_iou_accuracte_batch,
    mask_subtract_contained,
    compute_iou_batch
)
from slam.datasets.base import from_intrinsics_matrix
from config.settings import get_config

experiment_config = None

# Canonical list of "background" semantic classes: fused into the map as a
# single per-class bg object when cfg.skip_bg is False. Re-exported from
# slam.utils.mapping_utils for back-compat. Distinct from cfg.ignore_classes
# (which drops masks entirely — see filter_gobs).
BG_CLASSES = ["wall", "floor", "ceiling"]

# Per-process state for the CLIP-similarity ignore filter. Populated once at
# mapping/inference consumer startup by configure_ignore_clip_filter(); read
# per-frame inside filter_gobs() when cfg.skip_ignored_clip is True.
# text_feats: [N_ignore, D] torch.Tensor, L2-normalized in float32. Lives in
# the same CLIP embedding space as gobs['image_feats'] (see model_utils.py).
_IGNORE_CLIP_STATE: dict = {"text_feats": None, "threshold": 0.28}


def configure_ignore_clip_filter(text_feats, threshold: float = 0.28) -> None:
    """Cache pre-encoded CLIP text embeddings for the ignore-class filter.

    Called once per process at startup with the L2-normalized text features
    of cfg.ignore_classes. filter_gobs() then dot-products them against each
    mask's gobs['image_feats'] and drops masks whose max similarity exceeds
    `threshold` — a class-name-free way to drop humans (or anything else).
    """
    _IGNORE_CLIP_STATE["text_feats"] = text_feats
    _IGNORE_CLIP_STATE["threshold"] = float(threshold)


def reset_ignore_clip_filter() -> None:
    """Disable the CLIP ignore filter for this process (used in tests / teardown)."""
    _IGNORE_CLIP_STATE["text_feats"] = None


def compute_clip_ignore_drops(img_feats):
    """Return (drop_mask, max_sims) for a batch of CLIP image embeddings.

    img_feats: [N, D] or [D] numpy array or torch tensor, L2-normalized in the
    same CLIP space as _IGNORE_CLIP_STATE['text_feats'].
    Returns (None, None) if the filter isn't armed or there's nothing to score.

    Shared by filter_gobs() (pipelined path, gobs already carries feats) and
    filter_detections_clip_ignore() (non-pipelined path, feats are attached to
    each detection AFTER filter_gobs has run).
    """
    text_feats = _IGNORE_CLIP_STATE["text_feats"]
    if text_feats is None or img_feats is None or len(img_feats) == 0:
        return None, None
    if not isinstance(img_feats, torch.Tensor):
        img_feats_t = torch.as_tensor(img_feats, dtype=text_feats.dtype, device=text_feats.device)
    else:
        img_feats_t = img_feats.to(dtype=text_feats.dtype, device=text_feats.device)
    if img_feats_t.dim() == 1:
        img_feats_t = img_feats_t.unsqueeze(0)
    sims = img_feats_t @ text_feats.T                                       # [N, N_ignore]
    max_sims, _ = sims.max(dim=-1)
    drop = (max_sims > _IGNORE_CLIP_STATE["threshold"]).cpu().numpy()
    return drop, max_sims.detach().cpu().numpy()


def filter_detections_clip_ignore(cfg: DictConfig, fg_detection_list, bg_detection_list):
    """Drop detections whose CLIP image embedding matches an ignore-class text embedding.

    Counterpart to filter_gobs's CLIP-ignore block for the non-pipelined path:
    there, filter_gobs runs in a sibling thread before CLIP has completed, so
    gobs['image_feats'] is None when filter_gobs inspects it and that block is
    a no-op. By the time this function is called, clip_ft has been attached
    to each detection — so we can run the same drop logic on each det's
    already-attached feature.

    Returns new (fg_detection_list, bg_detection_list) with matching dets
    removed. Safe to call when the filter is disarmed or empty (returns inputs
    unchanged). Caller should pass the root cfg (for skip_ignored_clip /
    debug_ignore_drops flags). Uses compute_clip_ignore_drops() so tuning the
    threshold in one place affects both paths.
    """
    if not bool(cfg.get('skip_ignored_clip', False)):
        return fg_detection_list, bg_detection_list
    if len(fg_detection_list) == 0 and len(bg_detection_list) == 0:
        return fg_detection_list, bg_detection_list

    all_dets = list(fg_detection_list) + list(bg_detection_list)
    clip_fts = [obj.get('clip_ft') for obj in all_dets]
    if not all(ft is not None for ft in clip_fts):
        return fg_detection_list, bg_detection_list

    stacked = torch.stack([ft.view(-1) for ft in clip_fts])
    drop, sims = compute_clip_ignore_drops(stacked)
    if drop is None or not drop.any():
        return fg_detection_list, bg_detection_list

    n_fg = len(fg_detection_list)
    fg_keep = [i for i in range(n_fg) if not drop[i]]
    bg_keep = [i for i in range(len(bg_detection_list)) if not drop[n_fg + i]]

    if bool(cfg.get('debug_ignore_drops', False)):
        dropped_info = [
            (all_dets[i].get('class_name', ['?'])[0], round(float(sims[i]), 3))
            for i in range(len(drop)) if drop[i]
        ]
        print(f"[clip_ignore] dropped {int(drop.sum())} detection(s): {dropped_info}")

    return (
        DetectionList([fg_detection_list[i] for i in fg_keep]),
        DetectionList([bg_detection_list[i] for i in bg_keep]),
    )

def get_classes_colors(classes):
    class_colors = {}

    # Generate a random color for each class
    for class_idx, class_name in enumerate(classes):
        # Generate random RGB values between 0 and 255
        r = np.random.randint(0, 256)/255.0
        g = np.random.randint(0, 256)/255.0
        b = np.random.randint(0, 256)/255.0

        # Assign the RGB values as a tuple to the class in the dictionary
        class_colors[class_idx] = (r, g, b)

    class_colors[-1] = (0, 0, 0)

    return class_colors

def create_or_load_colors(cfg, filename="gsa_classes_tag2text"):
    
    # get the classes, should be saved when making the dataset
    classes_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}.json"
    classes  = None
    with open(classes_fp, "r") as f:
        classes = json.load(f)
    
    # create the class colors, or load them if they exist
    class_colors  = None
    class_colors_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}_colors.json"
    if class_colors_fp.exists():
        with open(class_colors_fp, "r") as f:
            class_colors = json.load(f)
        print("Loaded class colors from ", class_colors_fp)
    else:
        class_colors = get_classes_colors(classes)
        class_colors = {str(k): v for k, v in class_colors.items()}
        with open(class_colors_fp, "w") as f:
            json.dump(class_colors, f)
        print("Saved class colors to ", class_colors_fp)
    return classes, class_colors

def create_object_pcd(depth_array, mask, cam_K, image, obj_color=None, frameNumer = None, time_dict=None, cfg=None) -> o3d.geometry.PointCloud:
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    pre_open3d_start = time.perf_counter_ns()
    # Also remove points with invalid depth values
    mask = np.logical_and(mask, depth_array > 0)

    if mask.sum() == 0:
        pcd = o3d.geometry.PointCloud()
        return pcd
        
    height, width = depth_array.shape
    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)
    
    # Apply the mask, and unprojection is done only on the valid points
    masked_depth = depth_array[mask] # (N, )
    u = u[mask] # (N, )
    v = v[mask] # (N, )

    # Convert to 3D coordinates
    x = (u - cx) * masked_depth / fx
    y = (v - cy) * masked_depth / fy
    z = masked_depth

    convert_to_3d = time.perf_counter_ns()
    # time_dict['convert_to_3d_time'] += (convert_to_3d - pre_open3d_start)/1e6

    # Stack x, y, z coordinates into a 3D point cloud
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)

    stacking_time = time.perf_counter_ns()
    # time_dict['stacking_time'] += (stacking_time - convert_to_3d)/1e6
    
    # Perturb the points a bit to avoid colinearity (using cheap fixed pattern)
    
    # Use pre-computed perturbation pattern instead of expensive random generation
    # This is ~99% faster than np.random.normal() while still avoiding colinearity
    perturbation_pattern = np.array([
        [1e-3, -2e-3, 1e-3],
        [-2e-3, 1e-3, -1e-3], 
        [1e-3, 1e-3, -2e-3],
        [-1e-3, -1e-3, 1e-3],
        [2e-3, -1e-3, -1e-3]
    ], dtype=np.float32)
    
    # Apply pattern cyclically to points
    pattern_idx = np.arange(points.shape[0]) % len(perturbation_pattern)
    points += perturbation_pattern[pattern_idx]
        
    perturbing_time = time.perf_counter_ns()
    # time_dict['perturbing_time'] += (perturbing_time - stacking_time)/1e6
    

    if obj_color is None: # color using RGB
        # # Apply mask to image
        colors = image[mask] / 255.0
    else: # color using group ID
        # Use the assigned obj_color for all points
        colors = np.full(points.shape, obj_color)

    post_color_time = time.perf_counter_ns()
    # time_dict['color_time'] += (post_color_time - perturbing_time)/1e6
    
    if points.shape[0] == 0:
        import pdb; pdb.set_trace()
    pre_open3d_end = time.perf_counter_ns()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    post_open3d_start = time.perf_counter_ns()
    # time_dict['pre_open3d_time'] += (pre_open3d_end - pre_open3d_start)/1e6
    # time_dict['open3d_time'] += (post_open3d_start - pre_open3d_end)/1e6
    
    return pcd

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd

def process_pcd(pcd, cfg, run_dbscan=True, frameNumer = None, caller=None, dataset_type='replica', time_dict=None):  
    voxel_size = cfg.dataset_config.downsample_voxel_size
    downsample_start = time.perf_counter_ns()
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    

    # Approximate the number of points to be around 2000. 
    # Done soleley for performance. Added to improve the dataset performance. Shouldn't affect the iPad performance. 
    # Quality has not degraded becausee of this approximation.
    if experiment_config is None:
        raise ValueError("Experiment config is not set")
    object_based_downsampling = experiment_config.model.mapping.object_based_downsampling
    if dataset_type in ['replica', 'scannet', 'quest'] and object_based_downsampling:
        while len(pcd.points) > 2000:
            voxel_size *= cfg.dataset_config.mapping.object_based_downsampling
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pre_denoise_time = time.perf_counter_ns()
    # This is a step to remove the noise points in the point cloud. Quite expensive, so make sure that the number of points is not too large.
    if cfg.dbscan_remove_noise and run_dbscan:
        # print("Before dbscan:", len(pcd.points))
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=cfg.dbscan_eps, 
            min_points=cfg.dbscan_min_points
        )
        # print("After dbscan:", len(pcd.points))
    post_denoise_time = time.perf_counter_ns()
    # if time_dict is not None:
        # time_dict['denoise_time'] += (post_denoise_time - pre_denoise_time) / 1e6
        # time_dict['downsample_time'] += (pre_denoise_time - downsample_start) / 1e6
    return pcd

def get_bounding_box(cfg, pcd):
    if ("accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()

def merge_obj2_into_obj1(cfg, obj1, obj2, run_dbscan=True):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k == "history_idx":
                obj1[k] = obj1[k]  # Keep the initial history index
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft"]:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            else:
                # TODO: handle other types if needed in the future
                raise NotImplementedError
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd and bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], cfg, run_dbscan=run_dbscan, caller="merge_obj2_into_obj1")
    obj1['bbox'] = get_bounding_box(cfg, obj1['pcd'])
    obj1['bbox'].color = [0,1,0]
    
    # merge clip ft
    obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
                       obj2['clip_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    # merge text_ft
    obj2['text_ft'] = to_tensor(obj2['text_ft'], cfg.device)
    obj1['text_ft'] = to_tensor(obj1['text_ft'], cfg.device)
    obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
                       obj2['text_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)
    
    return obj1

def compute_overlap_matrix(cfg, objects: MapObjectList, time_dict=None):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor.
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object.
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i
    that are within a distance threshold of any point in point cloud j.
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    if n == 0:
        return overlap_matrix

    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]

    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    # Broad phase: single vectorized AABB overlap check over all N^2 pairs.
    # compute_3d_iou is already AABB-style under the hood (it uses each bbox's
    # min/max bound), but called from a Python double-loop it's ~20us * N^2.
    # Doing the same math via numpy broadcasting costs ~2ms for N=150 and kills
    # >95% of pairs before the FAISS kNN narrow phase. Toggle via cfg.
    bp_start = time.perf_counter_ns()
    if cfg.get('merge_aabb_broadphase', True):
        mins = np.empty((n, 3), dtype=np.float32)
        maxs = np.empty((n, 3), dtype=np.float32)
        for k, obj in enumerate(objects):
            mins[k] = np.asarray(obj['bbox'].get_min_bound())
            maxs[k] = np.asarray(obj['bbox'].get_max_bound())
        aabb_mask = np.all(maxs[:, None, :] >= mins[None, :, :], axis=2) & \
                    np.all(mins[:, None, :] <= maxs[None, :, :], axis=2)
        np.fill_diagonal(aabb_mask, False)
        candidate_pairs = np.argwhere(aabb_mask)
    else:
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        candidate_pairs = np.stack([ii.ravel(), jj.ravel()], axis=1)
        candidate_pairs = candidate_pairs[candidate_pairs[:, 0] != candidate_pairs[:, 1]]
    bp_end = time.perf_counter_ns()

    # Semantic prefilter: drop candidate pairs whose CLIP cosine similarity
    # is at or below cfg.merge_visual_sim_thresh. merge_overlap_objects
    # already requires visual_sim > thresh AND text_sim > thresh to merge,
    # so any pair we skip here would also have been skipped at the merge
    # gate — same merge decisions, just no wasted FAISS work. Single
    # matmul costs ~0.2 ms even at N=260.
    sem_start = time.perf_counter_ns()
    semantic_filtered = 0
    if cfg.get('merge_semantic_prefilter', True) and len(candidate_pairs) > 0:
        try:
            clip_fts = objects.get_stacked_values_torch('clip_ft')
            if clip_fts.dim() == 2 and clip_fts.shape[0] == n:
                clip_fts = F.normalize(clip_fts.float(), dim=1)
                cos_sim = clip_fts @ clip_fts.T
                ii_ = candidate_pairs[:, 0]
                jj_ = candidate_pairs[:, 1]
                sims = cos_sim[ii_, jj_].cpu().numpy()
                thresh = float(cfg.merge_visual_sim_thresh)
                keep = sims > thresh
                semantic_filtered = int((~keep).sum())
                candidate_pairs = candidate_pairs[keep]
        except Exception as e:
            # Defensive: if any object lacks clip_ft or shapes mismatch,
            # skip the prefilter rather than crash. The narrow phase will
            # produce a correct overlap_matrix either way.
            print(f"[compute_overlap_matrix] semantic prefilter skipped: {e}")
    sem_end = time.perf_counter_ns()

    # Narrow phase: only iterate over surviving pairs.
    narrow_start = time.perf_counter_ns()
    narrow_survivors = 0
    for i, j in candidate_pairs:
        box_i = objects[i]['bbox']
        box_j = objects[j]['bbox']

        # Kept for safety — with the AABB prefilter on, this will essentially
        # never reject a survivor, but is still correct when the filter is off.
        iou = compute_3d_iou(box_i, box_j)
        if iou == 0:
            continue

        D, I = indices[j].search(point_arrays[i], 1)
        overlap = (D < cfg.downsample_voxel_size ** 1.7).sum() # D is the squared distance
        overlap_matrix[i, j] = overlap / len(point_arrays[i])
        if overlap_matrix[i, j] > 0:
            narrow_survivors += 1
    narrow_end = time.perf_counter_ns()

    if time_dict is not None:
        time_dict['merge_n_objects'] = n
        # broadphase_survivors = AABB-only count (before semantic filter), so
        # the ratio against semantic_filtered is interpretable.
        time_dict['merge_broadphase_survivors'] = int(len(candidate_pairs)) + semantic_filtered
        time_dict['merge_semantic_filtered'] = semantic_filtered
        time_dict['merge_narrowphase_survivors'] = narrow_survivors
        time_dict['merge_broadphase_ms'] = (bp_end - bp_start) / 1e6
        time_dict['merge_semantic_ms'] = (sem_end - sem_start) / 1e6
        time_dict['merge_narrowphase_ms'] = (narrow_end - narrow_start) / 1e6

    return overlap_matrix

def compute_overlap_matrix_2set(cfg, objects_map: MapObjectList, objects_new: DetectionList) -> np.ndarray:
    '''
    compute pairwise overlapping between two set of objects in terms of point nearest neighbor. 
    objects_map is the existing objects in the map, objects_new is the new objects to be added to the map
    Suppose len(objects_map) = m, len(objects_new) = n
    Then we want to construct a matrix of size m x n, where the (i, j) entry is the ratio of points 
    in point cloud i that are within a distance threshold of any point in point cloud j.
    '''
    m = len(objects_map)
    n = len(objects_new)
    overlap_matrix = np.zeros((m, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_map = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_map] # m arrays
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map] # m indices
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, points_map):
        index.add(arr)
        
    points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new] # n arrays
        
    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    
    # try:
        # iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
    # except ValueError:
    bbox_map = []
    bbox_new = []
    for pcd in objects_map.get_values('pcd'):
        bbox_map.append(np.asarray(
            pcd.get_axis_aligned_bounding_box().get_box_points()))
    for pcd in objects_new.get_values('pcd'):
        bbox_new.append(np.asarray(
            pcd.get_axis_aligned_bounding_box().get_box_points()))
    bbox_map = torch.from_numpy(np.stack(bbox_map))
    bbox_new = torch.from_numpy(np.stack(bbox_new))
    
    iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)

    # Broad phase survivors: iterate only over pairs with non-trivial AABB
    # overlap. compute_iou_batch is already vectorized, so this just avoids a
    # Python-level M*N skip-loop and runs FAISS on the small surviving set.
    if cfg.get('merge_aabb_broadphase', True):
        candidate_pairs = np.argwhere(iou.numpy() >= 1e-6) if hasattr(iou, 'numpy') \
                          else np.argwhere(np.asarray(iou) >= 1e-6)
    else:
        ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
        candidate_pairs = np.stack([ii.ravel(), jj.ravel()], axis=1)

    for i, j in candidate_pairs:
        if iou[i, j] < 1e-6:
            continue

        D, I = indices[i].search(points_new[j], 1) # search new object j in map object i

        overlap = (D < cfg.downsample_voxel_size ** 1.7).sum() # D is the squared distance

        # Calculate the ratio of points within the threshold
        overlap_matrix[i, j] = overlap / len(points_new[j])

    return overlap_matrix

def _summarize_obj_texts(obj) -> str:
    # Compact description of an object's text labels (class_name list + caption dict values),
    # deduped while preserving order. Used for merge-event logging in merge_overlap_objects.
    parts = []
    for x in (obj.get('class_name') or []):
        if x and x not in parts:
            parts.append(str(x))
    cap = obj.get('caption') or {}
    for v in cap.values():
        if v and v not in parts:
            parts.append(str(v))
    return "[" + " | ".join(parts) + "]" if parts else "<no-text>"

def merge_overlap_objects(cfg, objects: MapObjectList, overlap_matrix: np.ndarray, history_map: dict):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]
    removed_objects = []
    edited_objects = []

    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    # Snapshot original text descriptions before any merging mutates objects in-place.
    obj_descs = [_summarize_obj_texts(obj) for obj in objects]
    # target_idx -> {'target': desc, 'sources': [desc, ...]}; chains hoist sources up.
    merge_events: dict = {}

    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, overlap_ratio):
        visual_sim = F.cosine_similarity(
            to_tensor(objects[i]['clip_ft']),
            to_tensor(objects[j]['clip_ft']),
            dim=0
        )
        text_sim = F.cosine_similarity(
            to_tensor(objects[i]['text_ft']),
            to_tensor(objects[j]['text_ft']),
            dim=0
        )
        if ratio > cfg.merge_overlap_thresh:
            if visual_sim > cfg.merge_visual_sim_thresh and \
                text_sim > cfg.merge_text_sim_thresh:
                if kept_objects[j]:
                    # Then merge object i into object j --- remove object i and edit object j
                    history_map[objects[i]['history_idx']] = None
                    removed_objects.append(objects[i]['history_idx'])
                    edited_objects.append(objects[j]['history_idx'])
                    # Record this merge before mutating: if i was previously a target, hoist its
                    # sources into j so the whole chain prints on one line.
                    sources_to_add = []
                    prior = merge_events.pop(int(i), None)
                    if prior:
                        sources_to_add.extend(prior['sources'])
                    sources_to_add.append(obj_descs[int(i)])
                    if int(j) in merge_events:
                        merge_events[int(j)]['sources'].extend(sources_to_add)
                    else:
                        merge_events[int(j)] = {'target': obj_descs[int(j)], 'sources': sources_to_add}
                    objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i], run_dbscan=True)
                    kept_objects[i] = False
        else:
            break
    
    # Remove the objects that have been merged
    new_objects = []
    for obj, keep in zip(objects, kept_objects):
        if keep:
            if obj['history_idx'] not in removed_objects:
                if history_map[obj['history_idx']] != len(new_objects):
                    history_map[obj['history_idx']] = len(new_objects)  
                    
            new_objects.append(obj)
        else:
            assert history_map[obj['history_idx']] == None, "The object should have been removed"
    # new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)

    return objects, removed_objects, edited_objects, history_map, merge_events

def denoise_objects(cfg, objects: MapObjectList):
    for i in range(len(objects)):
        og_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], cfg, run_dbscan=True, frameNumer=None ,caller="denoise_objects")
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = og_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(cfg, objects[i]['pcd'])
        objects[i]['bbox'].color = [0,1,0]
        
    return objects

def denoise_selected_objects(cfg, objects: MapObjectList, edited_objects_indices: list):
    for i in range(len(objects)):
        hist_idx = objects[i]['history_idx']
        if hist_idx not in edited_objects_indices:
            continue
        og_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], cfg, run_dbscan=True, frameNumer=None ,caller="denoise_selected_objects")
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = og_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(cfg, objects[i]['pcd'])
        objects[i]['bbox'].color = [0,1,0]
        
    return objects

def filter_objects(cfg, objects: MapObjectList, history_map: dict):
    # Remove the object that has very few points or viewed too few times
    removed_obj = []
    print("Before filtering:", len(objects))
    objects_to_keep = []
    for i,obj in enumerate(objects):
        if len(obj['pcd'].points) >= cfg.obj_min_points and obj['num_detections'] >= cfg.obj_min_detections:
            index_to_keep = len(objects_to_keep)    
            if history_map[obj['history_idx']] != index_to_keep:
                history_map[obj['history_idx']] = index_to_keep
            objects_to_keep.append(obj)
            
        else:
            history_map[obj['history_idx']] = None
            removed_obj.append(obj['history_idx'])
    objects = MapObjectList(objects_to_keep)
    print("After filtering:", len(objects))
    
    return objects, removed_obj, history_map

def merge_objects(cfg, objects: MapObjectList, history_map: dict, time_dict=None):
    removed_objects = []
    edited_objects = []

    if cfg.merge_overlap_thresh > 0:
        # Merge one object into another if the former is contained in the latter
        overlap_matrix = compute_overlap_matrix(cfg, objects, time_dict=time_dict)
        print("Before merging:", len(objects))
        objects, removed_objects, edited_objects, history_map, merge_events = merge_overlap_objects(cfg, objects, overlap_matrix, history_map=history_map)
        print("After merging:", len(objects))
        for ev in merge_events.values():
            print(f"  {ev['target']} <- {', '.join(ev['sources'])}")

    return objects, removed_objects, edited_objects, history_map

def filter_gobs(
    cfg: DictConfig,
    gobs: dict,
    image: np.ndarray,
    BG_CLASSES=BG_CLASSES,
    pipelined_mapping=True,
):
    # If no detection at all
    if len(gobs['xyxy']) == 0:
        return gobs

    # Ignore list (string-match): classes to drop entirely (never mapped,
    # never fused). Distinct from BG_CLASSES — BG classes are fused as a
    # single bg object when skip_bg is false; ignored classes are discarded.
    skip_ignored = bool(cfg.get('skip_ignored', False))
    ignore_classes = set(cfg.get('ignore_classes', []) or []) if skip_ignored else set()

    # Ignore filter (CLIP-similarity variant): drop masks whose CLIP image
    # embedding is close to any pre-encoded ignore-class text embedding.
    # Only fires when gobs arrives here with image_feats populated (pipelined
    # mapping path — inference.py stashes feats into gobs before sending to
    # the mapping queue). In the non-pipelined path (inference_pipeline.py),
    # filter_gobs runs in parallel with CLIP so image_feats is None here;
    # that path applies the CLIP-ignore filter post-hoc on each detection's
    # clip_ft AFTER attachment — see inference_consumer().
    clip_drop_mask = None
    if bool(cfg.get('skip_ignored_clip', False)):
        clip_drop_mask, max_sims_np = compute_clip_ignore_drops(gobs.get('image_feats'))
        if clip_drop_mask is not None and clip_drop_mask.any() and bool(cfg.get('debug_ignore_drops', False)):
            dropped_labels = [
                gobs['classes'][gobs['class_id'][i]]
                for i in range(len(clip_drop_mask)) if clip_drop_mask[i]
            ]
            dropped_sims = [round(float(max_sims_np[i]), 3)
                            for i in range(len(clip_drop_mask)) if clip_drop_mask[i]]
            print(f"[filter_gobs] CLIP-ignored {int(clip_drop_mask.sum())} mask(s): "
                  f"labels={dropped_labels} max_sims={dropped_sims}")

    # Filter out the objects based on various criteria
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]

        # CLIP-similarity ignore filter (vectorized above)
        if clip_drop_mask is not None and clip_drop_mask[mask_idx]:
            continue

        # Drop ignored classes outright (e.g. humans for SemanticXR demos)
        if ignore_classes and class_name in ignore_classes:
            continue

        # SKip masks that are too small
        if gobs['mask'][mask_idx].sum() < max(cfg.mask_area_threshold, 10):
            continue

        # Skip the BG classes
        if cfg.skip_bg and class_name in BG_CLASSES:
            continue
        
        # Skip the non-background boxes that are too large
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if bbox_area > cfg.max_bbox_area_ratio * image_area:
                # print(f"Skipping {class_name} with area {bbox_area} > {cfg.max_bbox_area_ratio} * {image_area}")
                continue
            
        # Skip masks with low confidence
        if gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < cfg.mask_conf_threshold:
                continue
        
        idx_to_keep.append(mask_idx)
    
    for k in gobs.keys():
        if isinstance(gobs[k], str) or k == "classes": # Captions
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            if (k == "image_feats" or k == "text_feats") and(  pipelined_mapping== False):
                continue
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}, where k is {k},pipe = {pipelined_mapping}")
    
    return gobs

def resize_gobs(
    gobs,
    image
):
    n_masks = len(gobs['xyxy'])

    new_mask = []
    
    for mask_idx in range(n_masks):
        # TODO: rewrite using interpolation/resize in numpy or torch rather than cv2
        mask = gobs['mask'][mask_idx]
        if mask.shape != image.shape[:2]:
            # Rescale the xyxy coordinates to the image shape
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            x1 = round(x1 * image.shape[1] / mask.shape[1])
            y1 = round(y1 * image.shape[0] / mask.shape[0])
            x2 = round(x2 * image.shape[1] / mask.shape[1])
            y2 = round(y2 * image.shape[0] / mask.shape[0])
            gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]
            
            # Reshape the mask to the image shape
            mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
            new_mask.append(mask)

    if len(new_mask) > 0:
        gobs['mask'] = np.asarray(new_mask)
        
    return gobs
import time
def gobs_to_detection_list(
    cfg, 
    image, 
    depth_array,
    cam_K, 
    idx, 
    gobs, 
    trans_pose = None,
    class_names = None,
    BG_CLASSES=BG_CLASSES,
    color_path = None,
    pipelined_mapping=True,
    dataset_type='replica',
    time_dict=None,
):
    '''
    Return a DetectionList object from the gobs
    All object are still in the camera frame. 
    '''
    global experiment_config
    if experiment_config is None:
        experiment_config = get_config()
    # cfg = experiment_config.mapping
    
    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()
    
    pcd_creation_time = []
    pcd_process_time = []
    resize_filter_start = time.perf_counter_ns()
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image, BG_CLASSES, pipelined_mapping=pipelined_mapping)
    
    
    if len(gobs['xyxy']) == 0:
        return fg_detection_list, bg_detection_list, None
    
    # Compute the containing relationship among all detections and subtract fg from bg objects
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    gobs['mask'] = mask_subtract_contained(xyxy, mask)
    resize_filter_end = time.perf_counter_ns()
    # time_dict['resize_filter_time'] = (resize_filter_end - resize_filter_start)/1e6
    idx_to_keep = []    
    n_masks = len(gobs['xyxy'])
    time_dict['pre_open3d_time'] = 0
    time_dict['open3d_time'] = 0
    time_dict['convert_to_3d_time'] = 0
    time_dict['stacking_time'] = 0
    time_dict['perturbing_time'] = 0
    time_dict['color_time'] = 0
    
    for mask_idx in range(n_masks):
        local_class_id = gobs['class_id'][mask_idx]
        mask = gobs['mask'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        global_class_id = -1 if class_names is None else class_names.index(class_name)
        pcd_start = time.perf_counter_ns()
        # make the pcd and color it
        camera_object_pcd = create_object_pcd(
            depth_array,
            mask,
            cam_K,
            image,
            obj_color = None,
            frameNumer = idx,
            time_dict=time_dict,
            cfg=cfg,
        )
        pcd_end = time.perf_counter_ns()
        
        # It at least contains 5 points
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5): 
            continue
        
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        
        # get largest cluster, filter out noise 
        global_object_pcd = process_pcd(global_object_pcd, cfg, frameNumer=idx, caller="gobs_to_detection_list", dataset_type=dataset_type)
        
        pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox.color = [0,1,0]
        process_pcd_time = time.perf_counter_ns()
        if pcd_bbox.volume() < 1e-6:
            continue
        
        # Treat the detection in the same way as a 3D object
        # Store information that is enough to recover the detection
        detected_object = {
            'image_idx' : [idx],                             # idx of the image
            'mask_idx' : [mask_idx],                         # idx of the mask/detection
            'color_path' : [color_path],                     # path to the RGB image
            'class_name' : [class_name],                         # global class id for this detection
            'class_id' : [global_class_id],                         # global class id for this detection
            'num_detections' : 1,                            # number of detections in this object
            'mask': [mask],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [mask.sum()],
            'contain_number': [None],                          # This will be computed later
            "inst_color": np.random.rand(3),                 # A random color used for this segment instance
            'is_background': class_name in BG_CLASSES,
            
            # These are for the entire 3D object
            'pcd': global_object_pcd,
            'bbox': pcd_bbox,
            'history_idx': None,
        }
        if pipelined_mapping:
            detected_object['clip_ft'] = to_tensor(gobs['image_feats'][mask_idx])
            detected_object['text_ft'] = to_tensor(gobs['text_feats'][mask_idx])
        idx_to_keep.append(mask_idx)
        if class_name in BG_CLASSES:
            bg_detection_list.append(detected_object)
        else:
            fg_detection_list.append(detected_object)
        pcd_creation_time.append((pcd_end - pcd_start)/1e6)
        pcd_process_time.append((process_pcd_time - pcd_end)/1e6)
    # print(f"[MAPPING SERvER]\t\t FrameNumber: {idx} PCD creation time: ", np.sum(pcd_creation_time), "ms  PCD process time = ",  np.sum(pcd_process_time), "ms")
    time_dict['pcd_creation_time'] = np.sum(pcd_creation_time)
    time_dict['pcd_process_time'] = np.sum(pcd_process_time)
    return fg_detection_list, bg_detection_list, idx_to_keep

def transform_detection_list(
    detection_list: DetectionList,
    transform: torch.Tensor,
    deepcopy = False,
):
    '''
    Transform the detection list by the given transform
    
    Args:
        detection_list: DetectionList
        transform: 4x4 torch.Tensor
        
    Returns:
        transformed_detection_list: DetectionList
    '''
    transform = to_numpy(transform)
    
    if deepcopy:
        detection_list = copy.deepcopy(detection_list)
    
    for i in range(len(detection_list)):
        detection_list[i]['pcd'] = detection_list[i]['pcd'].transform(transform)
        detection_list[i]['bbox'] = detection_list[i]['bbox'].rotate(transform[:3, :3], center=(0, 0, 0))
        detection_list[i]['bbox'] = detection_list[i]['bbox'].translate(transform[:3, 3])
        # detection_list[i]['bbox'] = detection_list[i]['pcd'].get_oriented_bounding_box(robust=True)
    
    return detection_list

def precompute_xy_maps(width, height, fx, fy, cx, cy, dtype=np.float32):
    """Cache this per resolution/intrinsics."""
    u = np.arange(width, dtype=dtype)
    v = np.arange(height, dtype=dtype)
    uu, vv = np.meshgrid(u, v)  # (H, W)
    xu = (uu - cx) / fx         # (H, W)
    yv = (vv - cy) / fy         # (H, W)
    return xu, yv

def masks_to_labels(masks, H, W):
    """
    masks: np.bool_ array of shape (N, H, W)
    Returns int32 label image with [-1]=background, [k]=object id.
    Assumes masks are disjoint after your mask_subtract_contained().
    """
    labels = np.full((H, W), -1, dtype=np.int32)
    # If overlaps could still exist, decide priority here (last wins below):
    for k in range(len(masks)):
        m = masks[k]
        if m.dtype != np.bool_:
            m = m.astype(bool, copy=False)
        labels[m] = k
    return labels

def build_objects_points_and_colors(image, depth, xu, yv, labels,
                                    add_noise=True, sigma=4e-3,
                                    obj_color=None,
                                    time_dict=None,
                                    cfg=None,
                                    masks=None,
                                    max_depth_m=None):
    """
    Do frame-wide unprojection, color once, optional jitter once.
    Returns:
      - points_by_obj: dict[obj_id] -> np.ndarray (Ni, 3) float32
      - colors_by_obj: dict[obj_id] -> np.ndarray (Ni, 3) float32
    """
    t0 = time.perf_counter_ns()

    valid = (depth > 0) & (labels >= 0)
    # Optional far-depth gate. ``max_depth_m`` is the session-level cap stamped
    # by the client (or pulled from config). None = no cap.
    if max_depth_m is not None and max_depth_m > 0:
        valid &= (depth <= max_depth_m)
    if not np.any(valid):
        print("No valid points")
        # if time_dict is not None:
        #     time_dict['convert_to_3d_time'] += 0.0
        #     time_dict['perturbing_time'] += 0.0
        #     time_dict['color_time'] += 0.0
        return {}, {}

    Z = depth[valid].astype(np.float32, copy=False)
    X = xu[valid].astype(np.float32, copy=False) * Z
    Y = yv[valid].astype(np.float32, copy=False) * Z

    pts = np.column_stack((X, Y, Z))  # (N, 3) float32

    t1 = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['convert_to_3d_time'] += (t1 - t0) / 1e6

    # Optional single jitter (tiny; skip unless you need robust OBB frequently)
    
    # n0 = time.perf_counter_ns()
    # noise = np.random.normal(0.0, sigma, size=pts.shape).astype(np.float32)
    # pts += noise
    # n1 = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['perturbing_time'] += (n1 - n0) / 1e6

        # Perturb the points a bit to avoid colinearity (using cheap fixed pattern)
    # if cfg is None or not hasattr(cfg, 'determinism') or not hasattr(cfg.determinism, 'disable_point_noise') or not cfg.determinism.disable_point_noise:
        # Use pre-computed perturbation pattern instead of expensive random generation
        # This is ~99% faster than np.random.normal() while still avoiding colinearity
    perturbation_pattern = np.array([
        [1e-3, -2e-3, 1e-3],
        [-2e-3, 1e-3, -1e-3], 
        [1e-3, 1e-3, -2e-3],
        [-1e-3, -1e-3, 1e-3],
        [2e-3, -1e-3, -1e-3]
    ], dtype=np.float32)
    
    # Apply pattern cyclically to points
    pattern_idx = np.arange(pts.shape[0]) % len(perturbation_pattern)
    pts += perturbation_pattern[pattern_idx]
    
    perturbing_time = time.perf_counter_ns()
    # time_dict['perturbing_time'] += (perturbing_time - t1)/1e6
   

    c0 = time.perf_counter_ns()
    if obj_color is None:
        cols = (image[valid].astype(np.float32, copy=False) / 255.0)
    else:
        cols = np.full((pts.shape[0], 3), obj_color, dtype=np.float32)
    c1 = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['color_time'] += (c1 - c0) / 1e6

    before_binning = time.perf_counter_ns()
    # # Bin points to objects via labels
    # lbl = labels[valid].ravel()
    # order = np.argsort(lbl, kind='stable')
    # lbl_sorted = lbl[order]
    # pts_sorted = pts[order]
    # cols_sorted = cols[order]
    # arg_srot_time = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['arg_sort_time'] = (arg_srot_time - before_binning) / 1e6

    # uniq, counts = np.unique(lbl_sorted, return_counts=True)
    # offsets = np.cumsum(counts)
    # unique_time = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['unique_time'] = (unique_time - arg_srot_time) / 1e6

    # points_by_obj, colors_by_obj = {}, {}
    # start = 0
    # for obj_id, end in zip(uniq.tolist(), offsets.tolist()):
    #     points_by_obj[obj_id] = pts_sorted[start:end]
    #     colors_by_obj[obj_id] = cols_sorted[start:end]
    #     start = end





        # ---- Bin to objects without sort/unique ----
    # Map between full-frame linear indices and rows in pts/cols
    valid_idx = np.flatnonzero(valid.ravel())        # (N,)
    # Avoid dtype churn in the loop
    valid_idx = np.ascontiguousarray(valid_idx, dtype=np.int64)

    points_by_obj, colors_by_obj = {}, {}

    # Tight loop: for each mask, select only among valid pixels (cheap)
    # NOTE: do not create a big (K x N) temporary; index per-mask to keep memory low.
    for k in range(masks.shape[0]):
        m_flat = masks[k].ravel()                    # view, no copy
        sel = m_flat[valid_idx]                      # boolean over N valid pixels
        if not np.any(sel):
            continue
        rows = np.flatnonzero(sel)                   # indices into pts/cols
        # Slice views; Open3D will copy later anyway
        points_by_obj[k] = pts[rows]
        colors_by_obj[k] = cols[rows]

    


    after_binning = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['binning_time'] = (after_binning - before_binning) / 1e6
        # time_dict['loop_time'] = (after_binning - unique_time) / 1e6

    return points_by_obj, colors_by_obj

def frame_to_object_pcds(depth_array, masks, cam_K, image,
                         add_noise=False, sigma=4e-3,
                         precomputed_xy=None,
                         time_dict=None,
                         cfg=None,
                         max_depth_m=None):
    """
    Returns a dict: obj_id -> open3d PointCloud
    - masks: np.bool_ array of shape (N, H, W) for the *filtered* detections
    - Maintains your timing keys in time_dict (convert_to_3d_time, perturbing_time, color_time, pre_open3d_time, open3d_time)
    """
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    H, W = depth_array.shape

    if precomputed_xy is None:
        xu, yv = precompute_xy_maps(W, H, fx, fy, cx, cy)
    else:
        xu, yv = precomputed_xy  # (H, W) each

    labels = masks_to_labels(masks, H, W)

    pre_o3d_start = time.perf_counter_ns()
    # single-pass points/colors
    pts_by_obj, cols_by_obj = build_objects_points_and_colors(
        image=image,
        depth=depth_array,
        xu=xu, yv=yv,
        labels=labels,
        add_noise=add_noise,
        sigma=sigma,
        obj_color=None,
        time_dict=time_dict,
        cfg=cfg,
        masks=masks,
        max_depth_m=max_depth_m,
    )

    # Wrap in Open3D
    pc_dict = {}
    pre_o3d_t0 = time.perf_counter_ns()
    
    for obj_id, pts in pts_by_obj.items():
        cols = cols_by_obj[obj_id]
        pts  = np.ascontiguousarray(pts,  dtype=np.float64)
        cols = np.ascontiguousarray(cols, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        pc_dict[obj_id] = pcd
    pre_o3d_t1 = time.perf_counter_ns()
    # if time_dict is not None:
    #     # amortize pre/post O3D attribution like before
    #     time_dict['open3d_time'] += (pre_o3d_t1 - pre_o3d_t0) / 1e6
    #     time_dict['pre_open3d_time'] += (pre_o3d_t0 - pre_o3d_start) / 1e6


    return pc_dict


def gobs_to_detection_list_optimized(
    cfg,
    image,
    depth_array,
    cam_K,
    idx,
    gobs,
    trans_pose = None,
    class_names = None,
    BG_CLASSES=BG_CLASSES,
    color_path = None,
    pipelined_mapping=True,
    dataset_type='replica',
    time_dict=None,
    max_depth_m=None,
):
    global experiment_config
    if experiment_config is None:
        experiment_config = get_config()
    # cfg = experiment_config.mapping
    """
    Entry point: now uses a single frame-wide unprojection.
    """
    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()

    # Defensive: segmentation can occasionally produce a malformed mask —
    # most often a 2D (H,W) instead of (1,H,W) for a single detection, or
    # an empty 1D array when xyxy/mask counts disagree. resize_gobs and the
    # per-mask unproject loop both assume gobs['mask'] is (N,H,W) and crash
    # deep inside on ``mask.shape[1]`` (IndexError) when it isn't, taking
    # the whole mapping consumer down. Drop the frame here instead — the
    # next frame is almost always fine.
    masks = gobs.get('mask')
    n_xyxy = len(gobs['xyxy'])
    if n_xyxy > 0 and (masks is None
                       or getattr(masks, 'ndim', 0) != 3
                       or masks.shape[0] != n_xyxy):
        print(f"[mapping] WARN: dropping frame {idx} — malformed masks "
              f"(ndim={getattr(masks, 'ndim', None)}, "
              f"shape={getattr(masks, 'shape', None)}, n_xyxy={n_xyxy})")
        return fg_detection_list, bg_detection_list, []

    pcd_creation_times_ms = []
    pcd_process_times_ms = []

    filter_start = time.perf_counter_ns()

    t0 = time.perf_counter_ns()
    gobs = resize_gobs(gobs, image)
    if time_dict is not None:
        time_dict['resize_masks_ms'] = (time.perf_counter_ns() - t0) / 1e6

    t0 = time.perf_counter_ns()
    gobs = filter_gobs(cfg, gobs, image, BG_CLASSES, pipelined_mapping=pipelined_mapping)
    if time_dict is not None:
        time_dict['drop_masks_ms'] = (time.perf_counter_ns() - t0) / 1e6

    if len(gobs['xyxy']) == 0:
        if time_dict is not None:
            time_dict['subtract_contained_ms'] = 0.0
            time_dict['n_contained_pairs'] = 0
            time_dict['gobs_filter_ms'] = (time.perf_counter_ns() - filter_start) / 1e6
        return fg_detection_list, bg_detection_list, []

    # Subtract containment as before
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    t0 = time.perf_counter_ns()
    gobs['mask'] = mask_subtract_contained(xyxy, mask, time_dict=time_dict)
    if time_dict is not None:
        time_dict['subtract_contained_ms'] = (time.perf_counter_ns() - t0) / 1e6
        # gobs_filter_ms = resize + class/conf/area filter + containment subtract.
        # Pure frame-local cost (no global-map references).
        time_dict['gobs_filter_ms'] = (time.perf_counter_ns() - filter_start) / 1e6

    # Init timing buckets (same keys you used before)
    if time_dict is not None:
        time_dict['pre_open3d_time'] = 0.0
        time_dict['open3d_time'] = 0.0
        time_dict['convert_to_3d_time'] = 0.0
        time_dict['perturbing_time'] = 0.0
        time_dict['color_time'] = 0.0
        time_dict['denoise_time'] = 0.0
        time_dict['downsample_time'] = 0.0

    n_masks = len(gobs['xyxy'])
    if time_dict is not None:
        time_dict['n_masks'] = n_masks
    if n_masks == 0:
        if time_dict is not None:
            time_dict['n_unprojected'] = 0
            time_dict['n_detections_kept'] = 0
        return fg_detection_list, bg_detection_list, []

    # ---- Frame-wide unprojection ----
    make_pcds_start = time.perf_counter_ns()
    masks = np.asarray(gobs['mask'], dtype=np.bool_)  # (N, H, W)
    obj_pcds = frame_to_object_pcds(
        depth_array=depth_array,
        masks=masks,
        cam_K=cam_K,
        image=image,
        add_noise=False,            # ← usually safe to skip; set True if you see OBB degeneracy
        sigma=4e-3,
        precomputed_xy=None,
        time_dict=time_dict,
        cfg=cfg,
        max_depth_m=max_depth_m,
    )
    make_pcds_end = time.perf_counter_ns()
    pcd_creation_times_ms.append((make_pcds_end - make_pcds_start)/1e6)
    if time_dict is not None:
        # n_unprojected = masks that produced >=1 valid 3D point from the depth
        # buffer. Gap between n_masks and n_unprojected = masks with no/bad depth.
        time_dict['n_unprojected'] = len(obj_pcds)

    idx_to_keep = []

    # ---- Per-object post-processing (downsample/DBSCAN/OBB etc.) ----
    for mask_idx in range(n_masks):
        if mask_idx not in obj_pcds:
            continue  # e.g., mask produced zero valid depth points

        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        global_class_id = -1 if class_names is None else class_names.index(class_name)

        camera_object_pcd = obj_pcds[mask_idx]

        # Minimum points check
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5):
            continue

        # Transform to world if provided
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd

        # Denoise/downsample as before
        pcd_post_start = time.perf_counter_ns()
        global_object_pcd = process_pcd(global_object_pcd, cfg, frameNumer=idx,
                                        caller="gobs_to_detection_list_batched",
                                        dataset_type=dataset_type,
                                        time_dict=time_dict)

        # BBox
        pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox.color = [0, 1, 0]

        pcd_post_end = time.perf_counter_ns()
        pcd_process_times_ms.append((pcd_post_end - pcd_post_start)/1e6)

        # Reject degenerate bbox
        if pcd_bbox.volume() < 1e-6:
            continue

        # Build detection dict (unchanged shape)
        detected_object = {
            'image_idx' : [idx],
            'mask_idx' : [mask_idx],
            'color_path' : [color_path],
            'class_name' : [class_name],
            'class_id' : [global_class_id],
            'num_detections' : 1,
            'mask': [gobs['mask'][mask_idx]],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [gobs['mask'][mask_idx].sum()],
            'contain_number': [None],
            "inst_color": np.random.rand(3),
            'is_background': class_name in BG_CLASSES,

            'pcd': global_object_pcd,
            'bbox': pcd_bbox,
            'history_idx': None,
        }
        if pipelined_mapping:
            detected_object['clip_ft'] = to_tensor(gobs['image_feats'][mask_idx])
            detected_object['text_ft'] = to_tensor(gobs['text_feats'][mask_idx])

        idx_to_keep.append(mask_idx)
        if class_name in BG_CLASSES:
            bg_detection_list.append(detected_object)
        else:
            fg_detection_list.append(detected_object)

    # Log timings like before
    # print(f"[MAPPING SERVER]\t FrameNumber: {idx} PCD creation(batch) = {pcd_creation_times_ms[0]:.2f} ms PCD postprocess = {pcd_process_times_ms[0]:.2f} ms")
    if time_dict is not None:
        time_dict['pcd_creation_time'] = np.sum(pcd_creation_times_ms)
        time_dict['pcd_process_time']  = np.sum(pcd_process_times_ms)
        # n_detections_kept = survivors of the per-mask loop (min_points,
        # degenerate bbox, etc.). This is the count that actually enters the map.
        time_dict['n_detections_kept'] = len(idx_to_keep)

    return fg_detection_list, bg_detection_list, idx_to_keep
