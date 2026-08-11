# SPDX-FileCopyrightText: Copyright (c) 2026 Rahul Singh, University of Illinois Urbana-Champaign <rahuls10@illinois.edu>
# SPDX-License-Identifier: Apache-2.0

"""Validate Quest collected data by loading and visualizing RGB-D-Pose frames.

Usage:
    conda run -n semanticxr python evaluation/validate_quest_data.py --data_dir datasets/quest_data_collection

    # To also run through the SLAM dataset pipeline (getItems):
    conda run -n semanticxr python evaluation/validate_quest_data.py --data_dir datasets/quest_data_collection --test_pipeline
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from natsort import natsorted
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slam.datasets.quest import load_quest_meta as parse_meta_file


def discover_frames(data_dir):
    """Discover all valid frames that have matching RGB, depth, and meta files."""
    meta_files = natsorted(glob.glob(os.path.join(data_dir, 'meta', 'meta_*.json')))
    frames = []

    for meta_path in meta_files:
        frame_num = int(re.search(r'meta_(\d+)', meta_path).group(1))
        rgb_path = os.path.join(data_dir, 'decoded_jpg', f'frame_{frame_num:06d}.jpg')
        depth_npy_path = os.path.join(data_dir, 'depth', f'depth_{frame_num:06d}.npy')
        depth_png_path = os.path.join(data_dir, 'depth_png', f'depth_{frame_num:06d}.png')

        meta = parse_meta_file(meta_path)

        has_rgb = os.path.exists(rgb_path)
        has_depth = os.path.exists(depth_npy_path)

        if has_rgb and has_depth:
            frames.append({
                'frame_num': frame_num,
                'rgb_path': rgb_path,
                'depth_npy_path': depth_npy_path,
                'depth_png_path': depth_png_path,
                'meta_path': meta_path,
                'meta': meta,
            })

    return frames


def validate_data(data_dir, save_vis=True, test_pipeline=False):
    """Main validation: load all frames, check consistency, optionally visualize."""
    frames = discover_frames(data_dir)
    print(f"Found {len(frames)} complete frames (RGB + depth + meta)")

    if not frames:
        print("ERROR: No complete frames found!")
        return False

    # Check consistency across frames
    first_meta = frames[0]['meta']
    print(f"\nDataset summary:")
    print(f"  Image size: {first_meta['image_width']}x{first_meta['image_height']}")
    print(f"  Depth size: {first_meta['depth_width']}x{first_meta['depth_height']}")
    print(f"  Intrinsics: fx={first_meta['fx']:.4f} fy={first_meta['fy']:.4f} "
          f"cx={first_meta['cx']:.4f} cy={first_meta['cy']:.4f}")
    print(f"  Frame range: {frames[0]['frame_num']} - {frames[-1]['frame_num']}")

    # Validate each frame
    depth_stats = []
    pose_positions = []
    errors = []

    for i, frame in enumerate(frames):
        meta = frame['meta']
        frame_num = frame['frame_num']

        # Check intrinsics consistency
        if (meta['fx'] != first_meta['fx'] or meta['fy'] != first_meta['fy'] or
            meta['cx'] != first_meta['cx'] or meta['cy'] != first_meta['cy']):
            errors.append(f"Frame {frame_num}: intrinsics changed!")

        # Load and validate depth
        depth = np.load(frame['depth_npy_path'])
        expected_shape = (meta['depth_height'], meta['depth_width'])
        if depth.shape != expected_shape:
            errors.append(f"Frame {frame_num}: depth shape {depth.shape} != expected {expected_shape}")

        depth_stats.append({
            'frame': frame_num,
            'min': depth.min(),
            'max': depth.max(),
            'mean': depth.mean(),
            'nonzero_pct': np.count_nonzero(depth) / depth.size * 100,
        })

        # Load and validate RGB
        rgb = Image.open(frame['rgb_path'])
        if rgb.size != (meta['image_width'], meta['image_height']):
            errors.append(f"Frame {frame_num}: RGB size {rgb.size} != meta {meta['image_width']}x{meta['image_height']}")

        # Collect pose positions for trajectory
        pose = meta.get('rgb_camera_pose')
        if pose is not None and pose.shape == (4, 4):
            pose_positions.append(pose[:3, 3])
        else:
            errors.append(f"Frame {frame_num}: invalid pose shape")

    # Report
    print(f"\nValidation: {len(errors)} errors found")
    for e in errors[:10]:
        print(f"  - {e}")

    depth_mins = [s['min'] for s in depth_stats]
    depth_maxs = [s['max'] for s in depth_stats]
    depth_means = [s['mean'] for s in depth_stats]
    print(f"\nDepth stats across all frames:")
    print(f"  Min depth:  {min(depth_mins):.4f} - {max(depth_mins):.4f}")
    print(f"  Max depth:  {min(depth_maxs):.4f} - {max(depth_maxs):.4f}")
    print(f"  Mean depth: {min(depth_means):.4f} - {max(depth_means):.4f}")
    print(f"  Nonzero:    {min(s['nonzero_pct'] for s in depth_stats):.1f}% - {max(s['nonzero_pct'] for s in depth_stats):.1f}%")

    positions = np.array(pose_positions)
    print(f"\nTrajectory extent:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

    if save_vis:
        vis_dir = os.path.join(data_dir, 'validation_output')
        os.makedirs(vis_dir, exist_ok=True)
        _save_visualizations(frames, positions, depth_stats, vis_dir)
        print(f"\nVisualizations saved to {vis_dir}/")

    if test_pipeline:
        _test_slam_pipeline(frames, first_meta)

    return len(errors) == 0


def _save_visualizations(frames, positions, depth_stats, vis_dir):
    """Save sample frames and trajectory visualization."""
    # Sample frames: first, middle, last
    sample_indices = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]

    for idx in sample_indices:
        frame = frames[idx]
        rgb = np.array(Image.open(frame['rgb_path']))
        depth = np.load(frame['depth_npy_path'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(rgb)
        axes[0].set_title(f"RGB - Frame {frame['frame_num']}")
        axes[0].axis('off')

        im = axes[1].imshow(depth, cmap='turbo')
        axes[1].set_title(f"Depth - min={depth.min():.3f}m max={depth.max():.3f}m")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='meters')

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'sample_frame_{frame["frame_num"]:06d}.png'), dpi=100)
        plt.close()

    # Trajectory plot (top-down XZ)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(positions[:, 0], positions[:, 2], 'b-', linewidth=0.5)
    axes[0].plot(positions[0, 0], positions[0, 2], 'go', markersize=8, label='Start')
    axes[0].plot(positions[-1, 0], positions[-1, 2], 'ro', markersize=8, label='End')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Z (m)')
    axes[0].set_title('Top-Down Trajectory (XZ)')
    axes[0].legend()
    axes[0].set_aspect('equal')

    axes[1].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=0.5)
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_title('Front View (XY)')
    axes[1].set_aspect('equal')

    frame_nums = [s['frame'] for s in depth_stats]
    axes[2].plot(frame_nums, [s['mean'] for s in depth_stats], 'b-', label='mean')
    axes[2].fill_between(frame_nums, [s['min'] for s in depth_stats],
                         [s['max'] for s in depth_stats], alpha=0.2, label='min-max')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Depth (m)')
    axes[2].set_title('Depth Stats Over Time')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'trajectory_and_depth.png'), dpi=100)
    plt.close()


def _test_slam_pipeline(frames, meta):
    """Instantiate QuestDataset against the on-disk scene and push a few frames."""
    print("\n--- Testing SLAM Pipeline ---")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import yaml
    import torch
    from slam.datasets.quest import QuestDataset

    quest_yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'slam', 'datasets', 'QUEST.yaml')
    with open(quest_yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    basedir = os.path.dirname(frames[0]['meta_path'])
    dataset = QuestDataset(
        config_dict=config_dict,
        basedir=basedir,
        desired_height=640,
        desired_width=640,
        device='cpu',
        dtype=torch.float,
    )

    test_frames = [frames[0], frames[len(frames) // 2], frames[-1]]
    for frame in test_frames:
        rgb = np.array(Image.open(frame['rgb_path']))
        depth = np.load(frame['depth_npy_path'])
        pose = [frame['frame_num']] + frame['meta']['rgb_camera_pose'].flatten().tolist()

        color_t, depth_t, intrinsics_t, pose_t = dataset.getItems(rgb, depth, pose)

        print(f"  Frame {frame['frame_num']}:")
        print(f"    color:      {color_t.shape} {color_t.dtype} range [{color_t.min():.1f}, {color_t.max():.1f}]")
        print(f"    depth:      {depth_t.shape} {depth_t.dtype} range [{depth_t.min():.4f}, {depth_t.max():.4f}]")
        print(f"    intrinsics: {intrinsics_t.shape}")
        print(f"    pose:       {pose_t.shape} det={torch.det(pose_t[:3,:3]).item():.4f}")

    print("\nPipeline test PASSED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Quest collected data')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to quest_data_collection directory')
    parser.add_argument('--test_pipeline', action='store_true',
                        help='Also test through QuestDataset pipeline')
    parser.add_argument('--no_vis', action='store_true',
                        help='Skip saving visualizations')
    args = parser.parse_args()

    success = validate_data(args.data_dir, save_vis=not args.no_vis, test_pipeline=args.test_pipeline)
    sys.exit(0 if success else 1)
