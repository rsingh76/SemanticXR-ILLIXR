"""Test Quest point cloud reconstruction from RGB-D-Pose without inference.

Verifies that depth unprojection + pose transforms produce a sensible 3D scene.
Usage:
    conda run -n semanticxr python evaluation/test_quest_reconstruction.py \
        --data_dir datasets/quest_data_collection --stride 5
"""

import argparse
import glob
import os
import sys
import re

import numpy as np
from natsort import natsorted
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slam.datasets.quest import load_quest_meta


def unproject_depth(depth, fx, fy, cx, cy):
    """Unproject depth image to 3D point cloud in camera frame."""
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    return points


def test_reconstruction(data_dir, stride=5, max_depth=5.0):
    """Build a colored point cloud from Quest data and save as PLY."""
    import open3d as o3d

    meta_files = natsorted(glob.glob(os.path.join(data_dir, 'meta', 'meta_*.json')))

    # Discover valid frames
    frames = []
    for meta_path in meta_files:
        frame_num = int(re.search(r'meta_(\d+)', meta_path).group(1))
        rgb_path = os.path.join(data_dir, 'decoded_jpg', f'frame_{frame_num:06d}.jpg')
        depth_path = os.path.join(data_dir, 'depth', f'depth_{frame_num:06d}.npy')
        if os.path.exists(rgb_path) and os.path.exists(depth_path):
            frames.append((frame_num, rgb_path, depth_path, meta_path))

    frames = frames[::stride]
    print(f"Processing {len(frames)} frames (stride={stride})")

    # Get intrinsics from first frame
    first_meta = load_quest_meta(frames[0][3])
    fx, fy = first_meta['fx'], first_meta['fy']
    cx, cy = first_meta['cx'], first_meta['cy']
    img_w, img_h = first_meta['image_width'], first_meta['image_height']
    depth_w, depth_h = first_meta['depth_width'], first_meta['depth_height']

    print(f"Image: {img_w}x{img_h}, Depth: {depth_w}x{depth_h}")
    print(f"Intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    # Scale intrinsics to depth resolution
    fx_d = fx * depth_w / img_w
    fy_d = fy * depth_h / img_h
    cx_d = cx * depth_w / img_w
    cy_d = cy * depth_h / img_h
    print(f"Scaled intrinsics (depth res): fx={fx_d:.2f} fy={fy_d:.2f} cx={cx_d:.2f} cy={cy_d:.2f}")

    all_points = []
    all_colors = []
    camera_positions = []

    for i, (frame_num, rgb_path, depth_path, meta_path) in enumerate(frames):
        meta = load_quest_meta(meta_path)
        pose = meta['rgb_camera_pose']  # 4x4 c2w matrix

        depth = np.load(depth_path)  # (320, 320) float32 meters
        rgb = np.array(Image.open(rgb_path))  # (1280, 1280, 3)

        # Resize RGB to depth resolution for coloring
        rgb_small = np.array(Image.open(rgb_path).resize((depth_w, depth_h), Image.BILINEAR))

        # Unproject at depth resolution
        pts_cam = unproject_depth(depth, fx_d, fy_d, cx_d, cy_d)  # (H, W, 3)
        colors = rgb_small.astype(np.float64) / 255.0

        # Filter by depth
        valid = (depth > 0.1) & (depth < max_depth) & ~np.isnan(depth)
        pts_cam_valid = pts_cam[valid]  # (N, 3)
        colors_valid = colors[valid]    # (N, 3)

        if len(pts_cam_valid) == 0:
            print(f"  Frame {frame_num}: no valid points")
            continue

        # Transform to world frame using pose (c2w)
        # Test multiple pose conventions
        pts_world_raw = (pose[:3, :3] @ pts_cam_valid.T).T + pose[:3, 3]

        all_points.append(pts_world_raw)
        all_colors.append(colors_valid)
        camera_positions.append(pose[:3, 3])

        if i % 10 == 0:
            print(f"  Frame {frame_num}: {len(pts_cam_valid)} points, "
                  f"cam pos: [{pose[0,3]:.2f}, {pose[1,3]:.2f}, {pose[2,3]:.2f}]")

    # Combine all points
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    camera_positions = np.array(camera_positions)

    print(f"\nTotal points: {len(all_points)}")
    print(f"Scene extent: X[{all_points[:,0].min():.2f}, {all_points[:,0].max():.2f}] "
          f"Y[{all_points[:,1].min():.2f}, {all_points[:,1].max():.2f}] "
          f"Z[{all_points[:,2].min():.2f}, {all_points[:,2].max():.2f}]")

    # Save as PLY
    output_dir = os.path.join(data_dir, 'reconstruction_test')
    os.makedirs(output_dir, exist_ok=True)

    # Raw pose (no convention conversion)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    pcd = pcd.voxel_down_sample(0.02)

    ply_path = os.path.join(output_dir, 'reconstruction_raw_pose.ply')
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"\nSaved: {ply_path} ({len(pcd.points)} points after voxelization)")

    # Also test with Y/Z negation (what QuestDataset.load_poses does)
    all_points_yz = []
    for i, (frame_num, rgb_path, depth_path, meta_path) in enumerate(frames):
        meta = load_quest_meta(meta_path)
        pose = meta['rgb_camera_pose'].copy()
        pose[:3, 1] *= -1
        pose[:3, 2] *= -1

        depth = np.load(depth_path)
        pts_cam = unproject_depth(depth, fx_d, fy_d, cx_d, cy_d)
        valid = (depth > 0.1) & (depth < max_depth) & ~np.isnan(depth)
        pts_cam_valid = pts_cam[valid]

        pts_world = (pose[:3, :3] @ pts_cam_valid.T).T + pose[:3, 3]
        all_points_yz.append(pts_world)

    all_points_yz = np.vstack(all_points_yz)
    pcd_yz = o3d.geometry.PointCloud()
    pcd_yz.points = o3d.utility.Vector3dVector(all_points_yz)
    pcd_yz.colors = o3d.utility.Vector3dVector(all_colors)
    pcd_yz = pcd_yz.voxel_down_sample(0.02)

    ply_path_yz = os.path.join(output_dir, 'reconstruction_yz_negate.ply')
    o3d.io.write_point_cloud(ply_path_yz, pcd_yz)
    print(f"Saved: {ply_path_yz} ({len(pcd_yz.points)} points after voxelization)")

    # Also test with inverted pose (what iPad does: np.linalg.inv)
    all_points_inv = []
    for i, (frame_num, rgb_path, depth_path, meta_path) in enumerate(frames):
        meta = load_quest_meta(meta_path)
        pose = np.linalg.inv(meta['rgb_camera_pose'])
        pose[:3, 1] *= -1
        pose[:3, 2] *= -1

        depth = np.load(depth_path)
        pts_cam = unproject_depth(depth, fx_d, fy_d, cx_d, cy_d)
        valid = (depth > 0.1) & (depth < max_depth) & ~np.isnan(depth)
        pts_cam_valid = pts_cam[valid]

        pts_world = (pose[:3, :3] @ pts_cam_valid.T).T + pose[:3, 3]
        all_points_inv.append(pts_world)

    all_points_inv = np.vstack(all_points_inv)
    pcd_inv = o3d.geometry.PointCloud()
    pcd_inv.points = o3d.utility.Vector3dVector(all_points_inv)
    pcd_inv.colors = o3d.utility.Vector3dVector(all_colors)
    pcd_inv = pcd_inv.voxel_down_sample(0.02)

    ply_path_inv = os.path.join(output_dir, 'reconstruction_inv_yz_negate.ply')
    o3d.io.write_point_cloud(ply_path_inv, pcd_inv)
    print(f"Saved: {ply_path_inv} ({len(pcd_inv.points)} points after voxelization)")

    # Save camera trajectory
    traj_path = os.path.join(output_dir, 'camera_trajectory.txt')
    np.savetxt(traj_path, camera_positions, fmt='%.4f')
    print(f"Saved camera trajectory: {traj_path}")

    print(f"\nOpen in MeshLab/CloudCompare to inspect. Compare the 3 PLY files to find correct convention.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--max_depth', type=float, default=5.0)
    args = parser.parse_args()

    test_reconstruction(args.data_dir, args.stride, args.max_depth)
