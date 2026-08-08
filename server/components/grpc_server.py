# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC server component for SLAM system."""

import json
import os
import time
from pathlib import Path

import numpy as np
from typing import Optional, Iterator, Any
from PIL import Image
import grpc

from config.settings import Config, get_config
from .video_processor import VideoProcessor
from .inference_service import InferenceService

# Import generated protobuf classes from parent package
from server import xr_service_pb2, xr_service_pb2_grpc

from slam.datasets.quest import build_depth_in_rgb_frame


# Meta XR environment depth: raw bytes are R16_SFloat (2 B/pixel, a single
# half-float per pixel storing (1 - ndc_depth)). Metric depth is sensor_near / R.
# Kept in sync with debug_server/unity_grpc_server.py::decode_depth in the
# SemanticXR-Quest-Client repo (the reference implementation).
_QUEST_DEPTH_BYTES_PER_PIXEL = 2


def _decode_quest_depth_bytes(depth_bytes: bytes, depth_w: int, depth_h: int,
                              inv_depth_factor: float) -> Optional[np.ndarray]:
    """Decode a Quest R16_SFloat depth payload to a (H, W) metric float32 array.

    ``inv_depth_factor`` is the ZBufferParams.x value from the proto
    (UpstreamSyncMessage_quest.depth_near_z), which equals ``-2 * sensor_near``
    for the infinite-far case. Returns None on size mismatch or when no
    conversion params are available.
    """
    expected = depth_w * depth_h * _QUEST_DEPTH_BYTES_PER_PIXEL
    if len(depth_bytes) != expected:
        return None
    r = np.frombuffer(depth_bytes, dtype=np.float16).astype(np.float32).reshape(depth_h, depth_w)
    if inv_depth_factor == 0:
        return None
    sensor_near = -inv_depth_factor / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        metric = np.where(r > 1e-6, sensor_near / r, 0.0).astype(np.float32)
    return np.clip(metric, 0.0, 100.0)


class SLAMGRPCServer(xr_service_pb2_grpc.XrServiceServicer):
    """Clean, focused gRPC server for SLAM operations."""
    
    def __init__(self, 
                 inference_queue,
                 config: Optional[Config] = None):
        """Initialize SLAM gRPC server.
        
        Args:
            inference_queue: Queue for inference requests
            config: Configuration object
        """
        self.config = config if config else get_config()
        
        # Initialize components
        self.video_processor = VideoProcessor(self.config)
        self.inference_service = InferenceService(inference_queue, self.config)
        
        # Dataset saving configuration from config. Per-type directories
        # (datasets/quest/dataset_<N>/, datasets/ipad/dataset_<N>/) are created
        # lazily on first frame, since the gRPC server doesn't know whether
        # incoming frames will be Quest or iPad until it sees one.
        self.data_dump_enabled = self.config.dataset.enabled
        self._capture_roots: dict = {}  # dataset_type ('quest'/'ipad') -> capture dir path

    def _resolve_client_fps(self, request) -> Optional[int]:
        """Translate the proto's int ``fps`` field to Optional[int].

        Wire convention:
          * positive int → return that value (client-stamped FPS)
          * 0 / unset / negative → ``None`` ("client did not specify"); the
            consumer falls back to ``frame_processing.default_client_fps``.

        See InferenceService.note_client_fps for the per-session freeze.
        """
        client_value = int(getattr(request, 'fps', 0) or 0)
        return client_value if client_value > 0 else None

    def _resolve_max_depth_m(self, request) -> Optional[float]:
        """Translate the proto's float ``max_depth_m`` field to Optional[float].

        Wire convention (matches xr_service.proto):
          * 0.0 / unset → ``None`` ("client did not specify"); the consumer
            falls back to the per-dataset YAML cfg.max_depth_m.
          * non-zero    → returned as-is. Negative is an explicit "no cap"
            request from the client; positive is the cap in meters. Final
            interpretation happens in
            ``slam.utils.mapping_utils.resolve_session_max_depth``.
        """
        client_value = float(getattr(request, 'max_depth_m', 0.0) or 0.0)
        return None if client_value == 0.0 else client_value

    def _get_or_create_capture_dir(self, dataset_type: str) -> str:
        """Resolve (and lazily mkdir) the per-type capture directory.

        Layout: ``<dataset.output_directory>/<dataset_type>/dataset_<N>/`` with
        ``N`` auto-incremented within the type subdir. With
        ``auto_increment_dirs=False`` we just use ``<output_directory>/<dataset_type>/``
        as a single fixed sink. Result is cached so all frames in the session
        write to the same scene root.
        """
        cached = self._capture_roots.get(dataset_type)
        if cached is not None:
            return cached

        type_root = os.path.join(self.config.dataset.output_directory, dataset_type)
        os.makedirs(type_root, exist_ok=True)
        if self.config.dataset.auto_increment_dirs:
            existing = [d for d in os.listdir(type_root) if d.startswith('dataset_')]
            numbers = [int(d.split('_', 1)[1]) for d in existing if d.split('_', 1)[1].isdigit()]
            next_n = max(numbers, default=-1) + 1
            scene_root = os.path.join(type_root, f'dataset_{next_n}')
        else:
            scene_root = type_root
        os.makedirs(scene_root, exist_ok=True)
        self._capture_roots[dataset_type] = scene_root
        print(f"📦 Capture dir for {dataset_type}: {scene_root}")
        return scene_root
    
    def _save_dataset_files(self, 
                           client_frame_number: int,
                           image_pil: Image.Image,
                           depth_array: np.ndarray, 
                           pose_string: str,
                           is_dataset: bool = False) -> None:
        """Save frame data for dataset if enabled.
        
        Args:
            client_frame_number: Frame number from client
            image_pil: PIL Image object
            depth_array: Depth data
            pose_string: Pose information
        """
        if not self.data_dump_enabled:
            return

        try:
            scene_root = self._get_or_create_capture_dir('ipad')
            results_dir = os.path.join(scene_root, self.config.dataset.results_subdir)
            os.makedirs(results_dir, exist_ok=True)

            image_pil.save(os.path.join(results_dir, f"frame{client_frame_number:06d}.jpg"))

            if is_dataset:
                depth_image = Image.fromarray(depth_array.astype(np.uint16))
                depth_image.save(os.path.join(results_dir, f"depth{client_frame_number:06d}.png"))
            else:
                depth_array = depth_array.reshape((144, 256))
                if not np.any(depth_array != 0):
                    print(f"************************** Depth array is all zeros for frame {client_frame_number}, skipping save **************************")
                    return
                np.save(os.path.join(results_dir, f"depth{client_frame_number:06d}.npy"), depth_array)
                depth_image = Image.fromarray(depth_array.astype(np.uint16))
                depth_image.save(os.path.join(results_dir, f"depth{client_frame_number:06d}.png"))

            with open(os.path.join(scene_root, "traj.txt"), 'a') as f:
                f.write(f"{pose_string}\n")

        except Exception as e:
            print(f"Error saving dataset files: {e}")
    
    def _quest_target_resolution(self) -> tuple:
        """Return the (width, height) processing resolution for Quest frames.

        Reads QUEST.yaml's camera_params.image_width/height (640x640 today). The
        inference pipeline downstream uses these via slam.utils.mapping_utils.setup,
        so we resample here to match and avoid internal rescale in the dataset
        preprocessor.
        """
        cached = getattr(self, '_quest_target_wh', None)
        if cached is not None:
            return cached
        import yaml
        quest_yaml = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'slam', 'datasets', 'QUEST.yaml',
        )
        with open(quest_yaml, 'r') as f:
            cfg = yaml.safe_load(f)
        cp = cfg['camera_params']
        self._quest_target_wh = (int(cp['image_width']), int(cp['image_height']))
        return self._quest_target_wh

    def _save_quest_replay_frame(
        self,
        frame_number: int,
        image_pil_native: Image.Image,
        depth_native: np.ndarray,
        request,
        server_timestamp_ns: int,
    ) -> None:
        """Save one Quest frame in the replay-ready scene layout.

        Writes (under the lazily-resolved Quest capture dir
        ``datasets/quest/dataset_<N>/`` as the scene root):
          * ``decoded_jpg/frame_NNNNNN.jpg`` — RGB at NATIVE resolution
            (pre-resize; the inference pipeline will downsample to its
            processing resolution at replay time, identical to live).
          * ``depth/depth_NNNNNN.npy`` — float32 metric depth at NATIVE depth
            resolution (PRE-alignment; ``build_depth_in_rgb_frame`` runs at
            replay time, identical to live).
          * ``meta/meta_NNNNNN.json`` — frame number + ``rgb_camera_pose`` and
            ``depth_pose`` saved EXACTLY as received from the proto (OpenXR
            right-handed Y-up, c2w). The OpenGL→OpenCV flip happens in
            ``QuestDataset.load_poses`` at consumption, not on disk. Also
            includes the three timestamps (rgb=BOOTTIME, depth=MONOTONIC,
            server=this server's perf_counter_ns at arrival).
          * ``intrinsics.json`` — written once on first call, at the scene
            root. Holds RGB and depth intrinsics + sizes; cy is stored as
            ``cy_yup`` (viewport convention, measured from image bottom).

        This layout is consumed by ``slam.datasets.quest.load_quest_meta`` /
        ``load_quest_intrinsics`` and by ``server/main.py::
        _process_quest_dataset`` for offline replay.
        """
        if image_pil_native is None or depth_native is None:
            return
        try:
            scene_root = self._get_or_create_capture_dir('quest')
            jpg_dir = os.path.join(scene_root, 'decoded_jpg')
            depth_dir = os.path.join(scene_root, 'depth')
            meta_dir = os.path.join(scene_root, 'meta')
            os.makedirs(jpg_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(meta_dir, exist_ok=True)

            image_pil_native.save(os.path.join(jpg_dir, f'frame_{frame_number:06d}.jpg'))
            np.save(os.path.join(depth_dir, f'depth_{frame_number:06d}.npy'), depth_native)

            meta = {
                'frame_number': int(frame_number),
                'rgb_camera_pose': np.array(request.rgb_camera_pose, dtype=np.float32).reshape(4, 4).tolist(),
                'depth_pose': np.array(request.depth_pose, dtype=np.float32).reshape(4, 4).tolist(),
                'rgb_timestamp_ns': int(getattr(request, 'rgb_timestamp_ns', 0)),
                'depth_timestamp_ns': int(getattr(request, 'depth_timestamp_ns', 0)),
                'server_timestamp_ns': int(server_timestamp_ns),
            }
            with open(os.path.join(meta_dir, f'meta_{frame_number:06d}.json'), 'w') as f:
                json.dump(meta, f)

            intrinsics_path = os.path.join(scene_root, 'intrinsics.json')
            if not os.path.exists(intrinsics_path):
                doc = {
                    'convention': 'OpenXR right-handed, Y-up; cy_yup measured from image bottom',
                    'rgb': {
                        'fx': float(request.intrinsics.fx),
                        'fy': float(request.intrinsics.fy),
                        'cx': float(request.intrinsics.cx),
                        'cy_yup': float(request.intrinsics.cy),
                        'image_width': int(request.image_width),
                        'image_height': int(request.image_height),
                    },
                    'depth': {
                        'fx': float(request.depth_intrinsics.fx),
                        'fy': float(request.depth_intrinsics.fy),
                        'cx': float(request.depth_intrinsics.cx),
                        'cy_yup': float(request.depth_intrinsics.cy),
                        'image_width': int(request.depth_width),
                        'image_height': int(request.depth_height),
                        'depth_near_z': float(request.depth_near_z),
                    },
                }
                with open(intrinsics_path, 'w') as f:
                    json.dump(doc, f, indent=2)
        except Exception as e:
            print(f"Error saving Quest replay frame {frame_number}: {e}")

    def _process_frame_request(self, request, is_dataset: bool = False, is_quest: bool = False) -> Optional[xr_service_pb2.VideoStatus]:
        """Process a single frame request.

        Args:
            request: gRPC request object
            is_dataset: Whether this is a dataset request
            is_quest: Whether this is a Quest request

        Returns:
            Upload response or None if processing failed
        TODO:: check if this is correct!!!!!!!!!!!!! Very last minute change!!!!!!!!!!!!! (Seems ok for now? -- Jan 6th 2026)
        """
        try:
            # Extract frame data (same for both message types)
            client_frame_number = request.image.frame_number
            client_timestamp = getattr(request, 'timestamp_ns', request.image.timestamp_us * 1000)  # Convert µs to ns if needed
            server_timestamp = time.perf_counter_ns()

            # Lock the per-session client FPS BEFORE the skip check, so the
            # very first frame's gap math uses the client-stamped value rather
            # than the YAML default. No-op after frame 1 of the session.
            self.inference_service.note_client_fps(self._resolve_client_fps(request))

            # Check if frame should be processed
            if not self.inference_service.should_process_frame(client_frame_number):
                return xr_service_pb2.VideoStatus(success=True)

            # Pull the encoded video bytes. The proto field is named ``data_h265``
            # for historical reasons: iPad / dataset streams are actually H.264,
            # while Meta Quest streams H.265 (HEVC). Choose the decoder by codec.
            video_data = None
            if hasattr(request.image, 'data_h265') and request.image.data_h265:
                video_data = request.image.data_h265

            image_pil = None
            image_array = None
            if video_data:
                codec = 'h265' if is_quest else 'h264'
                processed_frame, is_valid = self.video_processor.process_video_frame(video_data, codec=codec)

                if not is_valid:
                    print(f"Frame {client_frame_number} rejected (quality check)")
                    return xr_service_pb2.VideoStatus(success=False)

                # processed_frame should be the decoded image
                if processed_frame is not None:
                    if isinstance(processed_frame, Image.Image):
                        image_pil = processed_frame
                        image_array = np.array(processed_frame)
                    elif isinstance(processed_frame, np.ndarray):
                        image_array = processed_frame
                        image_pil = Image.fromarray(processed_frame)

            # Convert depth data - different formats for dataset vs regular vs quest
            if is_quest:
                # Quest streaming protocol (UpstreamSyncMessage_quest): depth is a
                # raw R16G16B16A16_SFloat texture; we decode to metric meters and
                # then resample into the RGB camera frame at processing resolution
                # (identical alignment math to local-file mode).
                #
                # Streaming path is end-to-end: this branch builds the aligned
                # (RGB, depth, rgb_camera_pose) tuple from the raw proto, and
                # the inference worker's QuestDataset initializes its K from
                # QUEST.yaml's streaming_defaults.rgb (device-fixed Quest 3S
                # intrinsics — same values the proto reports frame-by-frame).
                depth_native = _decode_quest_depth_bytes(
                    request.depth, request.depth_width, request.depth_height,
                    request.depth_near_z,
                )
                if depth_native is None:
                    print(f"Frame {client_frame_number} rejected (quest depth size mismatch: "
                          f"{len(request.depth)} B for {request.depth_width}x{request.depth_height})")
                    return xr_service_pb2.VideoStatus(success=False)
            elif is_dataset:
                # Dataset format: bytes depth
                depth_array = np.frombuffer(request.depth, dtype=np.uint16)

                scaling_factor = request.scaling_factor
                # reshape the depth array to the height and width of the image
                depth_array = depth_array.reshape((image_array.shape[0]//scaling_factor, image_array.shape[1]//scaling_factor))
            else:
                # Regular format: repeated float depthArr - TODO:: check if this is correct!!!!!!!!!!!!!
                depth_array = np.array(request.depthArr, dtype=np.float32)

            # Extract pose data - different formats for dataset vs regular vs quest
            if is_quest:
                # Use rgb_camera_pose (physical RGB sensor with lens offset) — NOT
                # the deprecated top-level ``pose`` field (which is the head pose).
                rgb_pose_values = list(request.rgb_camera_pose)
                depth_pose_values = list(request.depth_pose)
                if len(rgb_pose_values) != 16 or len(depth_pose_values) != 16:
                    print(f"Frame {client_frame_number} rejected (quest pose missing: "
                          f"rgb_camera_pose={len(rgb_pose_values)} depth_pose={len(depth_pose_values)})")
                    return xr_service_pb2.VideoStatus(success=False)
                pose_data = [client_frame_number] + rgb_pose_values
            elif is_dataset:
                # Dataset format: repeated float pose
                pose_values = list(request.pose)
                pose_data = [client_frame_number] + pose_values
            else:
                # Regular format: PoseData object
                pose_data = [
                    client_frame_number,
                    request.pose.pose0_0, request.pose.pose0_1, request.pose.pose0_2, request.pose.pose0_3,
                    request.pose.pose1_0, request.pose.pose1_1, request.pose.pose1_2, request.pose.pose1_3,
                    request.pose.pose2_0, request.pose.pose2_1, request.pose.pose2_2, request.pose.pose2_3,
                    request.pose.pose3_0, request.pose.pose3_1, request.pose.pose3_2, request.pose.pose3_3
                ]

            # Build metadata
            metadata = {'is_dataset': is_dataset, 'is_quest': is_quest}
            if is_quest:
                # Capture raw inputs for replay BEFORE alignment/resize so the
                # offline pipeline can reproduce live behavior end-to-end. Saved
                # ``image_pil`` is at native RGB resolution and ``depth_native``
                # is at native depth resolution (pre ``build_depth_in_rgb_frame``).
                if self.data_dump_enabled:
                    self._save_quest_replay_frame(
                        client_frame_number, image_pil, depth_native, request, server_timestamp,
                    )
                # Resample depth into RGB camera frame and downsample RGB to the
                # same processing resolution (see slam/datasets/quest.py for the
                # same treatment in local-file mode).
                target_w, target_h = self._quest_target_resolution()
                meta_for_align = {
                    'rgb_camera_pose': np.array(rgb_pose_values, dtype=np.float32).reshape(4, 4),
                    'depth_pose': np.array(depth_pose_values, dtype=np.float32).reshape(4, 4),
                    'fx': request.intrinsics.fx,
                    'fy': request.intrinsics.fy,
                    'cx': request.intrinsics.cx,
                    'cy': request.intrinsics.cy,
                    'depth_fx': request.depth_intrinsics.fx,
                    'depth_fy': request.depth_intrinsics.fy,
                    'depth_cx': request.depth_intrinsics.cx,
                    'depth_cy': request.depth_intrinsics.cy,
                    'image_width': request.image_width,
                    'image_height': request.image_height,
                }
                depth_array = build_depth_in_rgb_frame(
                    meta_for_align, depth_native, target_h, target_w,
                )
                if image_pil is not None:
                    image_pil = image_pil.convert('RGB').resize(
                        (target_w, target_h), Image.LANCZOS,
                    )
                    image_array = np.array(image_pil)
                metadata['intrinsics'] = {
                    'fx': request.intrinsics.fx,
                    'fy': request.intrinsics.fy,
                    'cx': request.intrinsics.cx,
                    'cy': request.intrinsics.cy,
                }
                metadata['image_width'] = request.image_width
                metadata['image_height'] = request.image_height

            # Save dataset files if enabled (iPad / dataset paths only —
            # Quest captures via ``_save_quest_replay_frame`` earlier in this
            # function, before alignment/resize, so it can preserve native
            # inputs for replay).
            if self.data_dump_enabled and not is_quest:
                pose_string = " ".join(map(str, pose_data[1:]))  # Skip frame number for pose string
                self._save_dataset_files(client_frame_number, image_pil, depth_array, pose_string, is_dataset)

            # Submit to inference
            success = self.inference_service.submit_inference_request(
                image_pil=image_pil,
                image_array=image_array,
                depth_array=depth_array,
                pose=pose_data,
                client_frame_number=client_frame_number,
                client_timestamp=client_timestamp,
                server_timestamp=server_timestamp,
                metadata=metadata,
                max_depth_m=self._resolve_max_depth_m(request),
            )

            if success:
                return xr_service_pb2.VideoStatus(success=True)
            else:
                return xr_service_pb2.VideoStatus(success=False)

        except Exception as e:
            error_msg = f"Error processing frame: {e}"
            print(error_msg)
            return xr_service_pb2.VideoStatus(success=False)
    
    def UploadSyncMessage(self, 
                         request_iterator, 
                         context):
        """Handle streaming upload requests from clients.
        
        Args:
            request_iterator: Iterator of upload requests
            context: gRPC context
            
        Returns:
            Single VideoStatus response
        """
        success_count = 0
        total_count = 0
        
        for request in request_iterator:
            total_count += 1
            response = self._process_frame_request(request, is_dataset=False)
            if response and response.success:
                success_count += 1
        
        # Return a single VideoStatus indicating overall success
        overall_success = success_count == total_count and total_count > 0
        return xr_service_pb2.VideoStatus(success=overall_success)
    
    def UploadSyncMessage_quest(self,
                               request_iterator,
                               context):
        """Handle streaming upload requests from Meta Quest devices.

        On client disconnect (iterator exhausted or RPC cancelled) we push a
        ``scene_completion`` signal onto the inference queue so the inference
        worker runs dump_semantic_map and finalizes performance logs the same
        way local-file mode does via _send_completion_signal. This is what
        turns ``--save_map`` into an actually-saved map for streaming sessions.

        Args:
            request_iterator: Iterator of UpstreamSyncMessage_quest requests
            context: gRPC context

        Returns:
            Single VideoStatus response
        """
        print(f"\n\n\n GRPC Server: UploadSyncMessage_quest called")
        success_count = 0
        total_count = 0

        try:
            for request in request_iterator:
                total_count += 1
                if total_count == 1:
                    print(f"Quest first frame: image={request.image_width}x{request.image_height}, "
                          f"depth={request.depth_width}x{request.depth_height}, "
                          f"intrinsics=({request.intrinsics.fx:.1f}, {request.intrinsics.fy:.1f}, "
                          f"{request.intrinsics.cx:.1f}, {request.intrinsics.cy:.1f})")
                response = self._process_frame_request(request, is_quest=True)
                if response and response.success:
                    success_count += 1
        finally:
            self._send_scene_completion_on_disconnect(total_count)

        overall_success = success_count == total_count and total_count > 0
        print(f"Quest processing completed: {success_count}/{total_count} frames successful")
        return xr_service_pb2.VideoStatus(success=overall_success)

    def _allocate_next_run_dir(self) -> Optional[Path]:
        """Compute and create the run_<N+1>/ dir for the next disconnect cycle.

        Mirrors the auto-increment logic in server/main.py so each client
        disconnect produces a fresh run dir for the *next* connection. Only
        applies in live (gRPC) mode, which is always the case here.

        Returns the Path on success; None if anything goes wrong (we don't
        want a one-off filesystem hiccup to take the server down).
        """
        try:
            dataset_type = os.environ.get('SLAM_DATASET_TYPE', 'quest').lower()
            live_root = Path(self.config.dataset.live_output_directory) / dataset_type
            live_root.mkdir(parents=True, exist_ok=True)
            existing = [
                d.name for d in live_root.iterdir()
                if d.is_dir() and d.name.startswith('run_')
            ]
            next_n = max(
                (int(d.split('_', 1)[1]) for d in existing if d.split('_', 1)[1].isdigit()),
                default=-1,
            ) + 1
            new_dir = live_root / f'run_{next_n}'
            new_dir.mkdir(parents=True, exist_ok=True)
            return new_dir
        except Exception as e:
            print(f"⚠️  [GRPC] Could not allocate next run dir: {e}")
            return None

    def _send_scene_completion_on_disconnect(self, total_count: int) -> None:
        """Push a ``scene_completion`` dict onto the inference queue.

        Called from the ``finally`` of UploadSyncMessage_quest. Each disconnect:
          * Allocates a fresh ``run_<N+1>/`` directory for the *next* session.
          * Updates SLAM_RUN_OUTPUT_DIR for parent-side reads.
          * Clears dataset-capture cache so the next session opens a fresh
            ``dataset_<N+1>/``.
          * Emits the signal with ``run_output_dir`` so each downstream worker
            (inference, mapping, viz) can rebind its perf manager + dump
            targets after flushing the just-completed session.

        Scene name comes from SLAM_SCENE_NAME (set by main.py at startup).
        """
        if total_count == 0:
            return
        scene_name = os.environ.get('SLAM_SCENE_NAME', 'quest')

        new_run_dir = self._allocate_next_run_dir()
        if new_run_dir is not None:
            os.environ['SLAM_RUN_OUTPUT_DIR'] = str(new_run_dir.resolve())

        # Drop dataset-capture cache so next session writes to dataset_<N+1>/.
        self._capture_roots = {}

        # Reset the inference-service frame cursors. Without this, the next
        # client's frame numbers (which often restart at 0) compare against the
        # previous session's last_frame_index and every frame is dropped by
        # should_process_frame() as "too close".
        self.inference_service.reset_session()

        signal = {
            'type': 'scene_completion',
            'scene_name': scene_name,
            'timestamp': time.perf_counter_ns(),
        }
        if new_run_dir is not None:
            signal['run_output_dir'] = str(new_run_dir.resolve())

        print(f"📤 [GRPC] Client disconnected after {total_count} frames; "
              f"emitting scene_completion for '{scene_name}'"
              + (f" (next run: {new_run_dir})" if new_run_dir is not None else ""))
        try:
            self.inference_service.inference_queue.put(signal, timeout=5.0)
        except Exception as e:
            print(f"⚠️  [GRPC] Could not enqueue scene_completion: {e}")

    def UploadSyncMessage_dataset(self,
                                 request_iterator,
                                 context):
        """Handle streaming upload requests for dataset processing.
        
        Args:
            request_iterator: Iterator of upload requests  
            context: gRPC context
            
        Returns:
            Single VideoStatus response
        """
        print(f"\n\n\n GRPC Server: UploadSyncMessage_dataset called")  
        success_count = 0
        total_count = 0
        
        for request in request_iterator:
            total_count += 1
            print(f"Processing frame {request.image.frame_number}")
            response = self._process_frame_request(request, is_dataset=True)
            if response and response.success:
                success_count += 1
        
        # Return a single VideoStatus indicating overall success
        overall_success = success_count == total_count and total_count > 0
        print(f"Dataset processing completed: {success_count}/{total_count} frames successful")
        return xr_service_pb2.VideoStatus(success=overall_success)
                
    def get_server_stats(self) -> dict:
        """Get current server statistics.
        
        Returns:
            Dictionary with server stats
        """
        inference_stats = self.inference_service.get_processing_stats()
        
        return {
            'inference': inference_stats,
            'video_processor': {
                'temp_dir': self.video_processor.temp_output_dir,
                'sharpness_threshold': self.video_processor.sharpness_threshold
            },
            'dataset_saving': {
                'enabled': self.data_dump_enabled,
                'capture_roots': dict(self._capture_roots),
            }
        }