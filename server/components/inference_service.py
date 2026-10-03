# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference service component for SLAM server."""

import time
import multiprocessing
import traceback
from typing import Optional, Tuple, Any
from queue import Queue
import numpy as np
from PIL import Image

from config.settings import Config, get_config


class InferenceService:
    """Handles inference coordination and queue management."""
    
    def __init__(self, 
                 inference_queue: multiprocessing.Queue,
                 config: Optional[Config] = None):
        """Initialize inference service.
        
        Args:
            inference_queue: Queue for sending inference requests
            config: Configuration object, defaults to global config
        """
        self.inference_queue = inference_queue
        self.config = config if config else get_config()
        
        # Frame tracking from config
        self.last_frame_index = self.config.frame_processing.initial_frame_index
        self.tick_num = 0

        # FPS control. ``target_fps`` is the server-side processing budget
        # (config-only, applies cross-session). ``client_fps`` is read from
        # the upstream proto's ``fps`` field on the FIRST frame of each
        # session, frozen on ``_session_client_fps``, and reset by
        # ``reset_session`` on disconnect. Until the freeze fires (or if the
        # client sends 0/unset), ``should_process_frame`` falls back to the
        # YAML default.
        self.target_fps = self.config.server.target_fps
        self._default_client_fps = self.config.frame_processing.default_client_fps
        self._session_client_fps: Optional[int] = None
        
    def should_process_frame(self, frame_number: int) -> bool:
        """Determine if frame should be processed based on FPS target.

        Args:
            frame_number: Current frame number

        Returns:
            True if frame should be processed
        """
        # Skip frames that are too close to last processed frame
        frame_diff = frame_number - self.last_frame_index
        client_fps = self._session_client_fps or self._default_client_fps
        min_frame_gap = max(1, int(client_fps / self.target_fps))

        return frame_diff >= min_frame_gap

    def note_client_fps(self, client_fps: Optional[int]) -> None:
        """Lock the per-session client FPS on the first non-None call.

        Called by the gRPC layer once per upstream frame before
        ``should_process_frame``. The first call after ``reset_session()``
        with a positive value freezes ``_session_client_fps``; subsequent
        calls in the same session are no-ops (matches the "read once per
        session" contract the headset relies on for max_depth_m).
        ``None`` means "client did not stamp the field" — in that case we
        leave the session value unset and ``should_process_frame`` falls
        through to the YAML default for this frame.
        """
        if self._session_client_fps is not None:
            return
        if client_fps is None or client_fps <= 0:
            return
        self._session_client_fps = int(client_fps)
        print(f"[GRPC]\t\t client_fps (session): {self._session_client_fps}")

    def reset_session(self) -> None:
        """Reset per-session frame cursors.

        Called on client disconnect so the next session's frame numbers (which
        often restart at 0 for a freshly-connected client) aren't compared
        against the previous session's last_frame_index — which would make
        should_process_frame() drop every frame as "too close". Also clears
        ``_session_client_fps`` so the next client's first-frame stamp wins
        again.
        """
        self.last_frame_index = self.config.frame_processing.initial_frame_index
        self.tick_num = 0
        self._session_client_fps = None

    def submit_inference_request(self,
                                image_pil: Image.Image,
                                image_array: np.ndarray,
                                depth_array: np.ndarray,
                                pose: list,
                                client_frame_number: int,
                                client_timestamp: int,
                                server_timestamp: int,
                                metadata: dict,
                                max_depth_m: Optional[float] = None) -> bool:
        """Submit inference request to processing queue.
        
        Args:
            image_pil: PIL Image object
            image_array: Image as numpy array
            depth_array: Depth data as numpy array  
            pose: Camera pose information
            client_frame_number: Frame number from client
            client_timestamp: Timestamp from client
            server_timestamp: Server-side timestamp
            metadata: Additional metadata
            
        Returns:
            True if request was successfully queued
        """
        if self.inference_queue.full():
            print(f"⚠️  [GRPC→INFERENCE] Queue full! Dropping frame {client_frame_number}")
            return False
        
        try:
            # Slot 8 is the per-frame ``time_dict`` that the inference consumer
            # fills with timings (see inference_pipeline.inference_consumer).
            # Must stay an empty *flat* dict — NOT ``metadata`` — or the perf
            # logger's ``value > 0`` check blows up on nested dicts like
            # metadata['intrinsics']. Slot 9 carries scalar per-frame controls
            # (currently just the session-level depth cap, ``Optional[float]``;
            # client stamps the same value every frame, server logs on change).
            inference_data = (
                image_pil,
                image_array,
                depth_array,
                pose,
                client_frame_number,
                client_timestamp,
                server_timestamp,
                {},
                max_depth_m,
            )

            self.inference_queue.put(inference_data)
            self.last_frame_index = client_frame_number
            self.tick_num += 1
            
            return True
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error submitting inference request: {e}")
            return False
    
    def get_processing_stats(self) -> dict:
        """Get current processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        return {
            'last_frame_index': self.last_frame_index,
            'tick_num': self.tick_num,
            'target_fps': self.target_fps,
            'queue_size': self.inference_queue.qsize() if hasattr(self.inference_queue, 'qsize') else -1
        }
    
    def wait_for_next_frame(self) -> None:
        """Wait appropriate time before processing next frame."""
        time.sleep(1.0 / self.target_fps)