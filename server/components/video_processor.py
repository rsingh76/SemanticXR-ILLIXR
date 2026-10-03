# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video processing component for SLAM server."""

import cv2
import os
import tempfile
import traceback
from typing import Tuple, Optional
import numpy as np
from PIL import Image

from config.settings import Config, get_config
from ..video_decoders import VideoDecoder


class VideoProcessor:
    """Handles video processing operations with clean separation of concerns."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize video processor.
        
        Args:
            config: Configuration object, defaults to global config
        """
        self.config = config if config else get_config()
        
        # Video processing settings from config
        self.image_width = self.config.video.image_width
        self.image_height = self.config.video.image_height
        self.depth_width = self.config.video.depth_width
        self.depth_height = self.config.video.depth_height
        self.dataset_depth_height = self.config.video.dataset_depth_height
        self.dataset_depth_width = self.config.video.dataset_depth_width
        self.sharpness_threshold = self.config.video.sharpness_threshold
        
        # Initialize video decoders. iPad / dataset paths stream H.264;
        # Meta Quest streams H.265 (HEVC). Each decoder is stateful so we
        # can't share one instance across codecs.
        self.h264_decoder = VideoDecoder('h264')
        self.h265_decoder = VideoDecoder('h265')
        
        # Setup temp directory from config
        self.temp_output_dir = self.config.video.temp_output_dir
        if not os.path.exists(self.temp_output_dir):
            os.makedirs(self.temp_output_dir)
        self._cleanup_temp_directory()

    def reset_decoders(self) -> None:
        """Start both decoders from a clean state (call at session start).

        The decoders are shared by every gRPC stream, so two clients streaming
        at the same time would still interfere with each other.
        """
        self.h264_decoder.reset()
        self.h265_decoder.reset()

    def _cleanup_temp_directory(self) -> None:
        """Clean up temporary output directory."""
        if os.path.exists(self.temp_output_dir):
            for file in os.listdir(self.temp_output_dir):
                file_path = os.path.join(self.temp_output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Sharpness value (higher = sharper)
        """
        # Convert to BGR if needed for cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
            
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def is_frame_sharp_enough(self, image: np.ndarray) -> bool:
        """Check if frame meets sharpness threshold.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if frame is sharp enough for processing
        """
        return self.calculate_sharpness(image) >= self.sharpness_threshold

    def save_frame_data(self,
                       client_frame_number: int,
                       image_pil: Image.Image, 
                       depth_array: np.ndarray, 
                       pose_string: str) -> Tuple[str, str]:
        """Save frame data to temporary files.
        
        Args:
            client_frame_number: Frame number from client
            image_pil: PIL Image object
            depth_array: Depth data as numpy array
            pose_string: Pose information as string
            
        Returns:
            Tuple of (image_path, depth_path)
        """
        # Save image
        image_filename = f"frame-{client_frame_number}.jpg"
        image_path = os.path.join(self.temp_output_dir, image_filename)
        image_pil.save(image_path)
        
        # Save depth data
        depth_filename = f"depth-{client_frame_number}.npy"
        depth_path = os.path.join(self.temp_output_dir, depth_filename)
        np.save(depth_path, depth_array)
        
        return image_path, depth_path
    
    def process_video_frame(self, frame_data: bytes, frame_number: int, codec: str = 'h264',
                            keep: bool = True) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Decode one message's video packet and check the image belongs to it.

        Args:
            frame_data: Encoded bytes of exactly one picture.
            frame_number: The message's client frame number. It rides through
                the decoder as the pts; the returned image must carry it back.
            codec: 'h264' (iPad / dataset streams) or 'h265' (Meta Quest).
            keep: False for frames that will be skipped. They are still decoded
                (later pictures reference earlier ones) but not copied off the GPU.

        Returns:
            Tuple of (image, reject_reason). ``reject_reason`` is None when the
            image is usable, otherwise "<category>: <detail>". The sharpness gate
            rejects hand-shake blur on iPad streams only: Quest is head-mounted
            with VIO, and its H.265 compression produces much lower Laplacian
            variance even on crisp content (~18 vs ~>>100 for iPad), so the gate
            is bypassed when codec=='h265'.
        """
        decoder = self.h265_decoder if codec == 'h265' else self.h264_decoder
        try:
            decoded = decoder.decode(frame_data, pts=frame_number, to_host=keep)
        except Exception as e:
            traceback.print_exc()
            return None, f"decoder exception: {type(e).__name__}: {e}"

        if not decoded:
            return None, "no image from decoder: the packet produced no picture"

        matching = [d for d in decoded if d.pts == frame_number]
        if not matching:
            return None, (f"timestamp mismatch: decoder returned frame(s) "
                          f"{[d.pts for d in decoded]} for message {frame_number}")
        if len(decoded) > 1:
            print(f"⚠️  [VIDEO] decoder returned {len(decoded)} images for message {frame_number}; "
                  f"dropped frame(s) {[d.pts for d in decoded if d.pts != frame_number]}")

        image = matching[0].image
        if not keep:
            return None, None

        # Sharpness gate: skip for Quest (see docstring).
        if codec != 'h265':
            sharpness = self.calculate_sharpness(image)
            if sharpness < self.sharpness_threshold:
                return image, f"blurry: sharpness {sharpness:.1f} < {self.sharpness_threshold}"

        return image, None