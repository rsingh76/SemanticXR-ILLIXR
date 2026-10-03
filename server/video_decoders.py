# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Video Decoder Classes for H.264/H.265 Stream Processing
Hardware-accelerated decoding via NVIDIA NVDEC (PyNvVideoCodec).
No software codec is invoked; all H.264/H.265 decode runs on NVDEC.
"""

import numpy as np
import torch
from typing import List, NamedTuple, Optional
import PyNvVideoCodec as nvc


def validate_codec(codec: str) -> None:
    if codec not in ['h264', 'h265']:
        raise ValueError(f"Invalid codec: {codec}")


_CODEC_MAP = {
    'h264': nvc.cudaVideoCodec.H264,
    'h265': nvc.cudaVideoCodec.HEVC,
}

_NAL_START_CODE = b'\x00\x00\x00\x01'
_SHORT_NAL_START_CODE = b'\x00\x00\x01'

# LOW: a packet's picture comes out of the same Decode() call that consumed it.
# The default (NATIVE) holds pictures back -- 2 per stream in practice -- so each
# image was paired with the depth/pose of a later message.
DECODE_LATENCY = nvc.DisplayDecodeLatencyType.LOW


class DecodedImage(NamedTuple):
    pts: int                      # the pts passed to decode() with this picture's packet
    image: Optional[np.ndarray]   # RGB HWC uint8; None when decoded with to_host=False


class VideoDecoder:
    def __init__(self, codec: str, gpu_id: int = 0):
        validate_codec(codec)
        self._codec_name = codec
        self._gpu_id = gpu_id
        self._decoder = self._create_decoder()

    def _create_decoder(self):
        return nvc.CreateDecoder(
            gpuid=self._gpu_id,
            codec=_CODEC_MAP[self._codec_name],
            usedevicememory=True,
            outputColorType=nvc.OutputColorType.RGB,
            latency=DECODE_LATENCY,
        )

    def reset(self) -> None:
        """Drop all decoder state (reference pictures, anything buffered).

        Call at the start of every client session so nothing from the previous
        stream can come out attached to the new session's frames.
        """
        self._decoder = self._create_decoder()

    def decode(self, encoded_data: bytes, pts: int = 0, to_host: bool = True) -> List[DecodedImage]:
        """Decode one packet holding exactly one complete picture.

        ``pts`` is carried through NVDEC and returned on the picture it belongs
        to, so callers can verify that an image matches the message it arrived
        with. ``to_host=False`` decodes (keeping the reference chain intact)
        without copying the image off the GPU.
        """
        # Each message is one complete access unit: Quest sends one MediaCodec
        # output buffer (VPS/SPS/PPS prepended on IDR frames), iPad one
        # VideoToolbox sample buffer. NVDEC expects Annex-B framing, so prepend
        # a start code when the payload lacks one.
        data = encoded_data
        if self._codec_name == 'h265' and not (
            data.startswith(_NAL_START_CODE) or data.startswith(_SHORT_NAL_START_CODE)
        ):
            data = _NAL_START_CODE + data

        # PyNvVideoCodec >= 2.0 takes the bitstream as a raw pointer (int), not
        # bytes. ``buf`` must stay alive until Decode() returns.
        buf = np.frombuffer(data, dtype=np.uint8)
        pkt = nvc.PacketData()
        pkt.bsl_data = buf.ctypes.data
        pkt.bsl = buf.size
        pkt.pts = pts
        # The packet ends on a picture boundary; without this the parser waits
        # for the next picture's start code before releasing this one.
        pkt.decode_flag = nvc.VideoPacketFlag.ENDOFPICTURE
        decoded = self._decoder.Decode(pkt)
        return [
            DecodedImage(f.timestamp, torch.from_dlpack(f).cpu().numpy() if to_host else None)
            for f in decoded
        ]

    def decode_frame(self, encoded_data: bytes) -> List[np.ndarray]:
        """Decode a packet and return only the images (no pts bookkeeping)."""
        return [d.image for d in self.decode(encoded_data)]
