# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Video Decoder Classes for H.264/H.265 Stream Processing
Hardware-accelerated decoding via NVIDIA NVDEC (PyNvVideoCodec).
No software codec is invoked; all H.264/H.265 decode runs on NVDEC.
"""

import numpy as np
import torch
from typing import List
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


class VideoDecoder:
    def __init__(self, codec: str, gpu_id: int = 0):
        validate_codec(codec)
        self._codec_name = codec
        self._decoder = nvc.CreateDecoder(
            gpuid=gpu_id,
            codec=_CODEC_MAP[codec],
            usedevicememory=True,
            outputColorType=nvc.OutputColorType.RGB,
        )

    def decode_frame(self, encoded_data: bytes) -> List[np.ndarray]:
        # The Quest client sends a single NAL unit per proto frame (mirrors
        # unity_grpc_server.H265Decoder). NVDEC expects Annex-B framing, so
        # prepend a start code when the payload lacks one. No-op for streams
        # that already carry start codes (iPad, dataset replay).
        data = encoded_data
        if self._codec_name == 'h265' and not (
            data.startswith(_NAL_START_CODE) or data.startswith(_SHORT_NAL_START_CODE)
        ):
            data = _NAL_START_CODE + data

        pkt = nvc.PacketData()
        pkt.bsl_data = data
        pkt.bsl = len(data)
        decoded = self._decoder.Decode(pkt)
        frames = []
        for f in decoded:
            tensor = torch.from_dlpack(f)
            frames.append(tensor.cpu().numpy())
        return frames
