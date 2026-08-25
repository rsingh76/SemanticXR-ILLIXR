# SPDX-FileCopyrightText: Copyright (c) 2026 Rahul Singh, University of Illinois Urbana-Champaign <rahuls10@illinois.edu>
# SPDX-License-Identifier: Apache-2.0

"""Contract test for the ILLIXR ingress conversion.

Feeds ``IllixrFrameConverter.convert()`` a synthetic ``semantic_data`` dict
shaped exactly like the one the ``semantic_python`` plugin injects (ILLIXR
commit 12a3f448 onward: ``image`` is a DECODED (H, W, 3) uint8 array) and checks
the 9-tuple the inference pipeline consumes.

No ILLIXR, no GPU, no headset required.

Run directly:
    python tests/test_illixr_ingress.py
Or via pytest:
    python -m pytest tests/test_illixr_ingress.py -q
"""

import sys
import types

import numpy as np
import pytest

# The ILLIXR path no longer decodes video, but importing anything under
# ``server.components`` still pulls in ``VideoProcessor`` -> ``PyNvVideoCodec``
# via that package's __init__. That binding is only needed by the gRPC path, so
# stub it out to keep this test runnable without the NVIDIA codec bindings.
# (A real ILLIXR run still needs the dependency installed until the import is
# decoupled -- see SEMANTICXR_ILLIXR_INTEGRATION.md.)
if "PyNvVideoCodec" not in sys.modules:
    try:
        import PyNvVideoCodec  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("PyNvVideoCodec")
        # video_decoders.py builds a codec map at import time.
        _stub.cudaVideoCodec = types.SimpleNamespace(H264=0, HEVC=1)
        sys.modules["PyNvVideoCodec"] = _stub

from server.illixr_ingress import IllixrFrameConverter, _quest_target_resolution

NATIVE_W = NATIVE_H = 1280   # Quest RGB sensor
DEPTH_W = DEPTH_H = 320      # Quest environment-depth sensor


def _fake_frame(frame_number=7, image=None, near_z=-0.2, max_depth=0.0):
    """A semantic_data dict with the key set and dtypes the plugin produces."""
    if image is None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
    # R16_UNORM depth: mid-range values so depth_m stays finite and positive.
    depth_u16 = np.full((DEPTH_H, DEPTH_W), 30000, dtype=np.uint16)
    eye = np.eye(4, dtype=np.float32)
    return {
        "image": image,
        "frame_number": frame_number,
        "image_width": NATIVE_W,
        "image_height": NATIVE_H,
        "depth": np.frombuffer(depth_u16.tobytes(), dtype=np.uint8),
        "depth_width": DEPTH_W,
        "depth_height": DEPTH_H,
        "depth_near_z": near_z,
        "intrinsics": np.array([600.0, 600.0, 640.0, 640.0], dtype=np.float32),
        "depth_intrinsics": np.array([150.0, 150.0, 160.0, 160.0], dtype=np.float32),
        "rgb_camera_pose": eye.copy(),
        "depth_pose": eye.copy(),
        "max_depth_m": max_depth,
    }


def test_convert_returns_well_formed_9_tuple():
    tw, th = _quest_target_resolution()
    out = IllixrFrameConverter().convert(_fake_frame(frame_number=7))

    assert out is not None, "converter dropped a well-formed frame"
    assert len(out) == 9, f"inference pipeline expects a 9-tuple, got {len(out)}"

    image_pil, image_array, depth_array, pose_data, frame_number, \
        client_ts, server_ts, time_dict, max_depth_m = out

    assert image_pil.size == (tw, th)
    assert image_pil.mode == "RGB"
    assert image_array.shape == (th, tw, 3) and image_array.dtype == np.uint8
    assert depth_array.shape == (th, tw) and depth_array.dtype == np.float32
    assert len(pose_data) == 17 and pose_data[0] == 7   # [frame] + 16 row-major
    assert frame_number == 7
    assert isinstance(client_ts, int) and isinstance(server_ts, int)
    assert time_dict == {}
    assert max_depth_m is None, "0.0 means unset -> None so the YAML default wins"


def test_per_frame_max_depth_is_passed_through():
    out = IllixrFrameConverter().convert(_fake_frame(max_depth=4.5))
    assert out is not None
    assert out[8] == pytest.approx(4.5)


@pytest.mark.parametrize("bad_image", [
    np.empty((0,), dtype=np.uint8),                 # cache miss / not yet decoded
    np.zeros((NATIVE_H, NATIVE_W), dtype=np.uint8),  # 2-D, not (H, W, 3)
    np.zeros((NATIVE_H, NATIVE_W, 4), dtype=np.uint8),  # RGBA
])
def test_unusable_image_drops_the_frame(bad_image):
    """The plugin hands back an empty array on a decoded-frame-cache miss; the
    converter must drop the frame rather than raise inside the worker."""
    assert IllixrFrameConverter().convert(_fake_frame(image=bad_image)) is None


def test_dimensions_come_from_the_decoded_array(capsys):
    """If the plugin's reported resolution disagrees with the decoded array, the
    array wins (it is what gets resized) and the mismatch is reported once."""
    small = np.zeros((640, 640, 3), dtype=np.uint8)
    f = _fake_frame(image=small)          # still reports 1280x1280
    conv = IllixrFrameConverter()
    assert conv.convert(f) is not None
    assert "do not match the pixels" in capsys.readouterr().out
    assert conv.convert(f) is not None    # warn-once, not once per frame


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
