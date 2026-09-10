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


def _fake_frame(frame_number=7, image=None, near_z=0.1, max_depth=0.0):
    """A semantic_data dict with the key set and dtypes the plugin produces."""
    if image is None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
    # ILLIXR encodes depth as u16_norm = 1 - near_z/depth_m, so
    # depth_m = near_z / (1 - u16/65535).  30000 -> 0.1/0.5422 = 0.184 m.
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
    # NB: depth_array is post-reprojection (build_depth_in_rgb_frame) with the
    # synthetic identity poses here, so it can legitimately be empty. The decode
    # contract itself is covered by convert() returning non-None -- it drops the
    # frame when no pixel yields positive depth -- and by
    # test_undecodable_depth_drops_the_frame below.
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


def test_undecodable_depth_drops_the_frame():
    """u16 == 65535 means 1 - u16_norm == 0: no finite depth, drop the frame."""
    f = _fake_frame()
    f["depth"] = np.frombuffer(
        np.full((DEPTH_H, DEPTH_W), 65535, dtype=np.uint16).tobytes(), dtype=np.uint8)
    assert IllixrFrameConverter().convert(f) is None


def test_dimensions_come_from_the_decoded_array(capsys):
    """If the plugin's reported resolution disagrees with the decoded array, the
    array wins (it is what gets resized) and the mismatch is reported once."""
    small = np.zeros((640, 640, 3), dtype=np.uint8)
    f = _fake_frame(image=small)          # still reports 1280x1280
    conv = IllixrFrameConverter()
    assert conv.convert(f) is not None
    assert "do not match the pixels" in capsys.readouterr().out
    assert conv.convert(f) is not None    # warn-once, not once per frame


class _RecordingDumper:
    """Stands in for ``SLAMGRPCServer``; captures the shim ``convert()`` builds."""

    def __init__(self):
        self.shim = None

    def _save_quest_replay_frame(self, frame_number, image_pil, depth_native, request, server_ts):
        self.shim = request


def test_rgb_intrinsics_are_rescaled_to_the_delivered_resolution():
    """android_sensors reports intrinsics at the RGB *sensor* resolution while the
    encoder delivers a downscaled image, so convert() has to rescale them (the Unity
    path does this before sending). Without the rescale fx/cx are off by the ratio
    and the principal point lands nowhere near the centre of the delivered image.

    Checked through the replay dumper because the 9-tuple carries no intrinsics.
    """
    from server.illixr_ingress import _rgb_intrinsics_resolution

    anchor_w, anchor_h = _rgb_intrinsics_resolution()
    delivered = 960
    assert delivered != anchor_w, "fixture must differ from the anchor or the branch never runs"

    rng = np.random.default_rng(1)
    f = _fake_frame(image=rng.integers(0, 256, size=(delivered, delivered, 3), dtype=np.uint8))
    wire_fx, wire_fy, wire_cx, wire_cy = (float(v) for v in f["intrinsics"])

    conv = IllixrFrameConverter()
    conv._dumper = _RecordingDumper()          # dataset.enabled path, without a real server
    assert conv.convert(f) is not None

    got = conv._dumper.shim.intrinsics
    sx, sy = delivered / anchor_w, delivered / anchor_h
    assert got.fx == pytest.approx(wire_fx * sx)
    assert got.fy == pytest.approx(wire_fy * sy)
    assert got.cx == pytest.approx(wire_cx * sx)
    assert got.cy == pytest.approx(wire_cy * sy)

    # The point of the rescale: the principal point sits at the centre of the image
    # that was actually delivered, not of the sensor it was measured on.
    assert got.cx / delivered == pytest.approx(0.5, abs=0.01)
    assert got.cy / delivered == pytest.approx(0.5, abs=0.01)

    # intrinsics.json has to describe the pixels in decoded_jpg/, so the shim
    # records the delivered size rather than the reported sensor size.
    assert (conv._dumper.shim.image_width, conv._dumper.shim.image_height) == (delivered, delivered)


def test_depth_intrinsics_are_not_rescaled():
    """Only RGB is downscaled by the encoder; the depth map arrives at native
    resolution, so its intrinsics must pass through untouched."""
    rng = np.random.default_rng(2)
    f = _fake_frame(image=rng.integers(0, 256, size=(960, 960, 3), dtype=np.uint8))
    wire = [float(v) for v in f["depth_intrinsics"]]

    conv = IllixrFrameConverter()
    conv._dumper = _RecordingDumper()
    assert conv.convert(f) is not None

    d = conv._dumper.shim.depth_intrinsics
    assert [d.fx, d.fy, d.cx, d.cy] == pytest.approx(wire)
    assert (conv._dumper.shim.depth_width, conv._dumper.shim.depth_height) == (DEPTH_W, DEPTH_H)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
