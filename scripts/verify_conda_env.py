# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the conda environment end to end (see docs/CONDA_SETUP.md).

Stages (each is reported PASS/FAIL, the script keeps going after a failure):
  1. torch sees the GPU and can run a kernel on it
  2. CUDA extensions (pytorch3d, chamferdist, groundingdino._C) execute on the GPU
  3. every model in the default pipeline loads its checkpoint
  4. a real forward pass through detection -> segmentation -> captioning -> CLIP

Usage (inside `conda activate semanticxr`):
    python scripts/verify_conda_env.py [--skip-models] [--image PATH] [--config YAML]
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

results = []


def stage(name):
    def deco(fn):
        def run(*a, **kw):
            t0 = time.time()
            try:
                out = fn(*a, **kw)
                results.append((name, True, f"{time.time() - t0:.1f}s"))
                print(f"[PASS] {name} ({time.time() - t0:.1f}s)")
                return out
            except Exception:
                results.append((name, False, ""))
                print(f"[FAIL] {name}")
                traceback.print_exc()
                return None
        return run
    return deco


@stage("torch + GPU kernel")
def check_torch():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    assert arch in torch.cuda.get_arch_list(), (
        f"GPU is {arch} but this torch build only has {torch.cuda.get_arch_list()}")
    x = torch.randn(256, 256, device="cuda")
    assert torch.isfinite((x @ x).sum()).item()
    print(f"       torch {torch.__version__}, CUDA {torch.version.cuda}, "
          f"{torch.cuda.get_device_name(0)} ({arch})")


@stage("pytorch3d CUDA ops")
def check_pytorch3d():
    import torch
    from pytorch3d.ops import knn_points
    a = torch.rand(1, 300, 3, device="cuda")
    b = torch.rand(1, 400, 3, device="cuda")
    assert knn_points(a, b, K=2).dists.is_cuda


@stage("chamferdist CUDA op")
def check_chamferdist():
    import torch
    from chamferdist import ChamferDistance
    a = torch.rand(1, 300, 3, device="cuda")
    b = torch.rand(1, 400, 3, device="cuda")
    assert torch.isfinite(ChamferDistance()(a, b)).item()


@stage("groundingdino MultiScaleDeformableAttention CUDA op")
def check_msdeform():
    import torch
    from groundingdino.models.GroundingDINO.ms_deform_attn import (
        MultiScaleDeformableAttnFunction,
    )
    N, M, D, L, P = 1, 8, 32, 2, 4
    shapes = torch.as_tensor([[8, 8], [4, 4]], dtype=torch.long, device="cuda")
    start = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
    S = int(shapes.prod(1).sum())
    value = torch.rand(N, S, M, D, device="cuda")
    loc = torch.rand(N, 5, M, L, P, 2, device="cuda")
    attn = torch.softmax(torch.rand(N, 5, M, L * P, device="cuda"), -1).view(N, 5, M, L, P)
    out = MultiScaleDeformableAttnFunction.apply(value, shapes, start, loc, attn, 64)
    assert out.is_cuda and torch.isfinite(out).all()


@stage("gRPC protobuf stubs present")
def check_stubs():
    import importlib
    importlib.import_module("slam.protocols.vis_proto.vis_pb2")
    sys.path.insert(0, str(REPO_ROOT / "server"))
    importlib.import_module("xr_service_pb2")


@stage("NVDEC H.264 decode (server/video_decoders.py)")
def check_nvdec():
    import shutil
    import subprocess
    import tempfile
    assert shutil.which("ffmpeg"), "ffmpeg not on PATH (conda install -c conda-forge ffmpeg)"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.h264"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "testsrc=size=640x480:rate=10", "-t", "1", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-f", "h264", str(path)], check=True)
        sys.path.insert(0, str(REPO_ROOT / "server"))
        from video_decoders import VideoDecoder
        frames = VideoDecoder("h264").decode_frame(path.read_bytes())
    assert len(frames) > 0 and frames[0].shape == (480, 640, 3), "NVDEC returned no/odd frames"


@stage("segment_anything import survives $GSA_PATH on sys.path")
def check_sam_import_order():
    # slam/models/*.py append $GSA_PATH to sys.path *before* importing
    # segment_anything; a PEP 660 editable install gets shadowed by GSA's
    # init-less outer segment_anything/ dir. Run in a fresh interpreter so
    # nothing is imported already.
    import subprocess
    gsa = os.environ.get("GSA_PATH", str(REPO_ROOT / "external/Grounded-Segment-Anything"))
    code = (f"import sys; sys.path.append({gsa!r}); "
            "from segment_anything import sam_model_registry")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr.strip().splitlines()[-1]


@stage("model checkpoints on disk")
def check_weights():
    from config.settings import get_config
    c = get_config()
    needed = [c.grounding_dino_checkpoint_path, c.sam_checkpoint_path,
              c.mobile_sam_checkpoint_path, c.gsa_path / "ram_swin_large_14m.pth"]
    missing = [str(p) for p in needed if not p.exists() or p.stat().st_size < 1_000_000]
    assert not missing, f"missing checkpoints: {missing}"


@stage("load all four models (default config)")
def load_models():
    from config.settings import get_config
    from slam.models.captioning import captioning
    from slam.models.clip import clipModel
    from slam.models.detection import detector
    from slam.models.segmentation import SegmentationModel
    c = get_config()
    dev = "cuda:0"
    m = {}
    m["caption"] = captioning(class_set=c.model.captioning.class_set, device=dev,
                              add_bg_classes=c.model.captioning.add_bg_classes,
                              accumu_classes=c.model.captioning.accumu_classes)
    m["detect"] = detector(detector="dino", device=dev, box_threshold=0.2,
                           text_threshold=0.2, nms_threshold=0.5)
    m["seg"] = SegmentationModel(device=dev, sam_variant=c.model.segmentation.sam_variant,
                                 batched_sam=c.model.segmentation.batched_sam,
                                 trt_sam=c.model.segmentation.trt_sam, useDetector=True)
    m["clip"] = clipModel(device=dev, batched_clip=c.model.clip.batched_clip,
                          trt_clip=c.model.clip.trt_clip, precision=c.model.clip.precision,
                          batch_size=c.model.clip.batch_size,
                          clip_model_name=c.model.clip.model_name,
                          pretrained=c.model.clip.pretrained)
    return m


@stage("forward pass: RAM tags -> GroundingDINO -> SAM -> CLIP")
def forward(models, image_path):
    import cv2
    import numpy as np
    from PIL import Image
    bgr = cv2.imread(str(image_path))
    assert bgr is not None, f"cannot read {image_path}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # Same call sequence as semantic_slam.engine.SemanticSLAM.push
    _caption, _text_prompt = models["caption"].gen_caption(pil)
    classes = models["caption"].classes
    print(f"       RAM classes: {classes[:8]}{' ...' if len(classes) > 8 else ''}")
    assert len(classes) > 0

    det = models["detect"].get_detections(bgr, classes)
    print(f"       GroundingDINO detections: {len(det.class_id)}")
    assert len(det.class_id) > 0

    masks, _, _ = models["seg"].run_segmentation(rgb, det)
    det.mask = masks
    assert len(masks) == len(det.class_id)
    print(f"       SAM masks: {masks.shape}, mean coverage {masks.mean():.3f}")

    crops, img_feats, txt_feats = models["clip"].get_clip_features(pil, rgb, det, classes)
    assert len(img_feats) == len(det.class_id)
    assert np.isfinite(np.asarray(img_feats)).all()
    print(f"       CLIP image features: {np.asarray(img_feats).shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true", help="only run the CUDA/extension checks")
    ap.add_argument("--image", default=None, help="test image (default: a GSA demo image)")
    ap.add_argument("--config", default=None,
                    help="YAML profile to load models from (default: config/defaults.yaml), "
                         "e.g. config/parallelization_mobileclip.yaml")
    args = ap.parse_args()

    if args.config:
        from config.settings import Config, set_config
        set_config(Config.from_yaml(args.config))
        print(f"       using config {args.config}")

    check_torch()
    check_pytorch3d()
    check_chamferdist()
    check_msdeform()
    check_stubs()
    check_nvdec()
    check_sam_import_order()
    if not args.skip_models:
        check_weights()
        models = load_models()
        if models:
            image = args.image or (Path(os.environ.get(
                "GSA_PATH", REPO_ROOT / "external/Grounded-Segment-Anything")) / "assets/demo1.jpg")
            forward(models, image)

    print("\n==== summary ====")
    for name, ok, t in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
