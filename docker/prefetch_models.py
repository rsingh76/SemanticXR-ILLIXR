# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-download the Hugging Face / whisper models a config will load at runtime.

Without this they are fetched lazily on first use (same cache locations), which
on a new site means a slow or failing first session instead of a failing
container start. Downloads only -- nothing is instantiated on the GPU.

    python docker/prefetch_models.py --config config/debug/quest_debug.yaml
"""

import argparse
import os
import sys

import yaml


def log(msg):
    print(f"\033[1;32m[prefetch]\033[0m {msg}", flush=True)


def clip(model_name, pretrained):
    import open_clip
    cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
    if not cfg:
        raise SystemExit(f"open_clip has no pretrained tag {pretrained!r} for {model_name!r}")
    log(f"CLIP {model_name} / {pretrained}")
    open_clip.download_pretrained(cfg)


def bert():
    # GroundingDINO's text encoder and RAM's tokenizer both use bert-base-uncased.
    from transformers import AutoTokenizer, BertModel
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()  # "weights not used" warning is expected for BertModel
    log("bert-base-uncased (GroundingDINO / RAM text encoder)")
    AutoTokenizer.from_pretrained("bert-base-uncased")
    BertModel.from_pretrained("bert-base-uncased")


def asr(cfg):
    if not cfg.get("enabled", True):
        log("ASR disabled in config; skipping")
        return
    backend = cfg.get("backend", "faster-whisper")
    model = cfg.get("model", "small.en")
    if backend == "openai-whisper":
        import whisper
        root = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "whisper")
        log(f"openai-whisper {model} -> {root}")
        whisper._download(whisper._MODELS[model], root, in_memory=False)
    elif backend == "faster-whisper":
        from faster_whisper import download_model
        log(f"faster-whisper {model}")
        download_model(model)
    else:
        log(f"ASR backend {backend!r} needs no local model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    model = cfg.get("model", {})
    clip_cfg = model.get("clip", {})

    # Defaults mirror config/settings.py (CLIPConfig / VisualizationConfig).
    wanted = {(clip_cfg.get("model_name", "ViT-H-14"), clip_cfg.get("pretrained", "laion2b_s32b_b79k"))}
    # The visualization service builds its own CLIP for query matching from
    # separate keys (visualization.clip_model / visualization.pretrained).
    vis = model.get("visualization", {})
    wanted.add((vis.get("clip_model", "ViT-H-14"), vis.get("pretrained", "laion2b_s32b_b79k")))
    for name, tag in sorted(wanted):
        clip(name, tag)
    if model.get("detection", {}).get("enabled", True):
        bert()
    asr(model.get("visualization", {}).get("asr", {}))


if __name__ == "__main__":
    sys.exit(main())
