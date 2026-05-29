# -*- coding: utf-8 -*-
# @Function: LHM++ model registry for HuggingFace and ModelScope

from __future__ import annotations

import os
from typing import Dict, FrozenSet, Tuple

ModelScope_Prior_MODEL_CARD = {
    "LHMPP": "Damo_XR_Lab/LHMPP-Prior",
}
HuggingFace_Prior_MODEL_CARD = {
    "LHMPP": "3DAIGC/LHMPP-Prior",
}
ModelScope_MODEL_CARD = {
    # SMPLX-FREE / ShapeHead + GS export: set repo id when published; until then use local checkpoint + --model_path
    "LHMPP-700M-PixelShuffle": "Damo_XR_Lab/LHMPP-700M-PixelShuffle",  # default; SMPLX-FREE + MLPPixelShuffle head + GS
    "LHMPP-700M-SMPLX-FREE": "Damo_XR_Lab/LHMPP-700M-SMPLX-FREE",
    "LHMPP-700M": "Damo_XR_Lab/LHMPP-700M",
    # "LHMPP-700MC": "Damo_XR_Lab/LHMPP-700MC",  # coming soon
    "LHMPPS-700M": "Damo_XR_Lab/LHMPPS-700M",
    # DNA-Rendering reconstruction (novel-view / novel-pose); not for dynamic animation metrics
    "LHMPP-700M-PS-DNA": "Damo_XR_Lab/LHMPP-700M-PixelShuffle-DNA",
    "LHMPP-700M-LVSM-DNA": "Damo_XR_Lab/LHMPP-700M-LVSM-DNA",
}

HuggingFace_MODEL_CARD = {
    # Publish repo id when releasing; otherwise use local weights (e.g. test_app_case.py --model_path).
    "LHMPP-700M-PixelShuffle": "3DAIGC/LHMPP-700M-PixelShuffle",  # default; SMPLX-FREE + MLPPixelShuffle head + GS
    "LHMPP-700M-SMPLX-FREE": "3DAIGC/LHMPP-700M-SMPLX-FREE",
    "LHMPP-700M": "3DAIGC/LHMPP-700M",
    # "LHMPP-700MC": "3DAIGC/LHMPP-700MC",  # coming soon
    "LHMPPS-700M": "3DAIGC/LHMPPS-700M",
    "LHMPP-700M-PS-DNA": "3DAIGC/LHMPP-700M-PixelShuffle-DNA",
    "LHMPP-700M-LVSM-DNA": "3DAIGC/LHMPP-700M-LVSM-DNA",
}

MODEL_CONFIG = {
    "LHMPP-700M-PixelShuffle": "./configs/train/LHMPP-pixelshuffle.yaml",
    "LHMPP-700M-SMPLX-FREE": "./configs/LHMPP-anyview-SMPLX-FREE.yaml",
    "LHMPP-700M": "./configs/train/LHMPP-any-view.yaml",
    # "LHMPP-700MC": "./configs/train/LHMPP-any-view-convhead.yaml",  # coming soon
    "LHMPPS-700M": "./configs/train/LHMPP-any-view-DPTS.yaml",
}

# DNA-Rendering reconstruction benchmark (``dna-rendering-novelview`` / ``novelpose``).
# Short CLI tags map to Hub repos above; weights architecture comes from checkpoint ``config.json``.
RECONSTRUCTION_MODEL_NAMES: Tuple[str, ...] = (
    "LHMPP-700M-PS-DNA",
    "LHMPP-700M-LVSM-DNA",
)
DEFAULT_RECONSTRUCTION_MODEL_NAME = "LHMPP-700M-PS-DNA"

_RECONSTRUCTION_MODEL_ALIASES: Dict[str, str] = {
    "LHMPP-700M-PixelShuffle-DNA": "LHMPP-700M-PS-DNA",
}

RECONSTRUCTION_MODEL_CONFIG = {
    "LHMPP-700M-PS-DNA": "./configs/reconstruction/LHMPP-700M-PixelShuffle-DNA.yaml",
    "LHMPP-700M-LVSM-DNA": "./configs/reconstruction/LHMPP-700M-LVSM-DNA.yaml",
}

RECONSTRUCTION_MODEL_NAMES_FROZEN: FrozenSet[str] = frozenset(RECONSTRUCTION_MODEL_NAMES)

# Flip to True when DNA novel-view / novel-pose inference and recon metrics are release-ready.
RECONSTRUCTION_BENCHMARK_ENABLED = False

RECONSTRUCTION_BENCHMARK_STAY_TUNED = (
    "DNA-Rendering reconstruction benchmark (novel-view / novel-pose inference "
    "and recon metrics) is not ready yet — stay tuned.\n"
    "Data download is available:\n"
    "  python scripts/download_evaluation/download_reconstruction_benchmark.py --download\n"
    "  python scripts/download_evaluation/download_reconstruction_benchmark.py --untar"
)


def ensure_reconstruction_benchmark_enabled() -> None:
    """Exit early until reconstruction inference / metrics are ready for users."""
    if RECONSTRUCTION_BENCHMARK_ENABLED:
        return
    import sys

    print(RECONSTRUCTION_BENCHMARK_STAY_TUNED, file=sys.stderr)
    raise SystemExit(2)

MEMORY_MODEL_CARD = {
    "LHMPP-700M-PixelShuffle": 8000,  # 8G, default
    "LHMPP-700M-SMPLX-FREE": 8000,  # 8G
    "LHMPP-700M": 8000,  # 8G
    # "LHMPP-700MC": 8000,  # 8G, coming soon
    "LHMPPS-700M": 8000,  # 8G
    "LHMPP-700M-PS-DNA": 8000,
    "LHMPP-700M-LVSM-DNA": 8000,
}

# App / inference: gs_render (Gaussian raster only, no neural refiner) and
# scripts/inference/to_gs_ply.py (GS PLY export, T-pose or SMPL-X frame).
# SMPLX-FREE variants (ShapeHead + featbacksplat GS); neural_renderer type may differ.
# First entry is the CLI default for to_gs_ply.py (--model_name).
GS_RENDER_SUPPORTED_MODEL_NAMES = [
    "LHMPP-700M-PixelShuffle",
    "LHMPP-700M-SMPLX-FREE",
]


def model_supports_gs_render(model_name: str) -> bool:
    return model_name in GS_RENDER_SUPPORTED_MODEL_NAMES


def normalize_model_name(model_name: str) -> str:
    """Map legacy / Hub-style DNA names to canonical CLI tags."""
    name = str(model_name).strip()
    return _RECONSTRUCTION_MODEL_ALIASES.get(name, name)


def is_reconstruction_model(model_name: str) -> bool:
    return normalize_model_name(model_name) in RECONSTRUCTION_MODEL_NAMES_FROZEN


def is_animation_model(model_name: str) -> bool:
    name = normalize_model_name(model_name)
    if name in RECONSTRUCTION_MODEL_NAMES_FROZEN:
        return False
    return name in MODEL_CONFIG


def require_reconstruction_model(model_name: str) -> str:
    """Return canonical reconstruction tag or raise ``ValueError``."""
    name = normalize_model_name(model_name)
    if name not in RECONSTRUCTION_MODEL_NAMES_FROZEN:
        allowed = ", ".join(RECONSTRUCTION_MODEL_NAMES)
        raise ValueError(
            f"Reconstruction benchmark requires one of: {allowed}. Got {model_name!r}."
        )
    return name


def require_animation_model(model_name: str) -> str:
    """Return canonical animation tag or raise ``ValueError``."""
    name = normalize_model_name(model_name)
    if name in RECONSTRUCTION_MODEL_NAMES_FROZEN:
        allowed_anim = ", ".join(sorted(MODEL_CONFIG.keys()))
        raise ValueError(
            f"Dynamic animation benchmark does not support reconstruction models "
            f"({', '.join(RECONSTRUCTION_MODEL_NAMES)}). "
            f"Use a non-reconstruction model card, e.g. {allowed_anim}. Got {model_name!r}."
        )
    if name not in MODEL_CONFIG:
        allowed_anim = ", ".join(sorted(MODEL_CONFIG.keys()))
        raise ValueError(
            f"Unknown animation model {model_name!r}. Supported: {allowed_anim}."
        )
    return name


def infer_reconstruction_model_from_root(root: str) -> str:
    """Read model tag from ``.../dna-rendering/<task>/<model_name>/`` (basename of ``root``)."""
    tag = os.path.basename(os.path.normpath(os.path.expanduser(root)))
    if not tag or tag in (".", ".."):
        raise ValueError(
            f"Cannot infer reconstruction model from --root {root!r}; "
            f"pass --model-name (one of: {', '.join(RECONSTRUCTION_MODEL_NAMES)})."
        )
    return require_reconstruction_model(tag)