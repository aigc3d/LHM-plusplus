#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DNA-Rendering reconstruction benchmark metrics: PSNR / SSIM / LPIPS.

**Not runnable yet** — exits with a stay-tuned message until release (see
``ensure_reconstruction_benchmark_enabled`` in ``core/utils/model_card.py``).

Pred/GT are composited onto a **white background** with GT mask, then metrics are
computed on the full frame (no bbox crop)::

    comp = rgb * mask + (1 - mask)

Expected tree (from ``dna-rendering-novelview`` / ``dna-rendering-novelpose``)::

    <root>/{scene_id}/{frame:05d}/
        gt/rgb/{cam:05d}.png
        gt/mask/{cam:05d}.png
        neural_render/rgb/{cam:05d}.png
        gs_render/rgb/{cam:05d}.png          # optional

Novel-view and novel-pose share this layout; only ``--root`` differs, e.g.::

    ./exps/benchmarks/dna-rendering/novel-view/{model_name}/
    ./exps/benchmarks/dna-rendering/novel-pose/{model_name}/

Aggregation: per camera stem -> mean per frame folder -> mean per scene -> global.

Depends on: ``torch``, ``lpips``, ``numpy``, ``Pillow``, ``scikit-image`` (see ``metric_utils.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from core.utils.model_card import (  # noqa: E402
    DEFAULT_RECONSTRUCTION_MODEL_NAME,
    RECONSTRUCTION_MODEL_NAMES,
    infer_reconstruction_model_from_root,
    require_reconstruction_model,
)
from metric_utils import (
    configure_logging,
    list_frame_dirs,
    list_scene_dirs,
    load_mask_float,
    load_rgb,
    paired_stems,
)

PSNR_DEFINITION = (
    "MSE(mean) over full frame of (rgb*mask+(1-mask)) pred vs gt, PSNR=10 log10(1/MSE)"
)

_LOG = logging.getLogger(__name__)

BRANCHES = ("neural", "gs")
_BRANCH_PRED_SUBDIR = {
    "neural": ("neural_render", "rgb"),
    "gs": ("gs_render", "rgb"),
}
_GT_RGB_PARTS = ("gt", "rgb")
_GT_MASK_PARTS = ("gt", "mask")


def _join_parts(base: str, parts: Tuple[str, ...]) -> str:
    return os.path.join(base, *parts)


def infer_task_from_root(root: str) -> str:
    norm = os.path.normpath(root).replace("\\", "/")
    if "novel-pose" in norm:
        return "novel-pose"
    if "novel-view" in norm:
        return "novel-view"
    return "unknown"


def resolve_task(task_arg: str, root: str) -> str:
    if task_arg != "auto":
        return task_arg
    return infer_task_from_root(root)


def composite_white_background(rgb: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """``rgb * mask + (1 - mask)`` with white (1.0) background."""
    m = mask_hw[..., None]
    return np.clip(rgb * m + (1.0 - m), 0.0, 1.0)


def fullimage_psnr_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    err = np.mean((pred - gt) ** 2)
    if err <= 1e-20:
        return 99.0
    return float(10.0 * np.log10(1.0 / err))


def fullimage_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)
    try:
        return float(structural_similarity(gt, pred, data_range=1.0, channel_axis=-1))
    except TypeError:
        return float(structural_similarity(gt, pred, data_range=1.0, multichannel=True))


def batched_lpips_fullimage(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    lpips_fn: torch.nn.Module,
    device: torch.device,
) -> List[float]:
    if not preds:
        return []
    xs = torch.stack(
        [torch.from_numpy(np.clip(p, 0.0, 1.0)).float().permute(2, 0, 1) for p in preds],
        dim=0,
    ).to(device) * 2.0 - 1.0
    ys = torch.stack(
        [torch.from_numpy(np.clip(g, 0.0, 1.0)).float().permute(2, 0, 1) for g in gts],
        dim=0,
    ).to(device) * 2.0 - 1.0
    with torch.no_grad():
        d = lpips_fn(xs, ys)
    return [float(d[i].item()) for i in range(d.shape[0])]


def default_out_meta(root: str, task: str) -> str:
    if task in ("novel-view", "novel-pose"):
        name = f"recon_metrics_{task}.meta.json"
    else:
        name = "recon_metrics.meta.json"
    return os.path.join(root, name)


def eval_frame_folder_branch(
    frame_path: str,
    branch: str,
    lpips_fn: torch.nn.Module,
    device: torch.device,
    *,
    with_per_view: bool,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Evaluate one time-step folder for a single render branch."""
    pred_parts = _BRANCH_PRED_SUBDIR[branch]
    pred_dir = _join_parts(frame_path, pred_parts)
    gt_rgb_dir = _join_parts(frame_path, _GT_RGB_PARTS)
    gt_mask_dir = _join_parts(frame_path, _GT_MASK_PARTS)

    stems = paired_stems(pred_dir, gt_rgb_dir, gt_mask_dir)
    errs: List[str] = []
    if not stems:
        return None, [f"no paired png under {pred_parts[0]}/rgb vs gt/rgb+mask"]

    psnrs: List[float] = []
    ssims: List[float] = []
    lpips_vals: List[float] = []
    per_view: Dict[str, Any] = {}

    preds_batch: List[np.ndarray] = []
    gts_batch: List[np.ndarray] = []
    ok_stems: List[str] = []

    for stem in stems:
        pp = os.path.join(pred_dir, stem)
        gp = os.path.join(gt_rgb_dir, stem)
        mp = os.path.join(gt_mask_dir, stem)
        try:
            pred = load_rgb(pp)
            gt = load_rgb(gp)
            mk = load_mask_float(mp)
        except Exception as ex:  # noqa: BLE001
            errs.append(f"{stem}: load failed {ex}")
            continue

        if pred.shape[:2] != gt.shape[:2] or mk.shape[:2] != pred.shape[:2]:
            errs.append(
                f"{stem}: shape mismatch pred{pred.shape[:2]} gt{gt.shape[:2]} "
                f"mask{mk.shape[:2]}"
            )
            continue

        pred_c = composite_white_background(pred, mk)
        gt_c = composite_white_background(gt, mk)
        psnr = fullimage_psnr_mse(pred_c, gt_c)
        ssim = fullimage_ssim(pred_c, gt_c)
        psnrs.append(psnr)
        ssims.append(ssim)
        preds_batch.append(pred_c)
        gts_batch.append(gt_c)
        ok_stems.append(stem)
        if with_per_view:
            per_view[stem] = {"psnr": psnr, "ssim": ssim}

    if not preds_batch:
        return None, errs

    lp_list = batched_lpips_fullimage(preds_batch, gts_batch, lpips_fn, device)
    for stem, lv in zip(ok_stems, lp_list):
        if with_per_view:
            per_view[stem]["lpips"] = lv
        lpips_vals.append(lv)

    n = len(psnrs)
    rec: Dict[str, Any] = {
        "n_views": n,
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
    }
    if with_per_view:
        rec["per_view"] = per_view
    return rec, errs


def _mean_metric_list(vals: List[Dict[str, float]], key: str) -> Optional[float]:
    xs = [v[key] for v in vals if key in v and v[key] is not None]
    if not xs:
        return None
    return float(np.mean(xs))


def eval_scene(
    scene_path: str,
    lpips_fn: torch.nn.Module,
    device: torch.device,
    *,
    with_per_frame: bool,
    with_per_view: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    frame_dirs = list_frame_dirs(scene_path)
    if not frame_dirs:
        return {"error": "no frame directories (expected {frame:05d}/)"}, []

    warn_all: List[str] = []
    scene_block: Dict[str, Any] = {}
    frames_out: Dict[str, Any] = {}
    branch_frame_rows: Dict[str, List[Dict[str, float]]] = {br: [] for br in BRANCHES}
    any_branch = False

    for fd in frame_dirs:
        fp = os.path.join(scene_path, fd)
        frame_entry: Dict[str, Any] = {}

        for br in BRANCHES:
            rec, errs = eval_frame_folder_branch(
                fp, br, lpips_fn, device, with_per_view=with_per_view
            )
            warn_all.extend([f"{fd}/{br}: {e}" for e in errs])
            if rec is None:
                continue
            branch_frame_rows[br].append(
                {"psnr": rec["psnr"], "ssim": rec["ssim"], "lpips": rec["lpips"]}
            )
            any_branch = True
            if with_per_frame:
                frame_entry[br] = rec if with_per_view else {
                    k: v for k, v in rec.items() if k != "per_view"
                }

        if with_per_frame and frame_entry:
            frames_out[fd] = frame_entry

    if not any_branch:
        return {
            "error": "no paired frames for neural or gs",
            "warnings": warn_all[:200] if warn_all else None,
        }, warn_all

    mean_over_frames: Dict[str, Dict[str, Any]] = {}
    for br in BRANCHES:
        rows = branch_frame_rows[br]
        if not rows:
            continue
        mean_over_frames[br] = {
            "psnr": _mean_metric_list(rows, "psnr"),
            "ssim": _mean_metric_list(rows, "ssim"),
            "lpips": _mean_metric_list(rows, "lpips"),
            "n_frames": len(rows),
        }
        scene_block[f"n_frames_{br}"] = len(rows)

    scene_block["mean_over_frames"] = mean_over_frames
    if with_per_frame and frames_out:
        scene_block["frames"] = frames_out
    if warn_all:
        scene_block["warnings"] = warn_all[:200]
        if len(warn_all) > 200:
            scene_block["warnings_note"] = f"truncated, total {len(warn_all)}"

    return scene_block, warn_all


def mean_over_scenes(
    scenes: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Arithmetic mean of scene-level ``mean_over_frames`` per branch."""
    pooled: Dict[str, List[Dict[str, float]]] = {br: [] for br in BRANCHES}
    for block in scenes.values():
        if block.get("error"):
            continue
        mof = block.get("mean_over_frames") or {}
        for br in BRANCHES:
            rec = mof.get(br)
            if isinstance(rec, dict) and rec.get("psnr") is not None:
                pooled[br].append(rec)

    out: Dict[str, Dict[str, Any]] = {}
    for br in BRANCHES:
        if not pooled[br]:
            continue
        out[br] = {
            "psnr": float(np.mean([x["psnr"] for x in pooled[br]])),
            "ssim": float(np.mean([x["ssim"] for x in pooled[br]])),
            "lpips": _mean_metric_list(pooled[br], "lpips"),
            "n_scenes": len(pooled[br]),
        }
    return out


def _fmt_branch(block: Dict[str, Any], branch: str) -> str:
    rec = block.get(branch)
    if not isinstance(rec, dict) or rec.get("psnr") is None:
        return f"{branch}=n/a"
    lp = rec.get("lpips")
    lp_s = f"{lp:.6f}" if lp is not None else "n/a"
    return (
        f"{branch} PSNR={rec['psnr']:.4f} SSIM={rec['ssim']:.6f} LPIPS={lp_s}"
    )


def _log_scene_result(scene_id: str, block: Dict[str, Any], warns: List[str]) -> None:
    if block.get("error"):
        _LOG.warning("  SCENE %s FAIL: %s", scene_id, block["error"])
        return
    mof = block.get("mean_over_frames") or {}
    parts = [_fmt_branch(mof, br) for br in BRANCHES if br in mof]
    nf = ", ".join(
        f"{br}={block.get(f'n_frames_{br}', '?')}" for br in BRANCHES if f"n_frames_{br}" in block
    )
    _LOG.info("  SCENE %s n_frames{%s} | %s", scene_id, nf or "?", " | ".join(parts) if parts else "n/a")
    for w in warns[:3]:
        _LOG.debug("    %s", w)


def _print_summary_table(task: str, mean_block: Dict[str, Dict[str, Any]]) -> None:
    header = f"{'task':<14} {'branch':<8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'n':>4}"
    print("\n" + header)
    print("-" * len(header))
    for br in BRANCHES:
        rec = mean_block.get(br)
        if not rec:
            print(f"{task:<14} {br:<8} {'n/a':>8} {'n/a':>8} {'n/a':>8} {'0':>4}")
            continue
        lp = rec.get("lpips")
        lp_s = f"{lp:8.6f}" if lp is not None else "     n/a"
        print(
            f"{task:<14} {br:<8} {rec['psnr']:8.4f} {rec['ssim']:8.6f} {lp_s} "
            f"{rec.get('n_scenes', 0):4d}"
        )


def eval_root(
    root: str,
    lpips_fn: torch.nn.Module,
    device: torch.device,
    *,
    scene_filter: Optional[List[str]],
    with_per_frame: bool,
    with_per_view: bool,
) -> Dict[str, Any]:
    if scene_filter is not None:
        scenes = [s for s in scene_filter if os.path.isdir(os.path.join(root, s))]
    else:
        scenes = list_scene_dirs(root)

    scenes_out: Dict[str, Any] = {}
    for si, scene_id in enumerate(scenes, start=1):
        scene_path = os.path.join(root, scene_id)
        _LOG.info("  scene %s/%s %s", si, len(scenes), scene_id)
        try:
            block, warns = eval_scene(
                scene_path,
                lpips_fn,
                device,
                with_per_frame=with_per_frame,
                with_per_view=with_per_view,
            )
        except (ValueError, FileNotFoundError) as ex:
            _LOG.warning("  scene=%s skip: %s", scene_id, ex)
            block = {"error": str(ex)}
            warns = []
        scenes_out[scene_id] = block
        _log_scene_result(scene_id, block, warns)

    return scenes_out


def main() -> None:
    from core.utils.model_card import ensure_reconstruction_benchmark_enabled  # noqa: E402

    ensure_reconstruction_benchmark_enabled()

    ap = argparse.ArgumentParser(
        description=(
            "PSNR / SSIM / LPIPS on white-background composited frames "
            "(pred*mask+1-mask vs gt*mask+1-mask, full image, no crop)."
        )
    )
    ap.add_argument(
        "--root",
        required=True,
        help=(
            "Run root with {scene_id}/{frame:05d}/gt|neural_render|gs_render/... "
            "(typically .../dna-rendering/<task>/<model_name>/)"
        ),
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            f"Reconstruction model tag (default: infer from --root basename, else "
            f"{DEFAULT_RECONSTRUCTION_MODEL_NAME}). "
            f"Allowed: {', '.join(RECONSTRUCTION_MODEL_NAMES)}"
        ),
    )
    ap.add_argument(
        "--task",
        choices=("novel-view", "novel-pose", "auto"),
        default="auto",
        help="Benchmark task label for JSON (auto: infer from --root path)",
    )
    ap.add_argument(
        "--out-meta",
        default=None,
        help="Output meta JSON (default: <root>/recon_metrics_<task>.meta.json)",
    )
    ap.add_argument(
        "--scenes",
        default=None,
        help="Optional comma-separated scene ids; default all subdirs of --root",
    )
    ap.add_argument(
        "--with-per-frame",
        action="store_true",
        help="Include per-frame metrics under each scene block",
    )
    ap.add_argument(
        "--with-per-view",
        action="store_true",
        help="Include per-camera (stem) metrics under each frame block",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--lpips-net", default="alex", choices=("alex", "vgg", "squeeze"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure_logging(args.log_level)
    root = os.path.abspath(os.path.expanduser(args.root))
    try:
        if args.model_name:
            model_name = require_reconstruction_model(args.model_name)
            inferred = infer_reconstruction_model_from_root(root)
            if model_name != inferred:
                _LOG.warning(
                    "--model-name %s differs from --root basename %s; using %s",
                    model_name,
                    inferred,
                    model_name,
                )
        else:
            model_name = infer_reconstruction_model_from_root(root)
    except ValueError as ex:
        _LOG.error("%s", ex)
        sys.exit(2)

    task = resolve_task(args.task, root)
    out_path = args.out_meta or default_out_meta(root, task)

    scene_filter: Optional[List[str]] = None
    if args.scenes:
        scene_filter = [s.strip() for s in args.scenes.split(",") if s.strip()]

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    import lpips as lpips_pkg  # noqa: WPS433

    lpips_fn = lpips_pkg.LPIPS(net=args.lpips_net, eval_mode=True, verbose=False).to(
        device
    )

    _LOG.info("root=%s task=%s model=%s device=%s", root, task, model_name, device)

    scenes_out = eval_root(
        root,
        lpips_fn,
        device,
        scene_filter=scene_filter,
        with_per_frame=args.with_per_frame,
        with_per_view=args.with_per_view,
    )

    if not scenes_out:
        _LOG.warning("no scenes under %s", root)
        sys.exit(1)

    mos = mean_over_scenes(scenes_out)
    report: Dict[str, Any] = {
        "root": root,
        "task": task,
        "model_name": model_name,
        "psnr_definition": PSNR_DEFINITION,
        "lpips_net": args.lpips_net,
        "branches": list(BRANCHES),
        "mean_over_scenes": mos,
        "scenes": scenes_out,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _LOG.info("Wrote %s", out_path)
    _print_summary_table(task, mos)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
