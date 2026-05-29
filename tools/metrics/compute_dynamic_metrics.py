#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic animation benchmark metrics: PSNR / SSIM / LPIPS on gs & neural vs GT.

Expected tree (from ``infer_eval_animation``)::

    <root>/<dataset>-<run_config>/<scene_id>/
        gt/rgb/{frame:05d}.png
        gt/mask/{frame:05d}.png
        view_{V:03d}/
            gs_rgb/{frame:05d}.png
            neural_rgb/{frame:05d}.png

Branches per frame: ``gs``, ``neural`` (full-image masked metrics). Each branch is
evaluated only when ``view_*/{gs_rgb,neural_rgb}`` pairs exist for that frame; one branch
may be present without the other.

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
    MODEL_CONFIG,
    require_animation_model,
)
from metric_utils import (
    PSNR_DEFINITION,
    branch_record,
    configure_logging,
    load_mask_float,
    load_rgb,
    metrics_triple,
)

_LOG = logging.getLogger(__name__)

BRANCHES = ("gs", "neural")


def discover_run_leaves(root: str, dataset: str) -> List[str]:
    prefix = f"{dataset}-"
    out: List[str] = []
    for name in sorted(os.listdir(root)):
        if name in ("logs",) or name.startswith("."):
            continue
        if not name.startswith(prefix):
            continue
        p = os.path.join(root, name)
        if os.path.isdir(p):
            out.append(name)
    return out


def _is_scene_dir(scene_path: str) -> bool:
    return os.path.isdir(os.path.join(scene_path, "gt", "rgb")) and os.path.isdir(
        os.path.join(scene_path, "gt", "mask")
    )


def list_scenes_in_run(run_path: str, scene_filter: Optional[List[str]] = None) -> List[str]:
    names: List[str] = []
    for name in sorted(os.listdir(run_path)):
        if name.startswith("."):
            continue
        p = os.path.join(run_path, name)
        if not os.path.isdir(p):
            continue
        if not _is_scene_dir(p):
            continue
        if scene_filter is not None and name not in scene_filter:
            continue
        names.append(name)
    return names


def list_view_dirs(scene_path: str) -> List[str]:
    return sorted(
        n
        for n in os.listdir(scene_path)
        if n.startswith("view_") and os.path.isdir(os.path.join(scene_path, n))
    )


def resolve_view_dir(scene_path: str, view_tag: str) -> Tuple[str, str]:
    """Return (view_dir_name, view_dir_abs). ``view_tag`` is ``auto`` or e.g. ``view_008``."""
    views = list_view_dirs(scene_path)
    if not views:
        raise FileNotFoundError(f"no view_* under {scene_path}")
    if view_tag == "auto":
        if len(views) == 1:
            name = views[0]
        else:
            raise ValueError(
                f"scene {scene_path}: multiple view_* ({views}); pass --view-tag explicitly"
            )
    else:
        name = view_tag if view_tag.startswith("view_") else f"view_{int(view_tag):03d}"
        if name not in views:
            raise FileNotFoundError(
                f"view dir {name} not in {scene_path} (available: {views})"
            )
    return name, os.path.join(scene_path, name)


def paired_frame_stems_for_branch(
    gt_rgb_dir: str,
    gt_mask_dir: str,
    pred_rgb_dir: str,
) -> List[str]:
    """Frame stems where GT rgb+mask and this branch's prediction all exist."""
    if not os.path.isdir(gt_rgb_dir) or not os.path.isdir(pred_rgb_dir):
        return []
    stems: List[str] = []
    for fname in sorted(os.listdir(gt_rgb_dir)):
        if not fname.lower().endswith(".png"):
            continue
        if not all(
            os.path.isfile(os.path.join(d, fname))
            for d in (gt_mask_dir, pred_rgb_dir)
        ):
            continue
        stems.append(fname)
    return stems


def eval_dynamic_frame_branch(
    branch: str,
    pred_path: str,
    gt_path: str,
    mask_path: str,
    lpips_fn: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate one frame for a single branch (full-image masked)."""
    pred = load_rgb(pred_path)
    gt = load_rgb(gt_path)
    mk = load_mask_float(mask_path)

    h, w = gt.shape[:2]
    if pred.shape[:2] != (h, w):
        raise ValueError(f"{branch} shape {pred.shape[:2]} != gt {(h, w)}")
    if mk.shape[:2] != (h, w):
        raise ValueError(f"mask shape {mk.shape[:2]} != gt {(h, w)}")

    return branch_record(*metrics_triple(pred, gt, mk, lpips_fn, device))


def _mean_branch_metrics(
    frame_rows: List[Dict[str, Dict[str, float]]],
    branch: str,
) -> Optional[Dict[str, float]]:
    vals: List[Dict[str, float]] = [r[branch] for r in frame_rows if branch in r]
    if not vals:
        return None
    return {
        "psnr": float(np.mean([v["psnr"] for v in vals])),
        "ssim": float(np.mean([v["ssim"] for v in vals])),
        "lpips": float(np.mean([v["lpips"] for v in vals])),
    }


_BRANCH_PRED_SUBDIR = {"gs": "gs_rgb", "neural": "neural_rgb"}


def eval_scene(
    scene_path: str,
    view_tag: str,
    lpips_fn: torch.nn.Module,
    device: torch.device,
    with_per_frame: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    view_name, view_dir = resolve_view_dir(scene_path, view_tag)
    gt_rgb_dir = os.path.join(scene_path, "gt", "rgb")
    gt_mask_dir = os.path.join(scene_path, "gt", "mask")

    warn_all: List[str] = []
    scene_block: Dict[str, Any] = {"view": view_name}
    frames_out: Dict[str, Any] = {}
    any_branch = False

    for br in BRANCHES:
        pred_dir = os.path.join(view_dir, _BRANCH_PRED_SUBDIR[br])
        stems = paired_frame_stems_for_branch(gt_rgb_dir, gt_mask_dir, pred_dir)
        if not stems:
            continue

        branch_rows: List[Dict[str, Dict[str, float]]] = []
        for stem in stems:
            try:
                metrics = eval_dynamic_frame_branch(
                    br,
                    os.path.join(pred_dir, stem),
                    os.path.join(gt_rgb_dir, stem),
                    os.path.join(gt_mask_dir, stem),
                    lpips_fn,
                    device,
                )
            except Exception as e:  # noqa: BLE001
                warn_all.append(f"{br}/{stem}: {e}")
                continue
            branch_rows.append({br: metrics})
            if with_per_frame:
                frames_out.setdefault(stem, {})[br] = metrics

        if not branch_rows:
            continue

        m = _mean_branch_metrics(branch_rows, br)
        if m is not None:
            scene_block[br] = m
            scene_block[f"n_frames_{br}"] = len(branch_rows)
            any_branch = True

    if not any_branch:
        return {
            "error": "no paired frames for gs or neural",
            "view": view_name,
            "warnings": warn_all[:200] if warn_all else None,
        }, warn_all

    if with_per_frame and frames_out:
        scene_block["frames"] = frames_out
    if warn_all:
        scene_block["warnings"] = warn_all[:200]
        if len(warn_all) > 200:
            scene_block["warnings_note"] = f"truncated, total {len(warn_all)}"
    return scene_block, warn_all


def mean_over_scenes(scenes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Arithmetic mean of scene-level branch metrics (skip scenes with errors)."""
    pooled: Dict[str, List[Dict[str, float]]] = {br: [] for br in BRANCHES}
    for block in scenes.values():
        if block.get("error"):
            continue
        for br in BRANCHES:
            if br in block and isinstance(block[br], dict):
                pooled[br].append(block[br])

    out: Dict[str, Dict[str, Any]] = {}
    for br in BRANCHES:
        if not pooled[br]:
            continue
        out[br] = {
            "psnr": float(np.mean([x["psnr"] for x in pooled[br]])),
            "ssim": float(np.mean([x["ssim"] for x in pooled[br]])),
            "lpips": float(np.mean([x["lpips"] for x in pooled[br]])),
            "n_scenes": len(pooled[br]),
        }
    return out


def _fmt_branch(block: Dict[str, Any], branch: str) -> str:
    rec = block.get(branch)
    if not isinstance(rec, dict):
        return f"{branch}=n/a"
    return (
        f"{branch} PSNR={rec['psnr']:.4f} SSIM={rec['ssim']:.6f} LPIPS={rec['lpips']:.6f}"
    )


def _log_scene_result(
    run_name: str,
    scene_id: str,
    block: Dict[str, Any],
    warns: List[str],
) -> None:
    if block.get("error"):
        _LOG.warning("  SCENE run=%s scene=%s FAIL: %s", run_name, scene_id, block["error"])
        return
    parts = [_fmt_branch(block, br) for br in BRANCHES if br in block]
    nf = ", ".join(
        f"{br}={block.get(f'n_frames_{br}', '?')}"
        for br in BRANCHES
        if f"n_frames_{br}" in block
    )
    _LOG.info(
        "  SCENE run=%s scene=%s n_frames{%s} view=%s | %s",
        run_name,
        scene_id,
        nf or "?",
        block.get("view"),
        " | ".join(parts) if parts else "no branch metrics",
    )
    for w in warns[:5]:
        _LOG.debug("    %s", w)


def _print_summary_table(runs: Dict[str, Any]) -> None:
    header = f"{'run':<48} {'branch':<12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'n':>4}"
    print("\n" + header)
    print("-" * len(header))
    for run_name in sorted(runs.keys()):
        mos = (runs[run_name] or {}).get("mean_over_scenes") or {}
        for br in BRANCHES:
            rec = mos.get(br)
            if not rec:
                print(f"{run_name:<48} {br:<12} {'n/a':>8} {'n/a':>8} {'n/a':>8} {'0':>4}")
                continue
            print(
                f"{run_name:<48} {br:<12} "
                f"{rec['psnr']:8.4f} {rec['ssim']:8.6f} {rec['lpips']:8.6f} "
                f"{rec.get('n_scenes', 0):4d}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PSNR / SSIM / LPIPS for dynamic animation benchmark exports."
    )
    ap.add_argument("--root", required=True, help="Benchmark root (contains <dataset>-* runs)")
    ap.add_argument("--dataset", required=True, help="Dataset prefix, e.g. neuman")
    ap.add_argument(
        "--model-name",
        type=str,
        default="LHMPP-700M-PixelShuffle",
        choices=sorted(MODEL_CONFIG.keys()),
        help=(
            "Model that produced the exports (must be a non-reconstruction animation "
            "model card; reconstruction DNA models are rejected)."
        ),
    )
    ap.add_argument(
        "--out-meta",
        "--out-json",
        dest="out_meta",
        default=None,
        help="Output meta JSON (default: <root>/dynamic_metrics_<dataset>.meta.json)",
    )
    ap.add_argument(
        "--view-tag",
        default="auto",
        help="view_* dir name or 'auto' when exactly one view_* exists per scene",
    )
    ap.add_argument(
        "--scenes",
        default=None,
        help="Optional comma-separated scene ids; default evaluates every scene under each run",
    )
    ap.add_argument(
        "--with-per-frame",
        action="store_true",
        help="Include per-frame metrics under each scene block",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--lpips-net", default="alex", choices=("alex", "vgg", "squeeze"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure_logging(args.log_level)
    try:
        model_name = require_animation_model(args.model_name)
    except ValueError as ex:
        _LOG.error("%s", ex)
        sys.exit(2)

    root = os.path.abspath(os.path.expanduser(args.root))
    out_path = args.out_meta or os.path.join(
        root, f"dynamic_metrics_{args.dataset}.meta.json"
    )
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

    run_names = discover_run_leaves(root, args.dataset)
    if not run_names:
        _LOG.warning("no run directories matching %s-* under %s", args.dataset, root)
        sys.exit(1)

    report: Dict[str, Any] = {
        "root": root,
        "dataset": args.dataset,
        "model_name": model_name,
        "psnr_definition": PSNR_DEFINITION,
        "lpips_net": args.lpips_net,
        "view_tag": args.view_tag,
        "runs": {},
    }

    _LOG.info(
        "root=%s dataset=%s model=%s runs=%s device=%s",
        root,
        args.dataset,
        model_name,
        len(run_names),
        device,
    )

    for ri, run_name in enumerate(run_names, start=1):
        run_path = os.path.join(root, run_name)
        scenes = list_scenes_in_run(run_path, scene_filter)
        _LOG.info("[%s/%s] run=%s scenes=%s", ri, len(run_names), run_name, len(scenes))

        scenes_out: Dict[str, Any] = {}
        for si, scene_id in enumerate(scenes, start=1):
            scene_path = os.path.join(run_path, scene_id)
            _LOG.info("  run=%s scene %s/%s %s", run_name, si, len(scenes), scene_id)
            try:
                block, warns = eval_scene(
                    scene_path,
                    args.view_tag,
                    lpips_fn,
                    device,
                    args.with_per_frame,
                )
            except (ValueError, FileNotFoundError) as e:
                _LOG.warning("  scene=%s skip: %s", scene_id, e)
                block = {"error": str(e)}
                warns = []
            scenes_out[scene_id] = block
            _log_scene_result(run_name, scene_id, block, warns)

        mos = mean_over_scenes(scenes_out)
        report["runs"][run_name] = {
            "mean_over_scenes": mos,
            "scenes": scenes_out,
        }
        mos_parts = [
            _fmt_branch(mos, br) if mos.get(br) else f"{br}=n/a" for br in BRANCHES
        ]
        _LOG.info("  RUN_MEAN %s | %s", run_name, " | ".join(mos_parts))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _LOG.info("Wrote %s", out_path)
    _print_summary_table(report["runs"])
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
