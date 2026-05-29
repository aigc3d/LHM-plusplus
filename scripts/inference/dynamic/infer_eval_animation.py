"""LHMPP dynamic benchmark inference (masked motion + Gaussian + neural paste).

Only ``--model-name`` is required for the checkpoint: weights are fetched via
``AutoModelQuery`` and training yaml from ``MODEL_CONFIG`` (same as ``app.py``).
Optional ``--model-path`` overrides the weight directory only; no separate ``-c`` / ``-w``.

Loads :class:`~core.datasets.evaluation.VideoHumanDynamicBenchmarkAnimation`. JSON manifests
(e.g. ``eval_lhmpp_neuman.json``) use ``img_path`` as the animation timeline (one evaluation target
per index ``t``) and ``{N}_view`` as the ``N`` fixed reference-view frame filenames; ``-v`` must
match ``N``. Motion is read from unpack folders ``{root_dirs}/{uid}/`` (``smplx_params`` +
``samurai_seg``); exports go under ``{--output-dir or ./exps/benchmarks/dynamic}/{dataset}-{H}x{W}-{predshape|datashape}/``;
under that, per-scene ``{uid}/gt/{{rgb,mask}}/`` plus
``{{uid}}/view_{V:03d}/neural_rgb`` and ``neural_mask`` (default). Pass ``--gs-output`` to also
write ``gs_rgb`` / ``gs_mask`` (Gaussian rasterization, no neural refiner).
Also writes ``{{uid}}/ref_img.png``: one horizontal strip of the JSON ``{{N}}_view`` reference inputs (not ``img_path`` targets).

Motion timeline GT RGB always uses ``imgs_png`` + mask composite per scene.
**Reference** views are uniformly sampled from ``{scene}/ref_imgs_png/*.png`` (JSON ``{N}_view`` ignored).
Reference ``source_rgbs`` PadRatio canvas defaults to **1036×616**; use ``--src-height`` (**840** or **1036**,
divisible by DINO patch **14**) or yaml ``motion_ref_source_tgt_max_size``.

Distributed sharding uses the same ``bins = ceil(n/split)`` case range as other inference CLIs;
no dependency on DNA-Rendering benchmark modules. Per JSON case / scene uid, Gaussian + latent reconstruction
runs once on the **first timeline index visited** (`time_indices[0]`); every frame thereafter only runs
``animation_infer`` against that cache.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DYNAMIC_DIR = os.path.abspath(os.path.dirname(__file__))
for _p in (_REPO_ROOT, _DYNAMIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch._dynamo.config.disable = True

from a4o_train_yaml_dataset_kw import (
    AUTO_SUBSET_NAME,
    SUBSET_NAME_FALLBACKS,
    build_a4o_tar_dataset_constructor_kwargs,
    pick_dataset_subset,
)
from core.datasets.evaluation import (
    DYNAMIC_BENCHMARK_CHOICES,
    DYNAMIC_BENCHMARK_PRESETS,
    DYNAMIC_BENCHMARK_REF_SUBDIR,
    VideoHumanDynamicBenchmarkAnimation,
)
from core.models.animation import patch_model_animation_infer
from core.utils.model_card import MODEL_CONFIG, require_animation_model
from core.utils.model_download_utils import AutoModelQuery, prior_model_check
from scripts.inference.app_inference import build_app_model, parse_app_configs
from video_benchmark_infer_helpers import (
    _cfg_get,
    benchmark_mask_animation_infer_from_gs_cache,
    build_benchmark_animation_gs_cache,
    _cfg_ref_imgs_subdir,
    _cfg_ref_source_height,
    patch_benchmark_item_ref_pad_resize,
    patch_processing_pipeline_ref_resolution,
    query_target_motion_for_frames,
    save_scene_reference_strip_png,
    save_unpadded_quads_under_view_dirs,
)

_LOG = logging.getLogger(__name__)

LHMPP_BENCHMARK_PRESET_CHOICES = DYNAMIC_BENCHMARK_CHOICES
LHMPP_BENCHMARK_PRESETS = DYNAMIC_BENCHMARK_PRESETS
LHMPPDynamicBenchmarkAnimation = VideoHumanDynamicBenchmarkAnimation

_DINO_PATCH_ALIGN = 14
_DEFAULT_SRC_HEIGHT = 1036
_DATASHAPE_REF_HEIGHT = 840
_DEFAULT_DYNAMIC_EXPORT_BASE = os.path.join(".", "exps", "benchmarks", "dynamic")


def _reference_canvas_hw_from_ds_kw(ds_kw: Dict[str, Any]) -> Tuple[int, int]:
    """Approximate PadRatio canvas ``H×W`` from ``processing_pipeline`` (same rule as PadRatio width rounding)."""

    ratio = 5.0 / 3.0
    th = 840
    ep = _DINO_PATCH_ALIGN
    for step in ds_kw.get("processing_pipeline") or []:
        if step.get("name") not in ("PadRatioWithScale", "PadRatio"):
            continue
        raw_r = step.get("target_ratio")
        if raw_r is not None:
            ratio = float(eval(raw_r)) if isinstance(raw_r, str) else float(raw_r)
        tlist = step.get("tgt_max_size_list") or [840]
        if isinstance(tlist, (list, tuple)) and len(tlist):
            th = int(max(int(x) for x in tlist))
        break
    tw = int(th / ratio) // ep * ep
    return th, tw


def _dynamic_export_leaf_dir_name(
    args: argparse.Namespace,
    meta_path: str,
    *,
    ref_canvas_h: int,
    ref_canvas_w: int,
    use_pred_shape: bool,
    gs_output: bool,
) -> str:
    """Run-specific folder under ``--output-dir``: ``{dataset}-{H}x{W}-{shape}-{render}``."""

    if getattr(args, "dataset", None):
        base = str(args.dataset).strip() or "custom"
    else:
        base = os.path.splitext(os.path.basename(os.path.abspath(meta_path)))[0] or "custom"
    shape_seg = "predshape" if use_pred_shape else "datashape"
    render_seg = "gs" if gs_output else "neural"
    return f"{base}-{int(ref_canvas_h)}x{int(ref_canvas_w)}-{shape_seg}-{render_seg}"


def _resolve_train_yaml_from_model_card(model_name: str) -> str:
    """``MODEL_CONFIG[model_name]`` → absolute path under repo root when relative."""
    rel = MODEL_CONFIG[model_name]
    if os.path.isabs(rel) and os.path.isfile(rel):
        return rel
    cand = os.path.normpath(os.path.join(_REPO_ROOT, str(rel).lstrip("./")))
    if os.path.isfile(cand):
        return cand
    if os.path.isfile(rel):
        return os.path.abspath(rel)
    raise FileNotFoundError(
        f"training yaml for model card {model_name!r} not found: tried {cand!r} and {rel!r}"
    )


def _resolve_model_card(
    model_name: str,
    *,
    model_path: Optional[str] = None,
    pretrained_dir: str = "./pretrained_models",
) -> Tuple[str, str]:
    """Resolve (checkpoint_dir, train_yaml) from ``--model-name`` only."""
    prior_model_check(save_dir=pretrained_dir)
    train_yaml = _resolve_train_yaml_from_model_card(model_name)
    if model_path:
        ckpt_dir = os.path.abspath(os.path.expanduser(model_path))
    else:
        ckpt_dir = AutoModelQuery(save_dir=pretrained_dir).query(model_name)
    return ckpt_dir, train_yaml


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _build_model(cfg):
    """Load via ``build_app_model`` (model card path), then patch ``animation_infer`` for GS export."""
    lhm = build_app_model(cfg)
    patch_model_animation_infer(lhm)
    return lhm


def _normalize_time_indices(
    n_targets: int, only_time_idx: Optional[int], label: str
) -> Sequence[int]:
    """Frame indices ``t`` in ``[0, n_targets)`` to visit for one case."""

    if only_time_idx is None:
        return list(range(n_targets))
    if only_time_idx < 0 or only_time_idx >= n_targets:
        raise IndexError(
            f"{label}={only_time_idx} out of range for this case (num_targets={n_targets})."
        )
    return [only_time_idx]


@torch.no_grad()
def lhmp_benchmark_animation_inference(
    lhm,
    dataset: LHMPPDynamicBenchmarkAnimation,
    output_root: str,
    *,
    infer_cfg,
    motion_root_dirs: Optional[str],
    split: int,
    gpus: int,
    ref_view_count: int,
    infer_batch_size: int = 40,
    only_case_idx: Optional[int] = None,
    only_time_idx: Optional[int] = None,
    use_pred_shape: bool = False,
    gs_output: bool = False,
) -> None:
    if lhm is not None:
        lhm.cuda()
        lhm.eval()

    os.makedirs(output_root, exist_ok=True)
    _LOG.info("export root: %s", output_root)
    roots = motion_root_dirs or dataset.root_dirs
    roots = os.path.abspath(os.path.expanduser(str(roots)))

    ref_inputs_subdir = _cfg_ref_imgs_subdir(infer_cfg, DYNAMIC_BENCHMARK_REF_SUBDIR)

    def _ref_sample_kw(scene_uid: str) -> dict:
        return dict(
            ref_uniform_inputs=True,
            ref_inputs_dir=os.path.join(roots, scene_uid, ref_inputs_subdir),
        )

    ncases = len(dataset)
    if only_case_idx is not None:
        if only_case_idx < 0 or only_case_idx >= ncases:
            raise IndexError(
                f"only_case_idx={only_case_idx} out of range [0, {ncases})."
            )
        case_indices: List[int] = [int(only_case_idx)]
        if split != 1 or gpus != 0:
            _LOG.warning(
                "only_case_idx is set; ignoring split=%s / gpus=%s (running a single JSON case).",
                split,
                gpus,
            )
    else:
        bins = int(math.ceil(ncases / split))
        case_lo = bins * int(gpus)
        case_hi = min(bins * (int(gpus) + 1), ncases)
        case_indices = list(range(case_lo, case_hi))

    n_shard = len(case_indices)
    for shard_i, case_idx in enumerate(case_indices):
        uid = str(dataset.uids[case_idx]["id"])
        motion_scene = os.path.join(roots, uid)
        n_t = dataset.num_targets(case_idx)
        try:
            time_indices = _normalize_time_indices(n_t, only_time_idx, "only_time_idx")
        except IndexError as ex:
            _LOG.warning(
                "skip case_idx=%s: %s",
                case_idx,
                ex,
            )
            print(f"skip case_idx={case_idx}: {ex}", file=sys.stderr)
            continue

        t_anchor = int(time_indices[0])
        try:
            _LOG.info(
                "uid=%s refs: linspace sample %s views from %s",
                uid,
                ref_view_count,
                os.path.join(roots, uid, ref_inputs_subdir),
            )
            item_anchor = dataset.sample(case_idx, t_anchor, **_ref_sample_kw(uid))
            patch_benchmark_item_ref_pad_resize(
                item_anchor,
                motion_scene_root=os.path.join(roots, uid),
                ref_subdir=ref_inputs_subdir,
                aspect_standard=float(dataset.aspect_standard),
            )
        except Exception as ex:
            _LOG.warning("skip case=%s anchor t=%s : %s", case_idx, t_anchor, ex)
            print(f"skip case={case_idx} anchor t={t_anchor}: {ex}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            continue

        fid_anchor = int(item_anchor["smpl_frame_id"].detach().cpu().item())
        try:
            motion_anchor = query_target_motion_for_frames(
                motion_scene, [fid_anchor], infer_cfg
            )
        except Exception as ex:
            _LOG.warning("motion anchor uid=%s frame=%s: %s", uid, fid_anchor, ex)
            print(
                f"skip motion anchor uid={uid} frame={fid_anchor}: {ex}",
                file=sys.stderr,
            )
            print(traceback.format_exc(), file=sys.stderr)
            continue

        torch.cuda.empty_cache()
        try:
            gs_cache = build_benchmark_animation_gs_cache(
                lhm,
                dict(item_anchor),
                motion_anchor,
                use_pred_shape=use_pred_shape,
            )
        except Exception as ex:
            _LOG.exception("reconstruction (infer_single_view) failed uid=%s anchor_fid=%s", uid, fid_anchor)
            print(
                f"reconstruction failed uid={uid} anchor_fid={fid_anchor}: {ex}",
                file=sys.stderr,
            )
            print(traceback.format_exc(), file=sys.stderr)
            continue

        _LOG.info(
            "scene cache ready uid=%s anchor_t=%s anchor_fid=%s (reuse for %s timeline frame(s))",
            uid,
            t_anchor,
            fid_anchor,
            len(time_indices),
        )

        ref_path = save_scene_reference_strip_png(
            output_root,
            uid,
            item_anchor["source_rgbs"],
            item_anchor.get("ref_imgs_bool"),
        )
        if ref_path:
            _LOG.info("saved reference strip once: %s", ref_path)

        for t in time_indices:
            try:
                item = dataset.sample(case_idx, t, **_ref_sample_kw(uid))
                patch_benchmark_item_ref_pad_resize(
                    item,
                    motion_scene_root=os.path.join(roots, uid),
                    ref_subdir=ref_inputs_subdir,
                    aspect_standard=float(dataset.aspect_standard),
                )
            except Exception as ex:
                _LOG.warning("skip case=%s t=%s : %s", case_idx, t, ex)
                print(f"skip case={case_idx} t={t}: {ex}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

            fid = int(item["smpl_frame_id"].detach().cpu().item())
            stem = f"{fid:05d}.png"
            target_manifest_name = str(dataset.uids[case_idx]["img_path"][t])
            out_uid_dir = os.path.join(output_root, uid)
            view_tag = f"view_{int(ref_view_count):03d}"
            _LOG.info(
                "animation infer [%s/%s cases] shard %s/%s: uid=%s case_idx=%s t=%s manifest_target=%s frame_id=%s stem=%s out_dir=%s view_dir=%s",
                case_idx + 1,
                ncases,
                shard_i + 1,
                n_shard,
                uid,
                case_idx,
                t,
                target_manifest_name,
                fid,
                stem,
                out_uid_dir,
                view_tag,
            )

            try:
                motion_pack = query_target_motion_for_frames(
                    motion_scene, [fid], infer_cfg
                )
            except Exception as ex:
                _LOG.warning("motion load uid=%s frame=%s: %s", uid, fid, ex)
                print(
                    f"skip motion uid={uid} frame={fid}: {ex}",
                    file=sys.stderr,
                )
                print(traceback.format_exc(), file=sys.stderr)
                continue

            torch.cuda.empty_cache()
            try:
                anim_out = benchmark_mask_animation_infer_from_gs_cache(
                    lhm,
                    dict(item),
                    motion_pack,
                    gs_cache,
                    batch_size=int(infer_batch_size),
                    use_pred_shape=use_pred_shape,
                    export_gs_outputs=gs_output,
                )
                if gs_output:
                    pred_rgb, pred_m, gs_r, gsm = anim_out
                else:
                    pred_rgb, pred_m = anim_out
                    gs_r, gsm = None, None
            except Exception as ex:
                _LOG.exception("animation_infer failed uid=%s frame=%s", uid, fid)
                print(
                    f"animation_infer failed uid={uid} frame={fid}: {ex}",
                    file=sys.stderr,
                )
                print(traceback.format_exc(), file=sys.stderr)
                continue

            n_views = pred_rgb.shape[0]
            view_dir_abs = os.path.join(out_uid_dir, view_tag)
            for vi in range(n_views):
                off = motion_pack["unpad_offset"][vi]
                gt_u8 = motion_pack["unpad_gt_img_list"][vi]
                mk_u8 = motion_pack["unpad_mask"][vi]
                save_unpadded_quads_under_view_dirs(
                    output_root,
                    uid,
                    ref_view_count,
                    stem,
                    pred_rgb[vi],
                    pred_m[vi],
                    off,
                    gt_u8,
                    mk_u8,
                    gs_rgb_f=gs_r[vi] if gs_output else None,
                    gs_mask_f=gsm[vi] if gs_output else None,
                    export_gs=gs_output,
                )

            gt_rgb_fp = os.path.join(out_uid_dir, "gt", "rgb", stem)
            gt_mask_fp = os.path.join(out_uid_dir, "gt", "mask", stem)
            neu_fp = os.path.join(view_dir_abs, "neural_rgb", stem)
            log_msg = (
                "saved uid=%s stem=%s n_views=%s export_root=%s scene_dir=%s | "
                "gt_rgb=%s | gt_mask=%s | neural_rgb=%s"
            )
            log_args = [
                uid,
                stem,
                n_views,
                output_root,
                out_uid_dir,
                gt_rgb_fp,
                gt_mask_fp,
                neu_fp,
            ]
            if gs_output:
                gs_rgb_fp = os.path.join(view_dir_abs, "gs_rgb", stem)
                log_msg += " | gs_rgb=%s | (also neural_mask, gs_mask under %s)"
                log_args.extend([gs_rgb_fp, view_dir_abs])
            else:
                log_msg += " | (also neural_mask under %s)"
                log_args.append(view_dir_abs)
            _LOG.info(log_msg, *log_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LHMPP dynamic benchmark masked animation inference (SelfCapture / NeuMan / Vid2Avatar). "
            "Default: neural renderer (LHMPP-700M-PixelShuffle model card). "
            "Use --gs-output to also export GS raster tiles."
        )
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        type=str,
        default="LHMPP-700M-PixelShuffle",
        choices=list(MODEL_CONFIG.keys()),
        help=(
            "Model card key in ``MODEL_CONFIG``: auto-downloads weights (``pretrained_models/``) "
            "and loads the paired training yaml. No ``-c`` / ``-w`` needed. "
            f"Choices: {', '.join(MODEL_CONFIG.keys())}."
        ),
    )
    parser.add_argument(
        "--model-path",
        "--model_path",
        type=str,
        default=None,
        help=(
            "Optional: local checkpoint directory only. Config still comes from ``MODEL_CONFIG`` "
            "for ``--model-name``; omit to use Hub/ModelScope download via AutoModelQuery."
        ),
    )
    parser.add_argument(
        "--dataset-subset",
        type=str,
        default=AUTO_SUBSET_NAME,
        help=(
            "Model-card training yaml ``dataset.subsets`` entry **name**, or ``auto`` to pick the "
            f"first match among: {', '.join(SUBSET_NAME_FALLBACKS)}. Not the same as ``--dataset`` "
            "(benchmark paths)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=LHMPP_BENCHMARK_PRESET_CHOICES,
        metavar="PRESET",
        help=(
            f"Shortcut preset (same paths as ``lhmpp_dynamic_benchmark_presets``): "
            f"{', '.join(LHMPP_BENCHMARK_PRESET_CHOICES)}. "
            "Otherwise pass both ``--root-dirs`` and ``--meta-path``."
        ),
    )
    parser.add_argument(
        "-v",
        "--view",
        type=int,
        default=8,
        help="Reference view count matching JSON `{v}_view` keys (1,2,4,8,16).",
    )
    parser.add_argument(
        "--root-dirs",
        type=str,
        default=None,
        help=(
            "Rec-MV unpacked root (scene folders `{id}` with motion/masks). "
            "Required if ``--dataset`` is omitted; optional override when ``--dataset`` is set."
        ),
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default=None,
        help=(
            "Benchmark JSON manifest: scene ``id``, ``img_path`` (animation timeline targets), "
            "``{v}_view`` (reference views). Required if ``--dataset`` is omitted; optional override "
            "when ``--dataset`` is set."
        ),
    )
    parser.add_argument(
        "--motion-root-dirs",
        type=str,
        default=None,
        help="Motion folder root defaulting to `--root-dirs` if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Base directory for exports; results are written to "
            "``{output-dir}/{dataset}-{H}x{W}-{predshape|datashape}-{neural|gs}/`` "
            "(render mode: default ``neural``, ``--gs-output`` → ``gs``). Default base if omitted: "
            f"``{_DEFAULT_DYNAMIC_EXPORT_BASE}``."
        ),
    )
    parser.add_argument(
        "-s",
        "--split",
        help="Distributed split factor (shard count; case indices use bins = ceil(ncases/split)).",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-g",
        "--gpus",
        help="Shard id in [0, split)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Do not load checkpoint or write outputs; only instantiate the dataset (and optionally "
            "``dataset.sample``). Pair with ``--only-case`` / ``--only-time`` instead of fixed (0,0)."
        ),
    )
    parser.add_argument(
        "--only-case",
        type=int,
        default=None,
        metavar="CASE_IDX",
        help=(
            "0-based JSON case index ``len(meta)==cases``. Full run: inference only this case "
            "(ignores `-s`/`-g`). Debug: smoke ``sample(case_idx , t)`` only."
        ),
    )
    parser.add_argument(
        "--only-time",
        type=int,
        default=None,
        metavar="T",
        help=(
            "Optional timeline index ``t`` for the chosen case (within ``num_targets(case)``); "
            "omit to run every frame row for that case."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--infer-batch-size",
        type=int,
        default=40,
        help="Chunks along motion ``camera_size`` for animation_infer batches.",
    )
    parser.add_argument(
        "--use-pred-shape",
        action="store_true",
        help=(
            "Same as top-level config ``use_pred_shape: true`` (merged into the loaded yaml after "
            "parse). Uses SMPL-X body shape predicted by the model (``infer_single_view`` ``pred_shape``) "
            "for animation_infer + GS cache; default ``false`` uses per-frame betas from motion JSON. "
            "Ignored when ``--src-height`` (or yaml ``motion_ref_source_tgt_max_size``) "
            f"is not {_DATASHAPE_REF_HEIGHT}."
        ),
    )
    parser.add_argument(
        "--gs-output",
        "--gs_output",
        action="store_true",
        help=(
            "Also export Gaussian splatting rasterization (``gs_rgb`` / ``gs_mask`` under each view). "
            "Default: neural renderer outputs only (``neural_rgb`` / ``neural_mask``)."
        ),
    )
    parser.add_argument(
        "--ref-subdir",
        type=str,
        default=DYNAMIC_BENCHMARK_REF_SUBDIR,
        metavar="DIRNAME",
        help=(
            f"Per-scene reference PNG folder under each scene (default ``{DYNAMIC_BENCHMARK_REF_SUBDIR}``). "
            "Timeline GT stays ``imgs_png`` + mask composite."
        ),
    )
    parser.add_argument(
        "--src-height",
        type=int,
        default=_DEFAULT_SRC_HEIGHT,
        metavar="H",
        help=(
            f"PadRatio reference canvas height for ``source_rgbs`` (default **{_DEFAULT_SRC_HEIGHT}**; "
            f"also **{_DATASHAPE_REF_HEIGHT}**). Must be divisible by DINO ViT patch size 14. "
            f"When not {_DATASHAPE_REF_HEIGHT}, uses **datashape** (motion JSON betas); "
            "``--use-pred-shape`` is disabled. Override via yaml ``motion_ref_source_tgt_max_size``."
        ),
    )
    args = parser.parse_args()

    try:
        args.model_name = require_animation_model(args.model_name)
    except ValueError as ex:
        parser.error(str(ex))

    if args.dataset:
        preset = LHMPP_BENCHMARK_PRESETS[args.dataset]
        root_dirs = args.root_dirs if args.root_dirs is not None else preset["root_dirs"]
        meta_path = args.meta_path if args.meta_path is not None else preset["meta_path"]
    elif args.root_dirs and args.meta_path:
        root_dirs, meta_path = args.root_dirs, args.meta_path
    else:
        parser.error(
            "Specify --dataset {{{}}} or both --root-dirs and --meta-path.".format(
                ",".join(LHMPP_BENCHMARK_PRESET_CHOICES),
            ),
        )

    model_path, model_config = _resolve_model_card(
        args.model_name,
        model_path=args.model_path,
    )

    model_cards = {
        args.model_name: {
            "model_path": model_path,
            "model_config": model_config,
        }
    }

    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": args.model_name,
            "APP_TYPE": "infer.human_lrm_a4o",
            "NUMBA_THREADING_LAYER": "omp",
        }
    )

    Accelerator()
    cfg, cfg_train = parse_app_configs(model_cards)
    _LOG.info(
        "model card %s | weights=%s | config=%s (from MODEL_CONFIG, no manual -c/-w)",
        args.model_name,
        model_path,
        model_config,
    )
    ref_sub = str(args.ref_subdir).strip() or DYNAMIC_BENCHMARK_REF_SUBDIR
    ref_patch = {
        "motion_ref_imgs_subdir": ref_sub,
        "motion_ref_source_tgt_max_size": int(args.src_height),
    }
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.merge(cfg, OmegaConf.create(ref_patch))
        OmegaConf.resolve(cfg)
    elif isinstance(cfg, dict):
        cfg = {**cfg, **ref_patch}
    if args.use_pred_shape:
        ups = {"use_pred_shape": True}
        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.merge(cfg, OmegaConf.create(ups))
            OmegaConf.resolve(cfg)
        elif isinstance(cfg, dict):
            cfg = {**cfg, **ups}
    _configure_logging(args.log_level or str(cfg.get("logger", "INFO")))
    if OmegaConf.is_config(cfg):
        OmegaConf.resolve(cfg)

    OmegaConf.resolve(cfg_train)
    subset = pick_dataset_subset(cfg_train.dataset.subsets, args.dataset_subset)
    ds_kw = build_a4o_tar_dataset_constructor_kwargs(cfg_train, subset, args.view)

    ref_th, ref_tw = _reference_canvas_hw_from_ds_kw(ds_kw)
    ref_tgt_h = _cfg_ref_source_height(cfg, int(args.src_height))
    ref_th, ref_tw = patch_processing_pipeline_ref_resolution(ds_kw, tgt_height=ref_tgt_h)
    _LOG.info(
        "Reference PadRatio canvas: %d×%d (multiples of DINO patch 14); "
        "``source_rgbs`` loaded from ``ref_imgs_png`` then resized to this canvas.",
        ref_th,
        ref_tw,
    )

    ds = LHMPPDynamicBenchmarkAnimation(
        root_dirs=os.path.abspath(os.path.expanduser(root_dirs)),
        meta_path=os.path.abspath(os.path.expanduser(meta_path)),
        ref_view_count=int(args.view),
        **ds_kw,
    )

    use_pred_shape = bool(_cfg_get(cfg, "use_pred_shape", False))
    if ref_tgt_h != _DATASHAPE_REF_HEIGHT:
        if use_pred_shape:
            _LOG.warning(
                "Reference height %d != %d: forcing datashape (motion JSON betas); "
                "``--use-pred-shape`` / ``use_pred_shape`` ignored.",
                ref_tgt_h,
                _DATASHAPE_REF_HEIGHT,
            )
        use_pred_shape = False
    gs_output = bool(args.gs_output)
    leaf = _dynamic_export_leaf_dir_name(
        args,
        meta_path,
        ref_canvas_h=ref_th,
        ref_canvas_w=ref_tw,
        use_pred_shape=use_pred_shape,
        gs_output=gs_output,
    )
    out_base_raw = (args.output_dir or "").strip()
    if out_base_raw:
        out_base = os.path.abspath(os.path.expanduser(out_base_raw))
    else:
        out_base = os.path.abspath(_DEFAULT_DYNAMIC_EXPORT_BASE)
    output_root = os.path.join(out_base, leaf)
    _LOG.info(
        "export under base %s leaf %s -> %s",
        out_base,
        leaf,
        os.path.abspath(output_root),
    )

    if args.debug:
        preset_note = f" (--dataset={args.dataset})" if args.dataset else ""
        print(
            f"dataset len={len(ds)} preset{preset_note} root_dirs={os.path.abspath(os.path.expanduser(root_dirs))}"
        )
        if not len(ds):
            return
        case_idx = int(args.only_case) if args.only_case is not None else 0
        if case_idx < 0 or case_idx >= len(ds):
            raise IndexError(f"--only-case={case_idx} out of range [0, {len(ds)})")

        max_t = ds.num_targets(case_idx)
        t_try = (
            args.only_time
            if args.only_time is not None
            else 0
        )
        if t_try < 0 or t_try >= max_t:
            raise IndexError(
                f"--only-time={t_try} out of range for case_idx={case_idx} (num_targets={max_t})"
            )

        uid_dbg = str(ds.uids[case_idx]["id"])
        ref_sub_dbg = _cfg_ref_imgs_subdir(cfg, DYNAMIC_BENCHMARK_REF_SUBDIR)
        rd = os.path.abspath(os.path.expanduser(root_dirs))
        sk = dict(
            ref_uniform_inputs=True,
            ref_inputs_dir=os.path.join(rd, uid_dbg, ref_sub_dbg),
        )

        _ = ds.sample(case_idx, t_try, **sk)
        patch_benchmark_item_ref_pad_resize(
            _,
            motion_scene_root=os.path.join(rd, uid_dbg),
            ref_subdir=ref_sub_dbg,
            aspect_standard=float(ds.aspect_standard),
        )
        print(
            "debug sample OK",
            f"case_idx={case_idx}",
            f"t={t_try}",
            "source_rgbs",
            _["source_rgbs"].shape,
        )
        return

    lhm = _build_model(cfg)
    lhmp_benchmark_animation_inference(
        lhm,
        ds,
        output_root,
        infer_cfg=cfg,
        motion_root_dirs=args.motion_root_dirs,
        split=args.split,
        gpus=args.gpus,
        ref_view_count=int(args.view),
        infer_batch_size=int(args.infer_batch_size),
        only_case_idx=args.only_case,
        only_time_idx=args.only_time,
        use_pred_shape=use_pred_shape,
        gs_output=gs_output,
    )


if __name__ == "__main__":
    main()
