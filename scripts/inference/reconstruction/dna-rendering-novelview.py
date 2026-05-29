#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DNA-Rendering novel-view benchmark: per-frame canonical + held-out camera renders.

**Not runnable yet** — exits with a stay-tuned message until release (see
``RECONSTRUCTION_BENCHMARK_ENABLED`` in ``core/utils/model_card.py``).

Requires untarred data under ``evaluation/reconstruction_benchmark/`` (see
``scripts/download_evaluation/download_reconstruction_benchmark.py``).

Example::

    python scripts/inference/reconstruction/dna-rendering-novelview.py --debug
    python scripts/inference/reconstruction/dna-rendering-novelview.py \\
        --model-name LHMPP-700M-PS-DNA \\
        --root-dirs ./evaluation/reconstruction_benchmark \\
        --t-start 0 --t-end 150
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import traceback

import torch
from accelerate import Accelerator

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import types  # noqa: E402

for _pkg, _rel in (
    ("core", "core"),
    ("core.datasets", "core/datasets"),
    ("core.datasets.evaluation", "core/datasets/evaluation"),
):
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_REPO_ROOT, _rel)]  # type: ignore[attr-defined]
        sys.modules[_pkg] = _mod

torch._dynamo.config.disable = True

from core.datasets.evaluation.reconstruction_benchmark import (  # noqa: E402
    BENCHMARK_USE_DATA_SHAPE,
    benchmark_case_time_length,
    build_lhm_from_model_card,
    build_multiview_motion_pack,
    build_reconstruction_benchmark_dataset,
    add_dna_benchmark_args,
    configure_logging,
    default_output_root,
    gs_export_enabled,
    run_one_benchmark_item,
)

_LOG = logging.getLogger(__name__)
_BENCHMARK_MODE = "novel_view"


def get_parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DNA LHMPP novel-view benchmark inference")
    add_dna_benchmark_args(p)
    return p.parse_args()


@torch.no_grad()
def run_novel_view_inference(
    lhm: torch.nn.Module,
    dataset,
    output_root: str,
    infer_cfg,
    cfg_train,
    *,
    split: int,
    gpus: int,
    t_start: int,
    t_end: int,
    infer_batch_size: int,
) -> None:
    if lhm is not None:
        lhm.cuda()
        lhm.eval()

    os.makedirs(output_root, exist_ok=True)
    n_cases = len(dataset)
    bins = int(math.ceil(n_cases / split))

    for case_idx in range(bins * gpus, min(bins * (gpus + 1), n_cases)):
        scene_id = dataset._scene_ids[case_idx]
        ref0 = dataset._ref_camera_ids[0]
        num_t = benchmark_case_time_length(
            dataset.root_dirs,
            scene_id,
            ref0,
            fallback_num_frames=dataset.num_frames,
        )
        ta = max(0, int(t_start))
        tb = min(int(t_end), num_t)
        print(f"case={case_idx} scene={scene_id} frames={num_t} t=[{ta}, {tb})")

        for t in range(ta, tb):
            try:
                item = dataset.sample(case_idx, t)
            except Exception as ex:
                print(f"skip case={case_idx} t={t}: {ex}", file=sys.stderr)
                traceback.print_exc()
                continue

            frame_id = int(item["tgt_frame_id"])
            try:
                motion_pack = build_multiview_motion_pack(
                    item["target_cam_dirs"], frame_id, infer_cfg
                )
                run_one_benchmark_item(
                    lhm,
                    item,
                    motion_pack,
                    output_root,
                    infer_cfg,
                    cfg_train,
                    forward_cache=None,
                    batch_size=infer_batch_size,
                )
            except Exception as ex:
                print(
                    f"inference failed scene={scene_id} t={t} frame={frame_id}: {ex}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                _LOG.exception("novel-view failed scene=%s t=%s", scene_id, t)


def main() -> None:
    from core.utils.model_card import (
        ensure_reconstruction_benchmark_enabled,
        require_reconstruction_model,
    )

    ensure_reconstruction_benchmark_enabled()

    args = get_parse()
    args.model_name = require_reconstruction_model(args.model_name)
    configure_logging(args.log_level)

    Accelerator()

    if args.debug:
        from omegaconf import OmegaConf
        from core.utils.model_card import RECONSTRUCTION_MODEL_CONFIG

        rel = RECONSTRUCTION_MODEL_CONFIG[args.model_name]
        train_yaml = rel if os.path.isabs(rel) else os.path.join(_REPO_ROOT, rel)
        cfg_train = OmegaConf.load(train_yaml)
        OmegaConf.resolve(cfg_train)
        infer_cfg = OmegaConf.create({"render_size": 420, "logger": "INFO"})
        lhm = None
    else:
        lhm, infer_cfg, cfg_train = build_lhm_from_model_card(
            args.model_name, model_path=args.model_path
        )

    dataset = build_reconstruction_benchmark_dataset(
        cfg_train,
        _BENCHMARK_MODE,
        args.view,
        root_dirs=args.root_dirs,
        meta_path=args.meta_path,
        dataset_subset=args.dataset_subset,
    )

    if args.view != len(dataset._ref_camera_ids):
        raise ValueError(
            f"--view {args.view} must match ref camera count "
            f"{len(dataset._ref_camera_ids)}"
        )

    output_root = default_output_root(_BENCHMARK_MODE, args.model_name, args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    _LOG.info(
        "Export root: %s gs_export=%s shape_mode=%s",
        output_root,
        gs_export_enabled(cfg_train),
        "datashape" if BENCHMARK_USE_DATA_SHAPE else "predshape",
    )

    if args.debug:
        print(f"dataset len={len(dataset)} mode={_BENCHMARK_MODE}")
        try:
            item = dataset.sample(0, 0)
            n_test = int(item["test_camera_ids"].numel())
            print(
                f"sample(0,0) OK uid={item['uid']} tgt_frame={item['tgt_frame_id']} "
                f"test_cams={n_test}"
            )
        except Exception as exc:
            print(f"smoke failed: {exc}")
            traceback.print_exc()
        return

    run_novel_view_inference(
        lhm,
        dataset,
        output_root,
        infer_cfg,
        cfg_train,
        split=args.split,
        gpus=args.gpus,
        t_start=args.t_start,
        t_end=args.t_end,
        infer_batch_size=args.infer_batch_size,
    )


if __name__ == "__main__":
    main()
