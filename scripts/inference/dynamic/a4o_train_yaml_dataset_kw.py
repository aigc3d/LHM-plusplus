# -*- coding: utf-8 -*-
"""Build Human LRM / tar-style video dataset kwargs from a training YAML.

Reads ``dataset.subsets`` for preprocessing / flags. Subset **name** differs across configs
(``dna_a4o_tar``, ``video_human_a4o_tar``, …). Use ``subset_name=auto`` (constant ``AUTO_SUBSET_NAME``)
to try ``SUBSET_NAME_FALLBACKS`` in order.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from omegaconf import OmegaConf

# Explicit name (DNA-style yaml)
LEGACY_DEFAULT_SUBSET_NAME = "dna_a4o_tar"

# CLI default: try these in order until one exists under ``dataset.subsets``
AUTO_SUBSET_NAME = "auto"
SUBSET_NAME_FALLBACKS: Tuple[str, ...] = (
    "dna_lhmpp_folder",
    LEGACY_DEFAULT_SUBSET_NAME,
    "video_human_lhm_a4o_tar",
    "video_human_a4o_tar",
)

_DEFAULT_PIPELINE: List[Dict[str, Any]] = [
    dict(name="PadRatioWithScale", target_ratio=5 / 3, tgt_max_size_list=[840]),
    dict(name="ToTensor"),
]


def _default_render_region(render_res_high: int) -> Tuple[int, int]:
    h = int(round(render_res_high * 700 / 420))
    w = render_res_high
    return (h, w)


def _mark_pipeline_val(processing_pipeline: List[MutableMapping[str, Any]]) -> None:
    for step in processing_pipeline:
        step["val"] = True


def pick_dataset_subset(
    full_subsets: Sequence[Any],
    subset_name: str = AUTO_SUBSET_NAME,
) -> MutableMapping[str, Any]:
    """Return the first matching subset: exact ``subset_name``, or each of ``SUBSET_NAME_FALLBACKS`` when ``subset_name=='auto'``."""

    if subset_name == AUTO_SUBSET_NAME:
        candidates: List[str] = list(SUBSET_NAME_FALLBACKS)
    else:
        candidates = [subset_name]

    names_seen: List[str] = []
    for want in candidates:
        for sub in full_subsets:
            name = sub.get("name") if hasattr(sub, "get") else sub["name"]
            if name not in names_seen:
                names_seen.append(str(name))
            if name != want:
                continue
            if OmegaConf.is_config(sub):
                return OmegaConf.to_container(sub, resolve=True)  # type: ignore[return-value]
            if isinstance(sub, dict):
                return dict(sub)
            raise TypeError(f"unexpected dataset.subsets entry type: {type(sub)}")

    tried = ", ".join(f'"{c}"' for c in candidates)
    raise ValueError(
        f"dataset.subsets has no entry for subset_name={subset_name!r} (tried {tried}). "
        f"Available names: {names_seen}. Pass --dataset-subset <name> explicitly."
    )


def build_a4o_tar_dataset_constructor_kwargs(
    cfg: Any, subset: Mapping[str, Any], ref_img_size: int
) -> Dict[str, Any]:
    """Constructor kwargs for :class:`VideoHumanLHMA4ODataset` descendants (eval / val)."""

    render_low = int(cfg.dataset.render_image.low)
    render_high = int(cfg.dataset.render_image.high)
    region = OmegaConf.select(cfg, "dataset.render_image.region", default=None)
    if region is None:
        render_region_size = _default_render_region(render_high)
    else:
        render_region_size = OmegaConf.to_container(region, resolve=True)

    sample_side_views = int(cfg.dataset.sample_side_views)
    source_image_res = int(cfg.dataset.source_image_res)
    multiply = int(OmegaConf.select(cfg, "dataset.multiply", default=14) or 14)

    processing_pipeline = subset.get("processing_pipeline")
    if not processing_pipeline:
        processing_pipeline = [dict(s) for s in _DEFAULT_PIPELINE]
    else:
        processing_pipeline = [dict(s) for s in processing_pipeline]
    _mark_pipeline_val(processing_pipeline)

    use_flame = bool(subset.get("use_flame", True))
    womask = bool(subset.get("womask", True))
    src_head_size = int(subset.get("src_head_size", 112))
    random_sample_ratio = float(subset.get("random_sample_ratio", 0.9))
    enlarge = subset.get("enlarge_ratio")
    if enlarge is not None:
        enlarge_ratio = list(enlarge)
    else:
        enlarge_ratio = [0.9, 1.1]

    _mv_en = subset.get("mv_val_mask_bbox_enlarge", 1.0)
    mv_val_mask_bbox_enlarge = float(_mv_en if _mv_en is not None else 1.0)

    raw_bg = subset.get("mv_val_bg_color", 1.0)
    mv_val_bg_color = float(raw_bg if raw_bg is not None else 1.0)
    mv_val_bg_color = max(0.0, min(1.0, mv_val_bg_color))

    return dict(
        sample_side_views=sample_side_views,
        render_image_res_low=render_low,
        render_image_res_high=render_high,
        render_region_size=render_region_size,
        source_image_res=source_image_res,
        repeat_num=1,
        enlarge_ratio=enlarge_ratio,
        debug=False,
        is_val=True,
        use_flame=use_flame,
        src_head_size=src_head_size,
        ref_img_size=ref_img_size,
        womask=womask,
        random_sample_ratio=random_sample_ratio,
        multiply=multiply,
        processing_pipeline=processing_pipeline,
        mv_val_mask_bbox_enlarge=mv_val_mask_bbox_enlarge,
        mv_val_bg_color=mv_val_bg_color,
    )
