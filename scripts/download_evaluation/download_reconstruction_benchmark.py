#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download LHMPP DNA reconstruction evaluation benchmark from ModelScope.

ModelScope repo: ``Damo_XR_Lab/LHMPP-DNA-Benchmark``

Two explicit steps (large download; run separately unless both flags are given)::

    # Step 1: download per-camera tar archives (~100 GB)
    python scripts/download_evaluation/download_reconstruction_benchmark.py --download

    # Step 2: extract into reconstruction_benchmark/ (needs extra disk space)
    python scripts/download_evaluation/download_reconstruction_benchmark.py --untar

    # Optional: one scene only (all {scene_id}-Cam*.tar)
    python scripts/download_evaluation/download_reconstruction_benchmark.py --download --scene_id 0012_09
    python scripts/download_evaluation/download_reconstruction_benchmark.py --untar --scene_id 0012_09

Default layout after download::

    {output_dir}/reconstruction_benchmark/
        0012_09-Cam00.tar
        0012_09-Cam01.tar
        ...

After untar, each archive expands under ``reconstruction_benchmark/`` using paths
inside the tar (no extra wrapper directory named after the tar)::

    {output_dir}/reconstruction_benchmark/
        0012_09-Cam00.tar
        0012_09-Cam00/          # from tar contents
            ...
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tarfile
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MODELSCOPE_MODEL_ID = "Damo_XR_Lab/LHMPP-DNA-Benchmark"
BENCHMARK_DIRNAME = "reconstruction_benchmark"
DEFAULT_OUTPUT_DIR = os.path.join(".", "evaluation")
MIN_SCENE_COUNT = 16
CAMERA_TAR_RE = re.compile(r"^\d{4}_\d{2}-Cam\d{2}\.tar$")
NESTED_BENCHMARK_DIR = "DNA-Rendering-Test-Benchmark"


def _abs_under_repo(path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(ROOT, path))


def benchmark_root(output_dir: str) -> str:
    """``{output_dir}/reconstruction_benchmark``."""
    return os.path.normpath(os.path.join(_abs_under_repo(output_dir), BENCHMARK_DIRNAME))


def _scene_from_tar_basename(name: str) -> str:
    return name.split("-Cam", 1)[0]


def _matches_camera_tar(name: str) -> bool:
    return bool(CAMERA_TAR_RE.match(name))


def _iter_camera_tar_paths(search_root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(search_root):
        for name in filenames:
            if _matches_camera_tar(name):
                yield os.path.join(dirpath, name)


def _matches_scene_id(name: str, scene_id: str) -> bool:
    return _scene_from_tar_basename(name) == scene_id


def _collect_camera_tars(
    search_root: str,
    *,
    scene_id: Optional[str] = None,
) -> List[str]:
    """Collect ``*-Cam*.tar`` paths; newest path wins per basename."""
    by_name: Dict[str, str] = {}
    for path in _iter_camera_tar_paths(search_root):
        name = os.path.basename(path)
        if scene_id is not None and not _matches_scene_id(name, scene_id):
            continue
        by_name[name] = path
    return sorted(by_name.values())


def _list_local_tars(
    output_dir: str,
    *,
    scene_id: Optional[str] = None,
) -> List[str]:
    root = benchmark_root(output_dir)
    if not os.path.isdir(root):
        return []
    paths: List[str] = []
    for name in sorted(os.listdir(root)):
        if not _matches_camera_tar(name):
            continue
        if scene_id is not None and not _matches_scene_id(name, scene_id):
            continue
        paths.append(os.path.join(root, name))
    return paths


def _scene_counts(tar_paths: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for path in tar_paths:
        counts[_scene_from_tar_basename(os.path.basename(path))] += 1
    return dict(counts)


def has_downloaded_tars(output_dir: str, *, min_scenes: int = MIN_SCENE_COUNT) -> bool:
    """Return True if at least ``min_scenes`` scene prefixes each have >=1 tar."""
    counts = _scene_counts(_list_local_tars(output_dir))
    return len(counts) >= min_scenes and all(count >= 1 for count in counts.values())


def _extracted_dir_for_tar(root: str, tar_path: str) -> str:
    """Directory expected after extracting one camera tar (basename without ``.tar``)."""
    return os.path.join(root, os.path.basename(tar_path)[:-4])


def has_untared_data(output_dir: str, *, scene_id: Optional[str] = None) -> int:
    """Count extracted camera directories for local tars."""
    root = benchmark_root(output_dir)
    if not os.path.isdir(root):
        return 0
    extracted = 0
    for tar_path in _list_local_tars(output_dir, scene_id=scene_id):
        if _dir_nonempty(_extracted_dir_for_tar(root, tar_path)):
            extracted += 1
    return extracted


def _no_tars_error(output_dir: str, scene_id: Optional[str], *, download_hint: bool) -> FileNotFoundError:
    root = benchmark_root(output_dir)
    if scene_id is not None:
        msg = f"No camera tar archives for scene_id {scene_id!r} under {root}."
    else:
        msg = f"No camera tar archives found under {root}."
    if download_hint:
        msg += " Run with --download first."
    return FileNotFoundError(msg)


def _dir_nonempty(path: str) -> bool:
    return os.path.isdir(path) and bool(os.listdir(path))


def download_from_modelscope(cache_dir: str) -> str:
    try:
        from modelscope import snapshot_download
    except ImportError as ex:
        raise ImportError(
            "modelscope is required. Install with: pip install modelscope"
        ) from ex

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading {MODELSCOPE_MODEL_ID} (ModelScope) -> cache {cache_dir} ...")
    local_dir = snapshot_download(MODELSCOPE_MODEL_ID, cache_dir=cache_dir)
    print(f"ModelScope snapshot: {local_dir}")
    return local_dir


def _materialize_tars(
    tar_paths: Sequence[str],
    output_dir: str,
    *,
    force: bool = False,
) -> Tuple[int, int]:
    """Copy tars flat into ``reconstruction_benchmark/``; return (copied, skipped)."""
    dest_root = benchmark_root(output_dir)
    os.makedirs(dest_root, exist_ok=True)
    copied = 0
    skipped = 0
    for src in tar_paths:
        name = os.path.basename(src)
        dest = os.path.join(dest_root, name)
        if os.path.isfile(dest) and not force:
            if os.path.getsize(dest) == os.path.getsize(src):
                skipped += 1
                continue
        print(f"Copying {name} -> {dest_root}/")
        shutil.copy2(src, dest)
        copied += 1
    return copied, skipped


def download_reconstruction_benchmark(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    force: bool = False,
    cache_dir: Optional[str] = None,
    scene_id: Optional[str] = None,
) -> str:
    """Download ModelScope snapshot and flatten camera tars; return benchmark root."""
    if has_downloaded_tars(output_dir) and not force and scene_id is None:
        target = benchmark_root(output_dir)
        print(f"reconstruction_benchmark tars already present at {target}, skip download.")
        return target

    cache = cache_dir or os.path.join(ROOT, ".cache", "lhmpp_dna_benchmark")
    downloaded = download_from_modelscope(cache)

    search_roots = [downloaded]
    nested = os.path.join(downloaded, NESTED_BENCHMARK_DIR)
    if os.path.isdir(nested):
        search_roots.append(nested)

    tar_paths: List[str] = []
    seen: Set[str] = set()
    for search_root in search_roots:
        for path in _collect_camera_tars(search_root, scene_id=scene_id):
            name = os.path.basename(path)
            if name in seen:
                continue
            seen.add(name)
            tar_paths.append(path)

    if not tar_paths:
        if scene_id is not None:
            raise FileNotFoundError(
                f"No camera tar archives for scene_id {scene_id!r} under {downloaded}"
            )
        raise FileNotFoundError(
            f"No camera tar archives matching *-Cam*.tar under {downloaded}"
        )

    if scene_id is not None:
        print(f"Materializing scene_id {scene_id!r}: {len(tar_paths)} tar(s)")

    copied, skipped = _materialize_tars(tar_paths, output_dir, force=force)
    local_tars = _list_local_tars(output_dir, scene_id=scene_id)
    scene_map = _scene_counts(local_tars)
    print(
        f"Download materialized: copied={copied}, skipped={skipped}, "
        f"scenes={len(scene_map)}, tars={len(local_tars)}"
    )
    if scene_id is None and len(scene_map) < MIN_SCENE_COUNT:
        print(
            f"Warning: expected at least {MIN_SCENE_COUNT} scenes, "
            f"found {len(scene_map)}.",
            file=sys.stderr,
        )
    return benchmark_root(output_dir)


def untar_reconstruction_benchmark(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    force: bool = False,
    scene_id: Optional[str] = None,
) -> str:
    """Extract each local camera tar into ``reconstruction_benchmark/``."""
    tar_paths = _list_local_tars(output_dir, scene_id=scene_id)
    if not tar_paths:
        raise _no_tars_error(output_dir, scene_id, download_hint=True)

    scene_map = _scene_counts(tar_paths)
    if scene_id is None and len(scene_map) < MIN_SCENE_COUNT:
        raise FileNotFoundError(
            f"Found only {len(scene_map)} scene(s) under "
            f"{benchmark_root(output_dir)} (expected >= {MIN_SCENE_COUNT}). "
            "Run with --download first."
        )

    root = benchmark_root(output_dir)
    extracted = 0
    skipped = 0
    if scene_id is not None:
        print(f"Untarring scene_id {scene_id!r}: {len(tar_paths)} tar(s)")
    for tar_path in tar_paths:
        name = os.path.basename(tar_path)
        extracted_dir = _extracted_dir_for_tar(root, tar_path)
        if _dir_nonempty(extracted_dir) and not force:
            skipped += 1
            continue
        if os.path.isdir(extracted_dir) and force:
            shutil.rmtree(extracted_dir)
        print(f"Extracting {name} -> {root}/")
        with tarfile.open(tar_path, "r:") as tf:
            tf.extractall(root)
        extracted += 1

    print(
        f"Untar complete: extracted={extracted}, skipped={skipped}, "
        f"extracted_dirs={has_untared_data(output_dir, scene_id=scene_id)}"
    )
    return root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download ModelScope snapshot and flatten *-Cam*.tar into reconstruction_benchmark/.",
    )
    parser.add_argument(
        "--untar",
        action="store_true",
        help="Extract each *-Cam*.tar into reconstruction_benchmark/ (paths from archive).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Parent directory for benchmark data (default: ./evaluation). "
            "Creates {output-dir}/reconstruction_benchmark/."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="ModelScope download cache (default: .cache/lhmpp_dna_benchmark under repo root).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-copy tars on download, or overwrite extracted dirs on untar.",
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        metavar="ID",
        help=(
            "Optional scene prefix filter, e.g. 0012_09. "
            "Only processes {scene_id}-Cam*.tar (works with --download and --untar)."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.download and not args.untar:
        build_arg_parser().print_help(file=sys.stderr)
        print(
            "\nError: specify --download and/or --untar.",
            file=sys.stderr,
        )
        return 2

    try:
        if args.download:
            download_reconstruction_benchmark(
                args.output_dir,
                force=args.force,
                cache_dir=args.cache_dir,
                scene_id=args.scene_id,
            )
        if args.untar:
            target = untar_reconstruction_benchmark(
                args.output_dir,
                force=args.force,
                scene_id=args.scene_id,
            )
            print(f"Reconstruction benchmark data at: {target}")
        elif args.download:
            print(f"Reconstruction benchmark tars at: {benchmark_root(args.output_dir)}")
    except Exception as ex:
        print(f"Failed: {ex}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
