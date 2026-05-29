# <span><img src="./assets/LHM++_logo.png" height="35" style="vertical-align: top;"> **LHM++** - Official PyTorch Implementation</span>

#####  <p align="center"> [Lingteng Qiu<sup>*</sup>](https://lingtengqiu.github.io/), [Peihao Li<sup>*</sup>](https://liphao99.github.io/), [Heyuan Li<sup>*</sup>](https://scholar.google.com/), [Qi Zuo](https://scholar.google.com/citations?user=UDnHe2IAAAAJ&hl=zh-CN), [Xiaodong Gu](https://scholar.google.com.hk/citations?user=aJPO514AAAAJ&hl=zh-CN&oi=ao), [Yuan Dong](https://scholar.google.com/), [Weihao Yuan](https://weihao-yuan.com/), [Rui Peng](https://scholar.google.com/), [Siyu Zhu](https://scholar.google.com/), [Xiaoguang Han](https://scholar.google.com/), [Guanying Chen<sup>✉</sup>](https://guanyingc.github.io/), [Zilong Dong<sup>✉</sup>](https://baike.baidu.com/item/%E8%91%A3%E5%AD%90%E9%BE%99/62931048)</p>
#####  <p align="center"> Tongyi Lab, Alibaba Group · SYSU · CUHK-SZ · Fudan University </p>

[![Project Website](https://img.shields.io/badge/🌐-Project_Website-blueviolet)](https://lingtengqiu.github.io/LHM++/)
[![arXiv Paper](https://img.shields.io/badge/📜-arXiv:2506-13766v2)](https://arxiv.org/pdf/2506.13766v2)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/Lingteng/LHMPP)
[![YouTube](https://img.shields.io/badge/▶️-YouTube_Video-red)](https://www.youtube.com/watch?v=Nipf3jdSi34)
[![Apache License](https://img.shields.io/badge/📃-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)


<p align="center">
  <img src="./assets/LHM++_teaser.png" heihgt="100%">
</p>

**LHM++** is an efficient large-scale human reconstruction model that generates high-quality, animatable 3D avatars within seconds from one or multiple pose-free images. It achieves dramatic speedups over LHM-0.7B  via an Encoder-Decoder Point-Image Transformer architecture. See the [project website](https://lingtengqiu.github.io/LHM++/) for more details.

#### Model Specifications

| Type | Views | 3DGS-OUTPUT | Feat. Dim | Attn. Heads | # GS Points | Encoder Dim. | Service Requirement | Inference Time (1v) | Inference Time (4v) | Inference Time (8v) | Inference Time (16v) |
|------|-------|-------------|------------|-------------|-------------|--------------|---------------------|--------------------|--------------------|--------------------|---------------------|
| LHMPP-700M-PixelShuffle | Any | ✓ | 1024 | 16 | 160,000 | 1024 | 8 GB | 0.79 s | 1.00 s | 1.31 s | 2.13 s |
| LHMPP-700M-SMPLX-FREE | Any | ✓ | 1024 | 16 | 160,000 | 1024 | 8 GB | 0.79 s | 1.00 s | 1.31 s | 2.13 s |
| LHMPP-700M | Any | — | 1024 | 16 | 160,000 | 1024 | 8 GB | 0.79 s | 1.00 s | 1.31 s | 2.13 s |
| LHMPPS-700M | Any | — | 1024 | 16 | 160,000 | 1024 | 7.3 GB | 0.79 s | 1.00 s | 1.31 s | 2.13 s |

#### Efficiency Analysis

LHM++ achieves dramatic speedups via the Encoder-Decoder Point-Image Transformer architecture. Below we show the efficiency comparison across different configurations.

<p align="center">
  <img src="./assets/efficiency_analysis/efficiency_analysis.png" width="90%">
</p>

<p align="center">
  <img src="./assets/efficiency_analysis/comparison_efficiency.jpg" width="90%">
</p>

<p align="center">
  <img src="./assets/efficiency_analysis/efficiency_animation.gif" width="90%">
</p>

If you prefer Chinese documentation, please see the [Chinese README](./README_CN.md).
## 📢 Latest Updates

- **LHMPP-700M-PixelShuffle (default):** SMPLX-FREE variant with an **MLPPixelShuffle** neural renderer (lighter dense head than DPT). Supports **3DGS-PLY export** and **`gs_render`** output (see [`GS_RENDER_SUPPORTED_MODEL_NAMES`](./core/utils/model_card.py)). Hub weights pending; use local checkpoint + `--model_path` until published.
- **LHMPP-700M (updated release):** We released a new LHMPP-700M build that supports **standard 3D Gaussian Splatting PLY (`3GS-PLY`)** as an output format.

### New features

- **Export `gs.ply`:** Run [`scripts/inference/to_gs_ply.py`](./scripts/inference/to_gs_ply.py) to save **3D Gaussian Splatting** as a standard **`.ply`**—either **canonical T-pose** (leave `--pose_dir` empty) or **a single SMPL-X JSON frame** (`--pose_dir`). Supported for **`LHMPP-700M-PixelShuffle`** (default) and **`LHMPP-700M-SMPLX-FREE`** (see [`GS_RENDER_SUPPORTED_MODEL_NAMES`](./core/utils/model_card.py)). Full usage is in **Export Gaussian Splatting PLY (`to_gs_ply.py`)** under **Getting Started** below.
- **GS render results:** In [`app.py`](./app.py), use **`gs_render`** for **RGB output from Gaussian splatting only** (no neural refinement). Launch with **`--gs`** when the model is **`LHMPP-700M-PixelShuffle`** or **`LHMPP-700M-SMPLX-FREE`**, or switch **Output Renderer** to **gs_render** in the UI. See **Local Gradio Run** below.

### TODO List

- [x] Core Inference Pipeline🔥🔥🔥
- [x] Release the codes and pretrained weights
- [x] HuggingFace Demo Integration 🤗🤗🤗
- [x] **Benchmarks — dynamic reconstruction:** evaluation code and validation data for **NeuMan**, **SelfCapture**, **Vid2Avatar** (see [Dynamic benchmark evaluation](#dynamic-benchmark-evaluation))
- [ ] **Benchmarks — novel view/pose synthesis:** TODO (**THuman-2.1**, etc.); **DNA-Rendering** data download script (`--download` / `--untar`) is available — evaluation metrics coming later (see [Reconstruction benchmark download](#reconstruction-benchmark-download))
- [ ] ModelScope Space Online Demo
- [ ] Release Training data & Testing Data (License Available)
- [ ] Training Codes Release 

## 🚀 Getting Started

### Environment Setup
Clone the repository.
```bash
git clone https://github.com/aigc3d/LHM-plusplus
cd LHM-plusplus
```

```bash
# install torch 2.3.0 cuda 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# install dependencies
pip install -r requirements.txt
pip install rembg[cpu]  # only use during extracting sparse view inputs.

# install pointops
cd ./lib/pointops/ && python setup.py install && cd ../../

pip install spconv-cu121
# pip install torch_scatter, see [wheel](https://data.pyg.org/whl/) for your CUDA version
# For example (PyTorch 2.3 + CUDA 12.1 + Python 3.10):
pip install torch_scatter-2.1.2+pt23cu121-cp310-cp310-linux_x86_64.whl

# install pytorch3d
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html

# install diff-gaussian-rasterization
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/
# or
# git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
# pip install ./diff-gaussian-rasterization

# install simple-knn
pip install git+https://github.com/camenduru/simple-knn/


# install gsplat
# pip install gsplat from pre-compiled [wheel](https://docs.gsplat.studio/whl/gsplat/)
# For example (PyTorch 2.3 + CUDA 12.1 + Python 3.10):
# gsplat-1.4.0+pt23cu121-cp310-cp310-linux_x86_64.whl
pip install gsplat-1.4.0+pt23cu121-cp310-cp310-linux_x86_64.whl
```

The installation has been tested with python3.10, CUDA 12.1.
Or you can install dependencies step by step, following [INSTALL.md](INSTALL.md).

### Model Weights 

#### One-Click Download (recommended)

Download assets (motion_video), prior models, and pretrained weights in one command:

```bash
# One-click: motion_video + prior models + pretrained weights
python scripts/download_all.py

# Skip parts (e.g. already have motion_video)
python scripts/download_all.py --skip-asset --skip-models

# Force re-download motion_video
python scripts/download_all.py --force-asset
```

#### Pretrained Model Download (individual)

Use the download script to fetch prior models (human_model_files, voxel_grid, arcface, etc.) and LHM++ weights. Skips items that already exist. Tries HuggingFace first, falls back to ModelScope.

```bash
# Download prior models + pretrained weights (default)
python scripts/download_pretrained_models.py

# Prior models only (human_model_files, voxel_grid, BiRefNet, etc.)
python scripts/download_pretrained_models.py --prior

# LHM++ model weights only (LHMPP-700M-PixelShuffle default, LHMPP-700M-SMPLX-FREE, LHMPP-700M, LHMPP-700MC, LHMPPS-700M)
python scripts/download_pretrained_models.py --models

# Custom save directory
python scripts/download_pretrained_models.py --save-dir /path/to/pretrained_models
```

#### Download from ModelScope (manual)
```python
from modelscope import snapshot_download

# LHMPP-700M-PixelShuffle (default model weights; use --model_path until Hub is published)
model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPP-700M-PixelShuffle', cache_dir='./pretrained_models')
# Or: LHMPP-700M-SMPLX-FREE, LHMPP-700M, LHMPP-700MC, LHMPPS-700M
# model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPP-700M-SMPLX-FREE', cache_dir='./pretrained_models')
# model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPP-700M', cache_dir='./pretrained_models')
# model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPP-700MC', cache_dir='./pretrained_models')
# model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPPS-700M', cache_dir='./pretrained_models')

# LHMPP-Prior (prior models: human_model_files, voxel_grid, BiRefNet, etc.)
model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPP-Prior', cache_dir='./pretrained_models')
```

#### Motion Video Download

Required for Gradio motion examples. If `./motion_video` at project root is missing, downloads from [Damo_XR_Lab/LHMPP-Assets](https://www.modelscope.cn/models/Damo_XR_Lab/LHMPP-Assets) (model, extracts motion_video.tar to project root):

```bash
# Requires: pip install modelscope
python scripts/download_motion_video.py

# Custom parent directory (default: . = project root)
python scripts/download_motion_video.py --save-dir .
```

After downloading weights and data, the project structure:
```bash
├── app.py
├── assets
│   ├── efficiency_analysis
│   ├── example_aigc_images
│   ├── example_multi_images
│   ├── example_videos
│   └── dynamic_metrics_table.md
├── benchmark
├── configs
│   └── train
│       ├── LHMPP-1view.yaml
│       ├── LHMPP-any-view.yaml
│       ├── LHMPP-any-view-convhead.yaml
│       └── LHMPP-any-view-DPTS.yaml
├── core
│   ├── datasets
│   ├── losses
│   ├── models
│   ├── modules
│   ├── outputs
│   ├── runners
│   ├── structures
│   ├── utils
│   └── launch.py
├── dnnlib
├── engine
│   ├── BiRefNet
│   ├── pose_estimation
│   └── ouputs.py
├── exps
│   ├── checkpoints
│   ├── releases
│   └── ...
├── lib
│   └── pointops
├── pretrained_models
│   ├── dense_sample_points
│   ├── gagatracker
│   ├── human_model_files
│   ├── voxel_grid
│   ├── arcface_resnet18.pth
│   ├── BiRefNet-general-epoch_244.pth
│   ├── Damo_XR_Lab
│   └── huggingface
├── scripts
│   ├── exp
│   ├── inference
│   ├── mvs_render
│   ├── pose_estimator
│   ├── test
│   ├── convert_hf.py
│   ├── download_all.py
│   ├── download_motion_video.py
│   ├── download_pretrained_models.py
│   └── upload_hub.py
├── tools
│   └── metrics
├── train_data
│   ├── example_imgs
│   └── motion_video
├── motion_video
├── INSTALL.md
├── INSTALL_CN.md
├── README.md
├── README_CN.md
└── requirements.txt
```

### 💻 Local Gradio Run
Now, we support user motion sequence input. As the pose estimator requires some GPU memory, this Gradio application requires at least 8 GB of GPU memory to run LHMPP-700M with 8-view inputs.
```bash
## Quick Start; Testing the Code
python ./scripts/test/test_app_video.py --input_video ./assets/example_videos/yuliang.mp4
python ./scripts/test/test_app_case.py

# Run LHM++ with Gradio API
python ./app.py --model_name [LHMPP-700M-PixelShuffle, LHMPP-700M-SMPLX-FREE, LHMPP-700M, LHMPPS-700M], default LHMPP-700M-PixelShuffle
```

**Render output mode (`app.py` only):** You can pick the startup default for the Gradio **Output Renderer** with mutually exclusive flags:

| Flag | Meaning |
|------|--------|
| *(none)* | Start with **neural_render** (Gaussian splat rasterization + neural refinement decoder). |
| `--neural` | Same as above; explicitly request **neural_render** on launch. |
| `--gs` | Start with **gs_render** (Gaussian splat RGB only, no neural refiner). **Supported for `LHMPP-700M-PixelShuffle` and `LHMPP-700M-SMPLX-FREE`.** For any other `--model_name`, the app logs a warning and forces **neural_render**. |

Examples:

```bash
# Default pipeline (neural_render); PixelShuffle needs --model_path until Hub publish
python ./app.py --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle

# Explicit neural_render
python ./app.py --model_name LHMPP-700M-PixelShuffle --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle --neural

# Prefer gs_render at startup (default model)
python ./app.py --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle --gs

# LHMPP-700M-SMPLX-FREE
python ./app.py --model_name LHMPP-700M-SMPLX-FREE --gs
```

You can still change **Output Renderer** in the Gradio UI when **gs_render** is available for the loaded model (`LHMPP-700M-PixelShuffle` or `LHMPP-700M-SMPLX-FREE`).

### Export Gaussian Splatting PLY (`to_gs_ply.py`)

Export a **Gaussian Splatting** point cloud as a standard **PLY** for offline viewers or downstream tooling. Only checkpoints listed in **`GS_RENDER_SUPPORTED_MODEL_NAMES`** in [`core/utils/model_card.py`](./core/utils/model_card.py) have the required GS heads (**`LHMPP-700M-PixelShuffle`** default, **`LHMPP-700M-SMPLX-FREE`**). The script exits with an error if you pick any other `--model_name`.

Run from the repo root (`LHM-plusplus`), after [environment setup](#environment-setup) and downloading **prior models + weights**.

| Mode | Trigger | What you get |
|------|---------|--------------|
| **T-pose (canonical)** | Omit **`--pose_dir`** (empty / default) | Gaussians in **canonical T-pose** SMPL-X space, with a **synthetic** single-frame camera when no motion file is provided. |
| **Any-pose (given SMPL-X frame)** | Set **`--pose_dir`** to **one** SMPL-X JSON | Gaussians **warped to that frame’s body pose** (and that JSON’s camera intrinsics are used in the forward pass). Not the full video / mask pipeline—only that file is read. |

**Pipeline (current implementation):** Both modes run `infer_single_view` on your reference images. **T-pose** then builds canonical SMPL-X angles and calls **`model.inference_gs`** → `save_ply`. **Any-pose** builds SMPL-X from the JSON and saves the **first posed view** from **`model.renderer.animate_gs_model`** (same Gaussian warp as **`forward_animate_gs`** in the renderer). If that path returns no posed models, the script **falls back** to **`model.inference_gs`**.

#### 1) T-pose (canonical GS, no pose JSON)

Leave **`--pose_dir` empty** (default). The script uses a synthetic single-frame camera and exports **canonical T-pose** Gaussians via **`inference_gs`**, as described above.

**Default output:** `<repo>/outputs/tpose_output/{ref_images_parent}.ply`  
(e.g. with `./assets/example_multi_images/*.png` → `.../tpose_output/example_multi_images.ply`)

```bash
cd LHM-plusplus

python scripts/inference/to_gs_ply.py \
  --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle \
  --image_glob "./assets/example_multi_images/00000_yuliang_*.png"
```

#### 2) Any-pose (one SMPL-X JSON file)

Pass **`--pose_dir`** to a **single** SMPL-X parameter JSON (e.g. `motion_video/BasketBall_I/smplx_params/00014.json`). The script reads **only that file** for camera intrinsics + body pose (no video pipeline, no segmentation masks). The exported PLY is the **posed** Gaussian cloud from **`animate_gs_model`** (not the internal canonical-template-only branch). Optional FLAME sidecar: `../flame_params/<same_basename>.json`.

**Default output:** `<repo>/outputs/animation_output/{sequence_name}/{ref_images_parent}_{json_stem}.ply`  
(e.g. `.../animation_output/BasketBall_I/example_multi_images_00014.ply`)

```bash
cd LHM-plusplus

python scripts/inference/to_gs_ply.py \
  --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle \
  --pose_dir "./motion_video/BasketBall_I/smplx_params/00014.json" \
  --image_glob "./assets/example_multi_images/00000_yuliang_*.png"
```

**Optional:** `--output /path/to/out.ply` overrides the defaults above; `--model_path /path/to/local-weights` uses local weights while keeping `--model_name` for the YAML config (recommended for **`LHMPP-700M-PixelShuffle`** until the Hub repo is published).

Useful flags: `--images_dir`, `--ref_view`, `--device`, `--work_dir`. Shape-from-image **betas** follow the app when `use_smplx_shape_estimator` is enabled in config (same as Gradio).

**Running Tips:** Ensure the input images are high resolution, preferably with visible hand details, and include at least one image where the body is fully extended/spread out.

### Dynamic benchmark evaluation

When comparing against our method, if you do **not rigorously align outputs with the validation set on motion**, reported metrics will be **substantially lower**. For a **fair comparison**, we release our **processed validation splits** so you can **directly reproduce our numbers**.

**Data:** scenes under `evaluation/dynamic_benchmark/` (**NeuMan**, **SelfCapture**, **Vid2Avatar**), with timelines in `benchmark/manifests/eval_lhmpp_*.json`. Each scene has masked timeline frames plus **16** reference PNGs in `ref_imgs_png/`; evaluation uses **8** uniformly sampled reference views.

Download from ModelScope [`Damo_XR_Lab/LHMPP-Evaluation-Benchmark`](https://modelscope.cn/models/Damo_XR_Lab/LHMPP-Evaluation-Benchmark):

```bash
python scripts/download_evaluation/download_dynamic_benchmarks.py
```

**Reported numbers:** [assets/dynamic_metrics_table.md](./assets/dynamic_metrics_table.md)

Run masked animation inference with [`infer_eval_animation.py`](./scripts/inference/dynamic/infer_eval_animation.py). Choose the benchmark via **`--dataset`**; choose the render branch by omitting or adding **`--gs-output`** (Neural vs GS export folders).

**Neural renderer** (default — writes `neural_rgb/` / `neural_mask/`):

```bash
cd LHM-plusplus

python scripts/inference/dynamic/infer_eval_animation.py --dataset neuman
python scripts/inference/dynamic/infer_eval_animation.py --dataset selfcapture
python scripts/inference/dynamic/infer_eval_animation.py --dataset vid2avatar
```

**GS raster** (writes `gs_rgb/` / `gs_mask/`):

```bash
python scripts/inference/dynamic/infer_eval_animation.py --dataset neuman --gs-output
python scripts/inference/dynamic/infer_eval_animation.py --dataset selfcapture --gs-output
python scripts/inference/dynamic/infer_eval_animation.py --dataset vid2avatar --gs-output
```

Exports go to `./exps/benchmarks/dynamic/{dataset}-1036x616-datashape-{neural|gs}/{scene_id}/` (default model **`LHMPP-700M-PixelShuffle`**, **8** reference views, **`--src-height 1036`**, datashape).

**Metrics** (PSNR / SSIM / LPIPS) after inference:

```bash
python tools/metrics/compute_dynamic_metrics.py --root ./exps/benchmarks/dynamic --dataset neuman
python tools/metrics/compute_dynamic_metrics.py --root ./exps/benchmarks/dynamic --dataset selfcapture
python tools/metrics/compute_dynamic_metrics.py --root ./exps/benchmarks/dynamic --dataset vid2avatar
```

Each run writes `dynamic_metrics_<dataset>.meta.json` under `./exps/benchmarks/dynamic/`.

### Reconstruction benchmark download

**DNA-Rendering** test benchmark for novel view / pose synthesis evaluation: **16** scenes, about **100 GB** of per-camera tar archives. Because of the size, download and extraction are **two separate steps** (download first, check disk space, then untar).

Data lands under `evaluation/reconstruction_benchmark/` (already in `.gitignore`). ModelScope: [`Damo_XR_Lab/LHMPP-DNA-Benchmark`](https://www.modelscope.cn/models/Damo_XR_Lab/LHMPP-DNA-Benchmark).

```bash
# Step 1: download all scene tars from ModelScope (~100 GB)
python scripts/download_evaluation/download_reconstruction_benchmark.py --download

# Step 2: extract all tars (needs additional disk space; tar files are kept)
python scripts/download_evaluation/download_reconstruction_benchmark.py --untar

# Optional: one scene only — download or untar all {scene_id}-Cam*.tar
python scripts/download_evaluation/download_reconstruction_benchmark.py --download --scene_id 0012_09
python scripts/download_evaluation/download_reconstruction_benchmark.py --untar --scene_id 0012_09
```

`--scene_id` filters by scene prefix (e.g. `0012_09` → `0012_09-Cam00.tar`, `0012_09-Cam01.tar`, …). `--untar` writes paths from inside each archive (e.g. `0012_09-Cam00/`) under `reconstruction_benchmark/` without an extra wrapper folder. Re-running without `--force` skips existing files. Evaluation inference and metrics for this benchmark are not included yet.


## More Works
Welcome to follow our team other interesting works:
- [LHM](https://github.com/aigc3d/LHM)
- [AniGS](https://github.com/aigc3d/AniGS)

## ✨ Star History

[![Star History](https://api.star-history.com/svg?repos=aigc3d/LHM-plusplus)](https://star-history.com/#aigc3d/LHM-plusplus&Date)

## Citation 

If you find our approach helpful, please consider citing our works.

**LHM++** (Efficient Large Human Reconstruction Model for Pose-free Images to 3D):

```
@article{qiu2025lhmpp,
  title={LHM++: An Efficient Large Human Reconstruction Model for Pose-free Images to 3D},
  author={Lingteng Qiu and Peihao Li and Heyuan Li and Qi Zuo and Xiaodong Gu and Yuan Dong and Weihao Yuan and Rui Peng and Siyu Zhu and Xiaoguang Han and Guanying Chen and Zilong Dong},
  journal={arXiv preprint arXiv:2503.10625},
  year={2025}
}
```

**LHM**:
```
@inproceedings{qiu2025LHM,
  title={LHM: Large Animatable Human Reconstruction Model from a Single Image in Seconds},
  author={Lingteng Qiu and Xiaodong Gu and Peihao Li and Qi Zuo and Weichao Shen and Junfei Zhang and Kejie Qiu and Weihao Yuan and Guanying Chen and Zilong Dong and Liefeng Bo},
  booktitle={ICCV},
  year={2025}
}
```
