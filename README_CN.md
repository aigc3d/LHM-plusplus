# <span><img src="./assets/LHM++_logo.png" height="35" style="vertical-align: top;"> **LHM++** - 官方 PyTorch 实现</span>

#####  <p align="center"> [Lingteng Qiu<sup>*</sup>](https://lingtengqiu.github.io/), [Peihao Li<sup>*</sup>](https://liphao99.github.io/), [Heyuan Li<sup>*</sup>](https://scholar.google.com/), [Qi Zuo](https://scholar.google.com/citations?user=UDnHe2IAAAAJ&hl=zh-CN), [Xiaodong Gu](https://scholar.google.com.hk/citations?user=aJPO514AAAAJ&hl=zh-CN&oi=ao), [Yuan Dong](https://scholar.google.com/), [Weihao Yuan](https://weihao-yuan.com/), [Rui Peng](https://scholar.google.com/), [Siyu Zhu](https://scholar.google.com/), [Xiaoguang Han](https://scholar.google.com/), [Guanying Chen<sup>✉</sup>](https://guanyingc.github.io/), [Zilong Dong<sup>✉</sup>](https://baike.baidu.com/item/%E8%91%A3%E5%AD%90%E9%BE%99/62931048)</p>
#####  <p align="center"> 通义实验室 · 阿里巴巴集团 · 中山大学 · 港中大（深圳）· 复旦大学 </p>

[![Project Website](https://img.shields.io/badge/🌐-项目主页-blueviolet)](https://lingtengqiu.github.io/LHM++/)
[![arXiv Paper](https://img.shields.io/badge/📜-arXiv:2506-13766v2)](https://arxiv.org/pdf/2506.13766v2)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/Lingteng/LHMPP)
[![YouTube](https://img.shields.io/badge/▶️-YouTube_视频-red)](https://www.youtube.com/watch?v=Nipf3jdSi34)
[![Apache License](https://img.shields.io/badge/📃-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)


<p align="center">
  <img src="./assets/LHM++_teaser.png" heihgt="100%">
</p>

**LHM++** 是一款高效的大规模人体重建模型，能够从一张或多张无姿态约束的图像在数秒内生成高质量、可驱动的 3D 虚拟人。通过 Encoder-Decoder Point-Image Transformer 架构，相比 LHM-0.7B 实现了显著加速。更多详情请参见[项目主页](https://lingtengqiu.github.io/LHM++/)。

#### 模型规格

| 类型 | 视角 | 3DGS-OUTPUT | 特征维度 | 注意力头数 | GS 点数 | 编码器维度 | 显存需求 | 推理时间 (1v) | 推理时间 (4v) | 推理时间 (8v) | 推理时间 (16v) |
|------|------|-------------|----------|------------|---------|------------|----------|---------------|---------------|---------------|----------------|
| LHMPP-700M-PixelShuffle | Any | ✓ | 1024 | 16 | 160,000 | 1024 | 8 GB | 0.79 s | 1.00 s | 1.31 s | 2.13 s |
| LHMPP-700M-SMPLX-FREE | Any | ✓ | 1024 | 16 | 160,000 | 1024 | 8 GB | 0.79 s | 1.00 s | 1.31 s | 2.13 s |
| LHMPP-700M | Any | — | 1024 | 16 | 160,000 | 1024 | 8 GB | 0.79 s | 1.00 s | 1.31 s | 2.13 s |
| LHMPPS-700M | Any | — | 1024 | 16 | 160,000 | 1024 | 7.3 GB | 0.79 s | 1.00 s | 1.31 s | 2.13 s |

#### 效率分析

LHM++ 通过 Encoder-Decoder Point-Image Transformer 架构实现了显著加速。以下展示不同配置下的效率对比。

<p align="center">
  <img src="./assets/efficiency_analysis/efficiency_analysis.png" width="90%">
</p>

<p align="center">
  <img src="./assets/efficiency_analysis/comparison_efficiency.jpg" width="90%">
</p>

<p align="center">
  <img src="./assets/efficiency_analysis/efficiency_animation.gif" width="90%">
</p>

For English readers, see [README in English](./README.md).

## 📢 最新动态

- **LHMPP-700M-PixelShuffle（默认）：** SMPLX-FREE 变体，神经渲染头为 **MLPPixelShuffle**（比 DPT 更轻量的 dense head）。支持 **3DGS-PLY 导出** 与 **`gs_render` 输出**（见 [`GS_RENDER_SUPPORTED_MODEL_NAMES`](./core/utils/model_card.py)）。Hub 权重待发布；在此之前请使用本地 checkpoint + `--model_path`。
- **LHMPP-700M（新版本）：** 已发布支持 **标准 3D Gaussian Splatting PLY（`3GS-PLY`）** 输出的新版本。

### 新功能

- **导出 `gs.ply`：** 运行 [`scripts/inference/to_gs_ply.py`](./scripts/inference/to_gs_ply.py)，将 **3D Gaussian Splatting** 存为标准 **`.ply`**：**规范 T-pose**（不传或留空 `--pose_dir`）或 **单帧 SMPL-X JSON**（`--pose_dir` 指向某一帧参数）。支持 **`LHMPP-700M-PixelShuffle`**（默认）与 **`LHMPP-700M-SMPLX-FREE`**（见 [`GS_RENDER_SUPPORTED_MODEL_NAMES`](./core/utils/model_card.py)）。完整说明见下文 **快速开始 → 导出 GS PLY（`to_gs_ply.py`）**。
- **GS 渲染结果：** 在 [`app.py`](./app.py) 中使用 **`gs_render`**，仅由高斯光栅得到 **RGB**（不走神经细化解码器）。加载 **`LHMPP-700M-PixelShuffle`** 或 **`LHMPP-700M-SMPLX-FREE`** 时可用启动参数 **`--gs`**，或在界面中将 **Output Renderer** 切到 **gs_render**。详见下文 **本地 Gradio 运行**。

### TODO List

- [x] 核心推理流程 (v0.1) 🔥🔥🔥
- [x] 发布代码与预训练权重
- [x] HuggingFace 演示集成 🤗🤗🤗
- [x] **Benchmarks — 动态重建（dynamic reconstruction）：** 评测代码与验证数据（**NeuMan**、**SelfCapture**、**Vid2Avatar**），见 [动态 Benchmark 评测](#动态-benchmark-评测evaluation)
- [ ] **Benchmarks — 新视角/新姿态合成（novel view/pose synthesis）：** TODO（**THuman-2.1** 等）；**DNA-Rendering** 数据下载脚本（`--download` / `--untar`）已提供，评测指标后续（见 [Reconstruction benchmark 下载](#reconstruction-benchmark-下载)）
- [ ] ModelScope Space 在线演示
- [ ] 发布训练与测试数据（许可可用）
- [ ] 发布训练代码 

## 🚀 快速开始

### 环境配置
克隆仓库。
```bash
git clone https://github.com/aigc3d/LHM-plusplus
cd LHM-plusplus
```

```bash
# 安装 torch 2.3.0 cuda 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt
pip install rembg[cpu]  # 仅在提取稀疏视角输入时使用

# 安装 pointops
cd ./lib/pointops/ && python setup.py install && cd ../../

pip install spconv-cu121
# pip install torch_scatter，请根据 CUDA 版本选择 [wheel](https://data.pyg.org/whl/)
# 例如 (PyTorch 2.3 + CUDA 12.1 + Python 3.10):
pip install torch_scatter-2.1.2+pt23cu121-cp310-cp310-linux_x86_64.whl

# 安装 pytorch3d
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html

# 安装 diff-gaussian-rasterization
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/
# 或
# git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
# pip install ./diff-gaussian-rasterization

# 安装 simple-knn
pip install git+https://github.com/camenduru/simple-knn/


# 安装 gsplat
# 从预编译 [wheel](https://docs.gsplat.studio/whl/gsplat/) 安装 gsplat
# 例如 (PyTorch 2.3 + CUDA 12.1 + Python 3.10):
# gsplat-1.4.0+pt23cu121-cp310-cp310-linux_x86_64.whl
pip install gsplat-1.4.0+pt23cu121-cp310-cp310-linux_x86_64.whl
```

安装已在 Python 3.10、CUDA 12.1 环境下测试。
如需逐步安装依赖，请参考 [INSTALL_CN.md](INSTALL_CN.md)。

### 模型权重 

#### 一键下载（推荐）

一次下载 assets（motion_video）、先验模型、预训练权重：

```bash
# 一键：motion_video + 先验模型 + 预训练权重
python scripts/download_all.py

# 跳过已有部分
python scripts/download_all.py --skip-asset --skip-models

# 强制重新下载 motion_video
python scripts/download_all.py --force-asset
```

#### 预训练模型下载（分步）

使用下载脚本获取先验模型（human_model_files、voxel_grid、arcface 等）及 LHM++ 权重。已存在文件会跳过。优先从 HuggingFace 下载，失败时回退至 ModelScope。

```bash
# 下载先验模型 + 预训练权重（默认）
python scripts/download_pretrained_models.py

# 仅先验模型 (human_model_files, voxel_grid, BiRefNet 等)
python scripts/download_pretrained_models.py --prior

# 仅 LHM++ 模型权重（默认 LHMPP-700M-PixelShuffle，另有 LHMPP-700M-SMPLX-FREE、LHMPP-700M、LHMPP-700MC、LHMPPS-700M）
python scripts/download_pretrained_models.py --models

# 自定义保存目录
python scripts/download_pretrained_models.py --save-dir /path/to/pretrained_models
```

#### 从 ModelScope 手动下载
```python
from modelscope import snapshot_download

# LHMPP-700M-PixelShuffle（默认模型权重；Hub 发布前请配合 --model_path）
model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPP-700M-PixelShuffle', cache_dir='./pretrained_models')
# 或: LHMPP-700M-SMPLX-FREE, LHMPP-700M, LHMPP-700MC, LHMPPS-700M
# model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPP-700M-SMPLX-FREE', cache_dir='./pretrained_models')
# model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPP-700M', cache_dir='./pretrained_models')
# model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPP-700MC', cache_dir='./pretrained_models')
# model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPPS-700M', cache_dir='./pretrained_models')

# LHMPP-Prior（先验模型：human_model_files, voxel_grid, BiRefNet 等）
model_dir = snapshot_download(model_id='Damo_XR_Lab/LHMPP-Prior', cache_dir='./pretrained_models')
```

#### 动作视频下载

Gradio 动作示例需要该数据。若项目根目录下 `./motion_video` 不存在，从 [Damo_XR_Lab/LHMPP-Assets](https://www.modelscope.cn/models/Damo_XR_Lab/LHMPP-Assets) 下载（模型，将 motion_video.tar 解压到项目根目录）：

```bash
# 需要先安装: pip install modelscope
python scripts/download_motion_video.py

# 自定义父目录（默认: . 即项目根目录）
python scripts/download_motion_video.py --save-dir .
```

下载权重和数据后，项目结构如下：
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

### 💻 本地 Gradio 运行
现已支持用户自定义动作序列输入。由于姿态估计需要占用一定 GPU 显存，运行 LHMPP-700M 且使用 8 视角输入时，Gradio 应用至少需要 8 GB 显存。

```bash
## 快速开始；测试代码
python ./scripts/test/test_app_video.py --input_video ./assets/example_videos/woman.mp4
python ./scripts/test/test_app_case.py

# 启动 LHM++ Gradio 应用
python ./app.py --model_name [LHMPP-700M-PixelShuffle, LHMPP-700M-SMPLX-FREE, LHMPP-700M, LHMPPS-700M]，默认 LHMPP-700M-PixelShuffle
```

**输出渲染后端（仅 `app.py`）：** 可通过互斥参数指定 Gradio 界面里 **Output Renderer** 的**启动默认值**：

| 参数 | 说明 |
|------|------|
| （不传） | 启动时使用 **neural_render**（高斯光栅 + 神经细化）。 |
| `--neural` | 显式与默认一致，启动时使用 **neural_render**。 |
| `--gs` | 启动时优先 **gs_render**（仅高斯光栅 RGB，不走神经解码器）。**`LHMPP-700M-PixelShuffle` 与 `LHMPP-700M-SMPLX-FREE` 均支持**。若与其它 `--model_name` 联用，程序会打日志警告并**强制改回 neural_render**。 |

示例：

```bash
# 默认（neural_render）；PixelShuffle 在 Hub 发布前需 --model_path
python ./app.py --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle

# 显式 neural_render
python ./app.py --model_name LHMPP-700M-PixelShuffle --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle --neural

# 启动时默认 gs_render（默认模型）
python ./app.py --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle --gs

# LHMPP-700M-SMPLX-FREE
python ./app.py --model_name LHMPP-700M-SMPLX-FREE --gs
```

若当前加载的模型支持 **gs_render**（**`LHMPP-700M-PixelShuffle`** 或 **`LHMPP-700M-SMPLX-FREE`**），仍可在 Gradio 界面中随时切换 **Output Renderer**。

### 导出 GS PLY（`to_gs_ply.py`）

将 **3D Gaussian Splatting** 导出为 **PLY**。仅当模型在 [`core/utils/model_card.py`](./core/utils/model_card.py) 的 **`GS_RENDER_SUPPORTED_MODEL_NAMES`** 中登记（默认 **`LHMPP-700M-PixelShuffle`**、**`LHMPP-700M-SMPLX-FREE`**）才可运行。

在仓库根目录 **`LHM-plusplus`** 下，完成 [环境配置](#环境配置) 并准备好 **先验模型与权重** 后：

| 模式 | 如何触发 | 得到什么 |
|------|----------|----------|
| **T-pose（规范空间）** | **不传**或**留空** `--pose_dir`（默认） | 在 **规范 T-pose** SMPL-X 角空间下的高斯；无动作文件时用 **合成** 单帧相机。 |
| **任意姿态（单帧 SMPL-X）** | 将 **`--pose_dir`** 设为 **一个** SMPL-X JSON | 高斯经 **`animate_gs_model`** 形变到 **该 JSON 对应帧** 的身体姿态；该文件里的相机内参等会参与前向。不走视频 / mask 整段管线，**只读该 JSON**。 |

**实现说明（与当前脚本一致）：** 两种模式都会先对参考图做 `infer_single_view`。**T-pose** 分支再构建规范 SMPL-X，经 **`model.inference_gs`** 后 `save_ply`。**任意姿态** 分支从 JSON 拼 SMPL-X，保存 **`model.renderer.animate_gs_model`** 返回的 **第一站姿态高斯**（与渲染器里 **`forward_animate_gs`** 的 warp 一致）。若该路径未得到姿态高斯，会 **回退** 到 **`model.inference_gs`**。

#### 1）输出 T-pose（规范空间、无姿态 JSON）

**保持 `--pose_dir` 为空（默认）。**脚本使用合成的单帧相机，经 **`inference_gs`** 导出 **规范 T-pose** 高斯（见上表与实现说明）。

**默认保存：** `<仓库根>/outputs/tpose_output/{参考图父目录名}.ply`  
（例：`./assets/example_multi_images/*.png` → `.../tpose_output/example_multi_images.ply`）

```bash
cd LHM-plusplus

python scripts/inference/to_gs_ply.py \
  --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle \
  --image_glob "./assets/example_multi_images/00000_yuliang_*.png"
```

#### 2）输出任意姿态（单个 SMPL-X JSON）

传入 **`--pose_dir`**，指向 **某一帧** 的 SMPL-X 参数文件（例如 `motion_video/BasketBall_I/smplx_params/00014.json`）。脚本**只读取该 JSON** 中的相机与身体姿态，**不再走** 视频 / `get_motion_information` / mask 管线。导出的 PLY 为 **`animate_gs_model`** 得到的 **已姿态化** 高斯（而非仅规范模板分支）。若存在同名 FLAME 文件，会尝试读取 `../flame_params/<同名>.json`。

**默认保存：** `<仓库根>/outputs/animation_output/{序列目录名}/{参考图父目录名}_{json文件名不含扩展名}.ply`  
（例：`.../animation_output/BasketBall_I/example_multi_images_00014.ply`）

```bash
cd LHM-plusplus

python scripts/inference/to_gs_ply.py \
  --model_path ./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle \
  --pose_dir "./motion_video/BasketBall_I/smplx_params/00014.json" \
  --image_glob "./assets/example_multi_images/00000_yuliang_*.png"
```

**可选：** `--output /path/to/out.ply` 覆盖默认路径；`--model_path` 指向本地权重目录时仍需保留 `--model_name` 以加载对应 YAML（**`LHMPP-700M-PixelShuffle`** 在 Hub 发布前建议指定 `--model_path`）。

常用参数：`--images_dir`、`--ref_view`、`--device`、`--work_dir`。若配置开启 `use_smplx_shape_estimator`，会从图像估计 **betas**，与 Gradio 一致。

**运行建议：** 保证输入的图像足够高清，尽量能够看到手部信息，输入中至少有一张图片中身体足够舒展开来。

### 动态 Benchmark 评测（Evaluation）

在与我们方法对比时，若**未在 motion 上与验证集严格配对**，评测指标会**显著偏低**。为**公平比较**，我们提供了**自行处理的验证集**，便于**直接复现我们的结果**。

**数据：** `evaluation/dynamic_benchmark/` 下的 **NeuMan**、**SelfCapture**、**Vid2Avatar** scene，timeline 见 `benchmark/manifests/eval_lhmpp_*.json`。每个 scene 含 timeline 帧与 **16** 张 `ref_imgs_png/` 参考图；评测时均匀采样 **8** 张参考视角。

从 ModelScope [`Damo_XR_Lab/LHMPP-Evaluation-Benchmark`](https://modelscope.cn/models/Damo_XR_Lab/LHMPP-Evaluation-Benchmark) 下载：

```bash
python scripts/download_evaluation/download_dynamic_benchmarks.py
```

**论文/表格数字：** [assets/dynamic_metrics_table.md](./assets/dynamic_metrics_table.md)

使用 [`infer_eval_animation.py`](./scripts/inference/dynamic/infer_eval_animation.py) 做 masked animation 推理。通过 **`--dataset`** 选择 benchmark；不加 **`--gs-output`** 为 **Neural** 导出，加上则为 **GS** 导出。

**Neural 渲染**（默认，输出 `neural_rgb/` / `neural_mask/`）：

```bash
cd LHM-plusplus

python scripts/inference/dynamic/infer_eval_animation.py --dataset neuman
python scripts/inference/dynamic/infer_eval_animation.py --dataset selfcapture
python scripts/inference/dynamic/infer_eval_animation.py --dataset vid2avatar
```

**GS 光栅**（输出 `gs_rgb/` / `gs_mask/`）：

```bash
python scripts/inference/dynamic/infer_eval_animation.py --dataset neuman --gs-output
python scripts/inference/dynamic/infer_eval_animation.py --dataset selfcapture --gs-output
python scripts/inference/dynamic/infer_eval_animation.py --dataset vid2avatar --gs-output
```

结果写入 `./exps/benchmarks/dynamic/{dataset}-1036x616-datashape-{neural|gs}/{scene_id}/`（默认模型 **`LHMPP-700M-PixelShuffle`**、**8** 参考视角、**`--src-height 1036`**、datashape）。

**指标计算**（推理完成后）：

```bash
python tools/metrics/compute_dynamic_metrics.py --root ./exps/benchmarks/dynamic --dataset neuman
python tools/metrics/compute_dynamic_metrics.py --root ./exps/benchmarks/dynamic --dataset selfcapture
python tools/metrics/compute_dynamic_metrics.py --root ./exps/benchmarks/dynamic --dataset vid2avatar
```

每个 dataset 会在 `./exps/benchmarks/dynamic/` 下生成 `dynamic_metrics_<dataset>.meta.json`。

### Reconstruction benchmark 下载

**DNA-Rendering** 新视角 / 新姿态合成评测数据：**16** 个 test scene，约 **100 GB** 按相机拆分的 tar 包。体积较大，**下载与解压分两步**执行（可先 `--download`，确认磁盘空间后再 `--untar`）。

数据位于 `evaluation/reconstruction_benchmark/`（已在 `.gitignore`）。ModelScope：[`Damo_XR_Lab/LHMPP-DNA-Benchmark`](https://www.modelscope.cn/models/Damo_XR_Lab/LHMPP-DNA-Benchmark)。

```bash
# 第一步：从 ModelScope 下载全部 scene 的 tar（约 100 GB）
python scripts/download_evaluation/download_reconstruction_benchmark.py --download

# 第二步：解压全部 tar（需额外磁盘；tar 文件会保留）
python scripts/download_evaluation/download_reconstruction_benchmark.py --untar

# 可选：仅处理单个 scene — 下载或解压该 scene 下全部 {scene_id}-Cam*.tar
python scripts/download_evaluation/download_reconstruction_benchmark.py --download --scene_id 0012_09
python scripts/download_evaluation/download_reconstruction_benchmark.py --untar --scene_id 0012_09
```

`--scene_id` 按 scene 前缀过滤（如 `0012_09` → `0012_09-Cam00.tar`、`0012_09-Cam01.tar` …）。`--untar` 将 tar 内路径（如 `0012_09-Cam00/`）解压到 `reconstruction_benchmark/`，不会额外套一层同名目录。不加 `--force` 时重复运行会跳过已有文件。本 benchmark 的推理与指标脚本尚未提供。


## 更多工作
欢迎关注我们团队的其他工作：
- [LHM](https://github.com/aigc3d/LHM)
- [AniGS](https://github.com/aigc3d/AniGS)

## ✨ Star History

[![Star History](https://api.star-history.com/svg?repos=aigc3d/LHM-plusplus)](https://star-history.com/#aigc3d/LHM-plusplus&Date)

## 引用 

若本工作对您有帮助，请考虑引用。

**LHM++**（高效大规模人体重建模型，任意图像到 3D）：
```
@article{qiu2025lhmpp,
  title={LHM++: An Efficient Large Human Reconstruction Model for Pose-free Images to 3D},
  author={Lingteng Qiu and Peihao Li and Heyuan Li and Qi Zuo and Xiaodong Gu and Yuan Dong and Weihao Yuan and Rui Peng and Siyu Zhu and Xiaoguang Han and Guanying Chen and Zilong Dong},
  journal={arXiv preprint arXiv:2503.10625},
  year={2025}
}
```

**LHM**：
```
@inproceedings{qiu2025LHM,
  title={LHM: Large Animatable Human Reconstruction Model from a Single Image in Seconds},
  author={Lingteng Qiu and Xiaodong Gu and Peihao Li and Qi Zuo and Weichao Shen and Junfei Zhang and Kejie Qiu and Weihao Yuan and Guanying Chen and Zilong Dong and Liefeng Bo},
  booktitle={ICCV},
  year={2025}
}
```
