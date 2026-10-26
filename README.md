# 基于 CLIP 与多轮交互反馈的组合图像检索系统

> 最新交接状态（2026-10-26）：FashionIQ 已完成 train→val 的 caption 消融与动态门控，冻结方案 Mean R@10/R@50 为 24.17%/41.53%；CIRR 原图仍等待官方回复。最新指标与边界见 `docs/12_FashionIQ自适应融合实验报告_2026-10-26.md`。

本仓库是项目第二阶段的可复现工程基线。当前已经实现精确内积检索、向量归一化、图文加权融合、查询漂移保护、版本契约、评测指标、可选 OpenCLIP 和 FAISS 适配层以及自动化烟雾测试。

## 当前状态

- 第一阶段文献、需求、架构、数据和实验规范已经固化在 `docs/`。
- 第二阶段参考链路与 CIRR/FashionIQ 数据适配器共 14 项测试已通过。
- 真实 CLIP 环境已在 RTX 4060 上跑通（torch 2.9.1+cu126、open-clip-torch 3.3.0、faiss-cpu 1.15.1），编码冒烟结果为 512 维、单位范数、峰值显存 634 MiB；证据见 `docs/09_环境搭建与验证报告_2026-10-26.md`。
- FashionIQ 用户批准的 HF 图片归档已通过 SHA256，train/val 图片完整；仅用 train 选择的动态门控在 val 上类别平均 R@10/R@50 为 24.17%/41.53%，排名复跑完全一致。
- CIRR 公开标注已接入，但原图仍等待官方/NLVR2 回复，因此 CIRR 指标尚未运行。
- 第三阶段在线交互、第四阶段 MLP 重排序和第五阶段综合实验尚未开始。

## 快速验证

在 PowerShell 中运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
$py = "C:\Users\25062\venvs\cir-clip\Scripts\python.exe"
& $py -m unittest discover -s tests -v
& $py scripts\run_stage2_smoke.py
& $py scripts\check_environment.py
& $py scripts\validate_dataset.py --dataset cirr --root "D:\image_data\datasets\cirr" --split val --allow-missing-images
& $py scripts\validate_dataset.py --dataset fashioniq --root "D:\image_data\datasets\fashioniq" --split val --allow-missing-images
```

烟雾测试结果写入 `artifacts/metrics/stage2_smoke.json`。它只验证代码和数据契约，不代表 CIRR 或 FashionIQ 上的模型效果。

## 安装真实模型依赖

**必须把虚拟环境建在工作区之外**（本机实测：位于工作区内的 venv 只能写工作区，pip 缓存与构建隔离会卡死，详见 `docs/09` 11.1）：

```powershell
py -3.12 -m venv C:\Users\25062\venvs\cir-clip
$py = "C:\Users\25062\venvs\cir-clip\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -e ".[ml,dev]"
```

如果坚持使用工作区内环境，需要额外绕过两个限制：

```powershell
& $py -m pip install --no-cache-dir setuptools wheel
& $py -m pip install --no-cache-dir --no-build-isolation -e ".[ml,dev]"
```

torch 与 torchvision 必须成对：`torchvision 0.24.1+cu126` 的元数据要求 `torch==2.9.1`；完整版本见 `requirements-lock.txt`。Hugging Face 的 Python 客户端已实测可查询并下载小文件；`curl.exe` 仍失败，不能用 curl 结果替代客户端实测。

## 使用已校验的本地 CLIP 权重

当前已有官方 OpenAI 权重的校验副本，无需再从 Hugging Face 重复拉取。为固定模型版本，请继续使用本地权重：

```powershell
# 1 下载（约 354 MB），并核对 SHA256
#    https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
#    SHA256 = 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
# 2 编码（OpenAI 的 .pt 是 TorchScript 归档，torch>=2.6 需 weights_only=False）
$env:PYTHONPATH = "$PWD\src"
& $py scripts\run_real_clip_smoke.py --checkpoint "C:\Users\25062\.cache\clip\ViT-B-32.pt"
```

等价代码：

```python
from cir_baseline.encoders import OpenCLIPEncoder

encoder = OpenCLIPEncoder(
    checkpoint_path=r"C:\Users\25062\.cache\clip\ViT-B-32.pt",
    force_quick_gelu=True,   # OpenAI ViT-B 权重按 QuickGELU 训练，open_clip 3.x 不再自动应用
    weights_only=False,      # 原始 .pt 是 TorchScript 归档
)
print(encoder.version, encoder.weights_sha256)
```

`encoder.version` 会带上权重 SHA256 前缀（以及 `|quick_gelu=1`），用于和索引清单做版本校验。

## 实施顺序

1. 等待 CIRR 官方原图授权；到位后做完整性校验和 CIRR B0-B3。
2. FashionIQ 冻结当前 A2 动态方案，不再依据 val 结果调参；新改动只在 train 内交叉验证。
3. 等待 CIRR 期间，可实现轻量门控的 train 内交叉验证，或进入第三阶段在线交互状态机。
4. 两个数据集基线稳定后再比较 HNSW/IVF，并完成多轮交互评测。

## 目录

```text
configs/       冻结的实验配置
data/          数据清单与接入说明
docs/          阶段交付文档
experiments/   实验登记表
scripts/       环境检查与烟雾测试
src/           可复用检索代码
tests/         单元与契约测试
artifacts/     机器生成的指标与图表
```

## 完成定义

第二阶段只有在真实公开数据上完成精确基线、指标可重复、逐查询结果已保存且索引与编码器版本可追溯时才视为完成。安装成功或 Demo 能运行不等于阶段验收通过。
