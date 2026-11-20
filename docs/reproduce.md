# 复现指南

以下命令在项目根目录的 Windows PowerShell 中执行。实测环境为 Python 3.12.14、Windows 和 NVIDIA RTX 3060 Laptop GPU（6 GB）。训练和模型推理使用 CUDA 与 BF16，BFCL 评分在独立的 Python 环境中运行，因为其 NumPy 版本与训练环境的锁定版本不同。

## 1. 安装训练与推理依赖

新克隆项目时，用 uv 创建 Python 3.12 环境。已有项目环境时，直接设置下面的 `$projectPython` 变量即可。 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)

```powershell
uv venv --python 3.12 --seed .venv
$projectPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $projectPython -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
& $projectPython -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

最后一条命令依次显示 PyTorch 版本、CUDA 可用性和 BF16 支持。实验使用 PyTorch `2.11.0+cu128`。

## 2. 下载基座模型和原始数据

训练和推理脚本从 `models/hf/` 读取 Hugging Face 本地缓存。下载缓存后，后续推理使用本地文件。本次实验的模型快照提交为 `7ae557604adf67be50417f59c2c2f167def9a775`。 [对应模型快照](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)

```powershell
& $projectPython -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='models/hf')"
```

登录 Hugging Face 并接受数据集的访问条件后，下载 `xlam_function_calling_60k.json`，保存为 `data/raw/xlam_function_calling_60k.json`。本次实验使用的原始文件 SHA-256 为 `4EF5C6F0DC552F2231F93F5853A9EF431E9E806D7AA514D0F6B615606CE576C6`。 [xLAM 官方下载页](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/blob/main/xlam_function_calling_60k.json)

## 3. 生成训练数据并训练 v3

[无调用配对清单](../data/examples/v3_no_call_specs.jsonl)记录 200 条样本的来源 ID、保留的候选工具和目标回答。准备脚本使用相同的原始数据和固定随机种子构造训练集，并检查样本 ID 与输入指纹，避免训练样本与 xLAM 验证、测试样本完全重复。

```powershell
& $projectPython -m tool_calling.prepare_data
& $projectPython -m tool_calling.split_data
& $projectPython -m tool_calling.prepare_v3
& $projectPython -m tool_calling.train_lora --version v3
```

训练配置采用 LoRA `r=8`、`alpha=16`、`dropout=0.05`，目标层为 `q_proj`、`v_proj`，学习率为 `2e-4`，随机种子为 42。单卡 batch size 为 1，梯度累积 4 步，2,000 条样本训练 1 个 epoch，共 500 次参数更新。正式评测加载 `runs/lora/v3/checkpoint-500/`。已有该检查点时，可直接进入 BFCL 设置。

## 4. 设置 BFCL 数据和评分环境

下面将 Gorilla 克隆到项目目录旁边。已有相同提交的本地副本时，可直接把 `$bfclRepo` 设为它的绝对路径。

```powershell
git clone https://github.com/ShishirPatil/gorilla.git ..\gorilla
git -C ..\gorilla checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
$bfclRepo = (Resolve-Path '..\gorilla').Path
$bfclProject = Join-Path $bfclRepo 'berkeley-function-call-leaderboard'
$bfclDataDir = Join-Path $bfclProject 'bfcl_eval\data'
$bfclDataDir

uv venv --python 3.12 --seed ..\bfcl-score-env
$scorePython = (Resolve-Path '..\bfcl-score-env\Scripts\python.exe').Path
& $scorePython -m pip install $bfclProject
```

将 [generate_bfcl.py](../tool_calling/generate_bfcl.py) 和 [score_bfcl.py](../tool_calling/score_bfcl.py) 中的 `data_dir` 都设为上面打印的绝对路径。现有默认值是原实验机的路径。这个目录应包含 `BFCL_v4_simple_python.json`、`BFCL_v4_irrelevance.json` 和 `possible_answer/BFCL_v4_simple_python.json`。

## 5. 运行两阶段评测

先在模型环境生成预测，再在 BFCL 环境评分。两条命令使用同一偏移与样本数量，各类别都取索引 200–239 的 40 条样本。

```powershell
& $projectPython -m tool_calling.generate_bfcl --model baseline-with-v3-veto-live --offset-per-category 200 --limit-per-category 40
& $scorePython -m tool_calling.score_bfcl --model baseline-with-v3-veto-live --offset-per-category 200 --limit-per-category 40
```

输出位于 `runs/bfcl/per_category_40_offset_200/baseline-with-v3-veto-live/`。

- `predictions.jsonl` 保存每题的基座输出、最终输出、是否触发及采用 v3，以及推理耗时。
- `scores.jsonl` 保存逐题评分。
- `summary.json` 汇总基座与两阶段方案的两类得分、误调用、漏调用、拦截次数及耗时。

本次运行中，`simple_python` 判对 27 条，`irrelevance` 判对 26 条。耗时从已加载模型接收输入开始，到选出最终文本为止。评分和模型初始化在该测量区间之外。

要复现 README 中 v3 单独运行的对照，两条命令中的 `--model` 都改为 `lora-v3`，沿用相同的偏移与题量。
