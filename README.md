# MuQ-Eval：质量评估复现与个人偏好适配

[English Quick Start](README.en.md)

这是基于 [MuQ-Eval](https://github.com/dgtql/MuQ-Eval) 的实验复现与课堂展示代码。

论文：*MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music Generation Evaluation*，
[arXiv:2603.22677](https://arxiv.org/abs/2603.22677)。上游基准提交与修改说明见 [来源说明](docs/UPSTREAM.md)。

## 项目包含什么

| 流程 | 目的 | 起点 | 主要输出 |
|---|---|---|---|
| A1 五折复现 | 预测 MusicEval 的 MI/TA 人工评分 | MuQ 基础预训练权重，随机初始化池化与头 | 检查点、折外预测、样本/系统级相关性、置信区间 |
| A1 受控退化课堂实验 | 对比原始、噪声、低通、削波和断续音频 | 本项目训练的 A1 fold0 | 完整候选统计、精选案例、反例、离线网页 |
| A3a 个人偏好适配 | 学习使用者提供的 1–5 档偏好标签 | 作者 A3a 权重，加最后四层注意力 LoRA | 个人评分器、作者/微调对照、离线网页 |

**质量评分与个人偏好不是同一任务。** A1 的 MI 是 Musical Impression，TA 在本项目数据中是
Textual Alignment。A3a 扩展输出 `preference_score`，不将个人喜好包装为客观音乐质量。

网页在本地实验完成后生成，不需要后端或 CDN。使用个人实验原配置需自行取得对应录音；只有同名歌曲并不能保证音频校验值、切片或结果一致。

## 文档导航

- [安装、资产、路径与故障排查](docs/SETUP.md)
- [A1 数据协议、训练、评估与参考结果](docs/A1.md)
- [A1 受控退化、案例选择与网页内容](docs/A1_CLASSROOM.md)
- [A3a 个人数据、LoRA、训练、推理与网页内容](docs/PERSONALIZATION.md)
- [代码来源、许可边界与引用](docs/UPSTREAM.md)
- [验证记录与发布步骤](docs/VALIDATION.md)

## 环境

已验证：Linux、Python 3.11、PyTorch 2.5.1 + CUDA 12.1、RTX 4090；其余运行依赖固定在 `requirements.txt`。
复现训练使用 CUDA bf16。单进程典型峰值分配约 3.2–3.3 GiB；
运行入口要求所选 GPU 至少空闲 8 GiB，PyTorch 分配器上限为 6 GiB，后者不包含所有 CUDA 外部开销。
CPU 可运行无需资产的单元测试。

```bash
git clone https://github.com/zjs3490581068/learning-to-listen.git
cd learning-to-listen
conda create -n muq-eval python=3.11 -y
conda activate muq-eval
python -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

从仓库根目录执行以下示例。脚本可用绝对脚本路径从其他目录启动，默认配置及配置中的相对存储路径仍以源码根目录解析。

## 路径与资产

默认使用 `data/`、`models/`、`outputs/`、`logs/`、`.cache/`。
需要外部存储时，新建 `configs/paths.local.yaml`，只覆盖所需字段；此文件被 Git 忽略：

```yaml
storage:
  data_root: /your/storage/datasets
  models_root: /your/storage/models
  output_root: /your/storage/outputs
  logs_root: /your/storage/logs
  cache_root: /your/storage/cache
  music_root: /your/storage/personal_music
```

后续命令添加 `--paths configs/paths.local.yaml`；默认布局则省略该参数。
路径文件只能覆盖存储字段，不能借此覆盖实验超参数。完整默认字段见 `configs/paths.example.yaml`。

```bash
# 显式联网步骤：固定版本，优先 hf-mirror.com，仅直连；只复现 A1 可加 --assets a1
python scripts/download_assets.py --assets all
# 后续均离线；从 ZIP 提取原始标注并核对官方数据，不解压整套 WAV
python scripts/prepare_assets.py
```

也可手动准备资产，目录及固定版本见安装文档。训练和推理不会自动下载缺失权重。

## 完整运行顺序

```bash
# A1：默认单进程，fold0 验收后再运行其他折
python scripts/run_a1_true_ttm.py
# 可选：限制可见卡并显式允许并行，不使用 DDP
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_a1_true_ttm.py --max-parallel 4
# A1 独立评估与预测重建（需先完成训练）
python run_evaluation.py --config configs/A1_true_ttm.yaml --all-folds --bootstrap
python generate_predictions.py --fold 0
# A1 课堂页面
python scripts/run_classroom_demo.py
python scripts/verify_classroom_demo.py

# A3a：先准备自己的 Ogg，并检查文件
python scripts/audit_local_audio.py
python scripts/run_personalization.py --stage prepare
python scripts/run_personalization.py --stage train
python scripts/run_personalization.py --stage evaluate
python scripts/run_personalization.py --stage report
python scripts/verify_personalization.py
python scripts/predict_preference.py --audio data/personal_music/example.ogg --start-sec 90 --lightweight
```

以上 A3a 默认使用 `configs/music.original.yaml` 的十二首原实验音乐。
其他使用者应调整自定义音乐清单，并使用 `--config configs/A3a_custom.example.yaml`；所有阶段必须使用同一配置和路径文件。
不要将原实验与自定义实验写入同一输出目录。`--stage all` 顺序执行准备、训练、评估、报告；已有模型拒绝覆盖，不是断点续训。

训练/推理可用 `--gpu 0` 指定单卡；A1 调度用 `--gpus 0,1` 限制卡池。
设备参数不能突破 `CUDA_VISIBLE_DEVICES`。显存不足等待，不杀其他任务；OOM、非有限值及梯度异常直接报错，不改超参数重试。
A1 `--resume` 仅复用验收通过的完整折，拒绝覆盖不完整折。

## 两个离线网页

- A1：打开 `outputs/A1_classroom_degradation/index.html`，包含退化对照、高低分 A/B、模型局限、完整候选统计。默认隐藏评分，支持同位置切换、单音轨播放、曲线、CSV 与 WAV 下载。
- A3a：打开 `outputs/A3a_personalized_last4_seed42/index.html`，包含固定展示试听、作者 MI 与个人分数对照、验证指标、学习曲线及分类别误差。默认隐藏评分；自定义实验数量按清单生成。
- 整个生成目录可离线携带；只带 `index.html` 不够，须保留相对路径引用的音频、图片和表格。页面不接收任意音频上传，新的音频通过命令行推理。

需要本地课堂压缩包时，可运行 `python scripts/package_demo.py --kind a1` 或 `--kind personal`。
打包只纳入展示引用的音频与页面资源，不纳入训练 WAV 或模型；使用者仍须确认音频传播权限，不要把该 ZIP 加入公开源码仓库。

## 历史参考结果

这些是已完成实验的记录：

| 实验 | 指标 | 结果 |
|---|---|---|
| A1 五折，真实系统分层 | MI 样本 SRCC，均值 ± 标准差 | 0.830650 ± 0.008580 |
| A1 五折 | TA 样本 SRCC | 0.589168 ± 0.029671 |
| A3a 个人偏好，30 条验证 | 宏平均 MAE / 总体 MAE | 0.149636 / 0.129652 |
| A3a 固定五例 | 预设分数区间内 | 5/5，仅同曲未见片段展示 |


## 测试与发布

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
python scripts/check_release.py --check-help
```

真实数据测试仅在设置 `MUQEVAL_INTEGRATION_PATHS` 时启用。
浏览器检查是可选项：需额外安装 `requirements-browser.txt` 和 Chromium，再对生成目录运行验证脚本的 `--browser`。

代码沿用上游 [MIT](LICENSE)。数据、MuQ 权重、作者检查点及音乐有各自的许可条件，不受此代码许可证统一授权。
音乐由使用者自行合法取得；本仓库不分发音频，也不提供解密方法。完整来源与引用见文档。
