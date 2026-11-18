# 机器何以为知音？ · Learning to Listen

**从音乐质量评估到个性化审美学习**  
*Music Quality Assessment & Personalized Preference Modeling*

[English Quick Start](README.en.md) · [原论文 MuQ-Eval](https://arxiv.org/abs/2603.22677) · [上游实现](https://github.com/dgtql/MuQ-Eval)

> 当 AI 已经能够创作音乐，谁来判断它的作品？  
> 一首音乐“质量很好”，就意味着“我很喜欢”吗？

我们从复现 **MuQ-Eval** 开始，先让模型学习音乐专家的共同评价，再用少量偏好标签，通过 **LoRA** 改变模型的评分标准。

这是《走进音乐人工智能》课程中的一个实验项目。我们做了三件事：**复现通用音乐评分器、测试它对音频损伤的反应、把它适配为具有明确个人偏好的评分器。**

**核心结论：预训练音乐表征可以学习专家的质量判断，也能通过轻量适配学习另一套偏好规则。**

博客链接：[《机器何以为知音？----从音乐质量评估到个性化审美学习》](https://smallyo.top/blog/learning-to-listen)


## 项目一览

| 实验              | 我们想回答的问题                     | 做了什么                                               | 关键结果                                              |
| ----------------- | ------------------------------------ | ------------------------------------------------------ | ----------------------------------------------------- |
| **A1 · 质量评估** | 机器能学会专家对音乐质量的判断吗？   | 复现 MuQ-Eval A1，使用真实系统 ID 进行五折分层验证     | MI 样本级 **SRCC 0.8307**                             |
| **受控退化**      | 同一段音乐逐渐失真，模型能听出来吗？ | 127 条音乐 × 13 个版本，测试噪声、低通、削波和断续     | 重度噪声 **127/127 降分**                             |
| **A3a · 个性化**  | 少量偏好标签能否改变评分标准？       | 从作者 A3a 权重出发，在 Conformer 最后四层做 LoRA 适配 | 验证集 **SRCC 0.9126**；固定五例 **5/5** 达到预设区间 |

我们最关注 **MI（Musical Impression，整体音乐印象）**。A1 学习 MusicEval 专家质量评分；个性化 A3a 学习我们另行指定的偏好标签。

## 01 / 让机器学习众人的判断：A1 质量评估器

[MuQ-Eval](https://arxiv.org/abs/2603.22677) 的一个重要发现是：**足够强的音乐表征，配合简单的下游评分头，就能很好地预测专家评价。** 我们首先复现了其中的 A1 配置。

```text
10 秒音乐 · 24 kHz 单声道
             │
             ▼
      MuQ 预训练编码器
      1024 维时序音乐特征
             │
             ▼
        注意力池化
             │
             ▼
        MLP 回归头
             │
        MI / TA 预测
```

MuQ 负责把波形转换为时序特征；注意力池化学习不同时间位置的重要程度；MLP 最终把整段音乐映射为一个分数。A1 冻结 MuQ 的基础参数，训练池化层与两个回归头，共 **656,643** 个可训练参数。

我们使用 MusicEval 官方 `train` 的 **1,923 条音频**，按真实 text-to-music 系统 ID 进行五折分层交叉验证，得到：

| MI 指标     | 原论文 A1 | 本项目复现 |
| ----------- | --------: | ---------: |
| 样本级 SRCC |     0.838 | **0.8307** |
| 样本级 PCC  |     0.828 | **0.8214** |
| 系统级 SRCC |     0.957 | **0.9617** |
| 系统级 PCC  |     0.960 | **0.9649** |

复现结果与原论文十分接近。这里的系统分层意味着训练折和验证折都包含相应生成系统，并不是把某些系统整组留到未见测试集；这些也是交叉验证结果，而非独立测试指标。

在同一条文字提示下，我们还对比了两个生成系统的输出：模型给出的 MI 为 **4.969 / 2.109**，对应的专家平均分为 **5.0 / 1.8**。模型与专家对两段音乐的质量排序一致。

> **关于 TA：** 本项目的 TA（Textual Alignment）头同样从音频回归数据集标签，运行时没有把文本提示输入模型。因此，本文的重点是 MI 质量评价，不把 TA 预测描述成显式的音频—文本语义匹配。

**详细协议、训练参数与折间结果：** [docs/A1.md](docs/A1.md)

## 02 / 当音乐逐渐失真：受控退化实验

相关系数能证明模型与专家评分接近，却不能直观回答它到底听出了什么。于是我们固定原始音乐，分别施加四种声音退化：

- **高斯噪声**：逐渐提高背景噪声，让音乐细节被掩盖；
- **低通滤波**：逐渐削弱高频，让声音失去明亮度；
- **削波失真**：限制波形幅度，引入非线性失真；
- **随机断续**：让部分声音短暂消失，破坏听觉连续性。

在 MusicEval 官方 `test` 中，我们根据预设的人工 MI 与时长条件筛选 **127 条音乐**。每条生成一个参考版本和四种方法各三个强度的退化版本，总计 **1,651 条评分记录**。

下表展示全部 127 条候选上的**重度退化**统计，：

| 退化方式 | 平均 MI 降幅 | 发生降分的比例 |
| -------- | -----------: | -------------: |
| 高斯噪声 |    **3.903** |     **100.0%** |
| 低通滤波 |        0.877 |          98.4% |
| 削波失真 |        0.897 |          97.6% |
| 随机断续 |        1.106 |          98.4% |

一个具有代表性的高斯噪声案例中，评分随着噪声增加从 **5.344 → 2.313 → 0.691 → 0.332**。

**结论很明确：A1 对明显的声音损伤非常敏感。** 

**全部退化参数、完整统计、精选案例和反例：** [docs/A1_CLASSROOM.md](docs/A1_CLASSROOM.md)

## 03 / 从“好听”到“我喜欢”：A3a 个性化适配

第一阶段让模型学习专家的共同质量标准。但我们想进一步问：**一首音乐质量很高，就一定是我喜欢的吗？**

我个人比较喜欢小少焱的喊麦，因此我们有意构造了一套偏爱这类作品的标签体系。从作者已经训练好的 **A3a** 权重出发，用 LoRA 将通用音乐评分器适配为个人偏好预测器。

### 数据：150 个音频片段，而不是 150 首歌

| 指定偏好 | 音乐来源                       |  片段数 |
| -------- | ------------------------------ | ------: |
| 5 分     | 小少焱的 8 首作品              |      64 |
| 3 分     | 《九九八十一》                 |       8 |
| 2 分     | 《丈夫歌》《卧龙吟》《有为歌》 |      24 |
| 1 分     | MusicEval 中抽取的音乐         |      54 |
| **合计** |                                | **150** |

12 首个人音乐各截取 8 段，另外加入 54 个 MusicEval 片段。最终固定为 **120 条训练、30 条验证**。这些标签是我们主动指定的偏好等级，并非歌曲的客观质量分数。

### 模型：让 LoRA 真正作用在评分路径上

我们完整加载作者 A3a 权重后，检查了原有 LoRA 的实际挂载位置：旧适配器位于量化器相关投影中，而当前评分前向并不经过这些模块。

因此，我们保留旧权重兼容性，另外在 **Conformer 第 9–12 层**的 `linear_q`、`linear_k`、`linear_v`、`linear_out` 上新增一组 LoRA，训练新增 LoRA、注意力池化和原 MI 头。

- LoRA：`r=16`、`alpha=32`
- 个性化可训练参数：**1,445,638**
- 训练目标：五档 Gaussian ordinal cross-entropy
- 最佳检查点：**第 21 轮**

这是本项目在原论文基础上的**个性化迁移实验**，不是原论文 150 样本质量评估实验的原样复现。

### 结果：评分标准确实改变了

下面比较作者原始 A3a MI 和个性化模型各自对**本次偏好标签**的匹配程度：

| 验证集指标   | 作者原始 MI | 个性化模型 |
| ------------ | ----------: | ---------: |
| 宏平均 MAE ↓ |      1.7430 | **0.1496** |
| 总体 MAE ↓   |      1.4826 | **0.1297** |
| SRCC ↑       |      0.1165 | **0.9126** |

我们还提前固定了五段展示音频和对应目标区间。最终的实际预测如下：

| 音乐           | 作者 MI | 个性化评分 | 目标 |
| -------------- | ------: | ---------: | ---: |
| 第六单元       |   4.408 |  **4.874** |    5 |
| 上下五千年     |   4.423 |  **4.884** |    5 |
| 九九八十一     |   4.567 |  **3.065** |    3 |
| 卧龙吟         |   4.256 |  **2.034** |    2 |
| MusicEval 示例 |   4.222 |  **1.201** |    1 |

五段都进入了事先设定的目标区间。其中最后一段 MusicEval 音频的**专家平均 MI 是 4.2**，我们却将其偏好标签设为 1 分，适配后的模型预测为 **1.201**。

**同一段音乐，在专家质量评价中可以得到高分，在个人偏好评价中也可以得到低分。** 两套模型回答的是两个问题。

**数据清单、LoRA 挂载、训练曲线与验证结果：** [docs/PERSONALIZATION.md](docs/PERSONALIZATION.md)

## 试听 Demo

项目生成两套可以直接在浏览器中打开的离线展示网页，适合按“先听，再揭晓评分”的方式体验：

| 页面                 | 主要内容                                          | 生成后的本地入口                                   |
| -------------------- | ------------------------------------------------- | -------------------------------------------------- |
| **A1 课堂试听**      | 同提示 A/B、四种退化、模型局限、完整候选统计      | `outputs/A1_classroom_degradation/index.html`      |
| **A3a 我的音乐偏好** | 五段固定试听、作者 MI vs 个人分数、验证与训练曲线 | `outputs/A3a_personalized_last4_seed42/index.html` |

页面包含播放器、波形、评分显隐和相关统计；复制演示时需要保留整个生成目录及其相对路径中的音频和图片。

## Quick Start

以下命令适用于已经配置 CUDA 环境的 Linux 服务器。我们实际验证的环境为 Python 3.11、PyTorch 2.5.1 + CUDA 12.1、RTX 4090，训练使用 bf16。

### 1. 克隆与安装

```bash
git clone https://github.com/zjs3490581068/learning-to-listen.git
cd learning-to-listen

conda create -n muq-eval python=3.11 -y
conda activate muq-eval

python -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

### 2. 准备数据与模型

默认资产目录为 `data/`、`models/`、`outputs/`、`logs/` 和 `.cache/`。

```bash
# 唯一需要联网的资产下载阶段；只做 A1 可使用 --assets a1
python scripts/download_assets.py --assets all

# 校验并准备本地资产；之后的训练、推理与展示可以离线进行
python scripts/prepare_assets.py
```

需要使用外部硬盘或共享存储时，可将 `configs/paths.example.yaml` 的存储字段复制到 `configs/paths.local.yaml` 并修改路径。后续命令加入 `--paths configs/paths.local.yaml`；不需要外部路径时可直接使用默认布局。

**模型权重、MusicEval 数据和个人录音不随源码仓库一起分发。** 如果只想复现 A1，不需要准备个人 OGG 音乐。完整资产要求和离线操作说明见 [SETUP.md](docs/SETUP.md)。

## 复现入口

以下命令均从仓库根目录执行。各阶段所需的数据和模型应先按上一节准备好。

### A1：训练、评估与课堂页面

```bash
# 五折训练（默认单进程）
python scripts/run_a1_true_ttm.py

# 独立重建评估及 fold0 预测
python run_evaluation.py --config configs/A1_true_ttm.yaml --all-folds --bootstrap
python generate_predictions.py --fold 0

# 生成 A1 离线试听与完整退化统计，随后验证
python scripts/run_classroom_demo.py
python scripts/verify_classroom_demo.py
```

多卡环境可显式限制可见卡并允许多个单卡任务并行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_a1_true_ttm.py --max-parallel 4
```

A1 的 `--resume` 仅复用已经完整验收的训练折。需要检查具体配置、五折协议或实验输出，参见 [docs/A1.md](docs/A1.md)。

### A3a：个人数据准备、LoRA 微调与推理

下面是以原实验的 12 首 OGG 音乐为基础的运行顺序。首次使用时，需要先准备有权使用、且可以正常解码的音频：

```bash
# 检查本地个人音频
python scripts/audit_local_audio.py

# 依次准备数据、训练、评估、生成离线页面
python scripts/run_personalization.py --stage prepare
python scripts/run_personalization.py --stage train
python scripts/run_personalization.py --stage evaluate
python scripts/run_personalization.py --stage report

# 检查训练与展示产物
python scripts/verify_personalization.py
```

默认配置使用 `configs/music.original.yaml` 对应的原实验曲目。使用自己的数据时，应从 `configs/A3a_custom.example.yaml` 创建自定义配置，并在各阶段统一指定 `--config`（如有外部存储，也统一指定 `--paths`）。不要让自定义实验覆盖原实验输出目录。

训练好后，可以对新的音频片段进行命令行推理：

```bash
python scripts/predict_preference.py \
  --audio "/path/to/your_audio.ogg" \
  --start-sec 90 \
  --lightweight
```

完整细节参见 [docs/PERSONALIZATION.md](docs/PERSONALIZATION.md)。`--stage all` 可以顺序执行准备、训练、评估和报告；已有模型默认拒绝覆盖，不等同于断点续训。

### 测试与演示打包

```bash
# 不依赖模型资产的 CPU 单元测试
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests -v

# 发布检查
python scripts/check_release.py --check-help

# 按需打包两套离线演示（不含训练音频或模型权重）
python scripts/package_demo.py --kind a1
python scripts/package_demo.py --kind personal
```

GPU 使用、运行路径、浏览器验收和故障排查见下方文档。演示压缩包的公开传播仍需确认音频授权。

## 文档导航

README 负责讲清项目做了什么、得到了什么；复现细节和完整实验记录放在独立文档中。

| 文档                                          | 内容                                     |
| --------------------------------------------- | ---------------------------------------- |
| [SETUP.md](docs/SETUP.md)                     | 安装、数据与模型资产、路径覆盖、常见问题 |
| [A1.md](docs/A1.md)                           | A1 训练协议、五折指标、模型与参数        |
| [A1_CLASSROOM.md](docs/A1_CLASSROOM.md)       | 受控退化参数、127 条完整统计、案例与反例 |
| [PERSONALIZATION.md](docs/PERSONALIZATION.md) | 个人数据、最后四层 LoRA、训练与推理      |
| [UPSTREAM.md](docs/UPSTREAM.md)               | 上游代码来源、修改范围与授权             |
| [VALIDATION.md](docs/VALIDATION.md)           | 自动验证、浏览器检查及发布过程           |

## 从识音到知音

这次项目带来的启发，不只是“机器能给音乐打分”。

A1 证明了预训练音乐表征能够有效学习专家对质量的共同判断；A3a 个性化实验则表明，**评价标准本身可以成为学习目标**。在未来，除了面向个人口味的评分器，我们也希望探索面向秦腔等特定音乐文化的评价模型：由熟悉该风格的听众或专家提供标准，让模型学习对应的审美关注点。

下一步最重要的实验，是以**整首歌曲**为单位隔离训练与测试，检验模型在新歌手、新曲目上的表现，并进一步区分风格偏好与歌手或音源特征。

**众耳知其妙，一心知其好。**

## 致谢、来源与引用

本项目是基于 [MuQ-Eval](https://github.com/dgtql/MuQ-Eval) 的课程复现及个性化扩展，不是原论文作者的官方实现。感谢原论文及 MuQ、MusicEval 相关工作的作者提供开放研究基础。

- **MuQ-Eval** · [论文](https://arxiv.org/abs/2603.22677) · [GitHub](https://github.com/dgtql/MuQ-Eval)
- **MuQ** · [论文](https://arxiv.org/abs/2501.01108) · [GitHub](https://github.com/tencent-ailab/MuQ)
- **MusicEval** · [论文](https://arxiv.org/abs/2501.10811)
- **LoRA** · [论文](https://arxiv.org/abs/2106.09685)

代码沿用上游 [MIT License](LICENSE)。原始数据、模型权重和音乐录音各自适用相应的许可及版权条件；本仓库不分发个人音乐文件，也不提供受保护音频的解密方法。具体来源、修改和许可边界请见 [docs/UPSTREAM.md](docs/UPSTREAM.md)。