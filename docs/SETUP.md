# 安装、资产与可移植运行

## 环境与依赖

推荐 Linux、Python 3.11；参考环境是 Python 3.11.18、PyTorch 2.5.1+cu121、torchaudio 2.5.1、
transformers 4.44.2、peft 0.15.2、datasets 3.6.0、NumPy 1.26.4、SciPy 1.14.1、scikit-learn 1.5.2、muq 0.1.0。
安装命令见主 README。`requirements.txt` 是已用版本的运行依赖列表，不是跨平台二进制锁文件；CUDA 驱动仍需与所选 PyTorch 发行包兼容。
没有纳入 FAD、SongEval、MERT 或 W&B 工作流。Pillow/Matplotlib 用于生成波形图和结果曲线。

可选浏览器检查依赖 `requirements-browser.txt`。Chromium 需要另行安装，或者设置 `PLAYWRIGHT_BROWSERS_PATH` 指向已有安装。
测试不会自动下载浏览器。若主机缺少浏览器系统依赖，先完成其他检查并如实标注浏览器未验收，不在共享服务器安装系统包。

## 必需资产与固定版本

| 资产 | 官方 Hugging Face 仓库 | 本实验固定 revision |
|---|---|---|
| MusicEval | `BAAI/MusicEval` | `20977b88b95a944bf4d55545652958c62af0d35f` |
| MuQ 基础模型 | `OpenMuQ/MuQ-large-msd-iter` | `0562a57814f6f8bbd9fdea0a25921a2fce1a841a` |
| 作者 A3a | `zhudi2825/MuQ-Eval` | `fb74cb55e50e0d008b730540fc533473f88c330c` |

`MuQ-Eval-A3a` 只是本地目录别名，不是这里使用的 Hugging Face 仓库 ID。
A1 从基础模型重新训练池化和头，不需要下载作者 A1。

默认目录布局：

```text
data/
  MusicEval/
    MusicEval-full.zip
    data/                         # 官方 train/validation/test parquet
  MusicEval-original-metadata/     # prepare_assets 提取，包含 sets/ 和提示文本表
  personal_music/                 # 使用者自行提供的 Ogg
models/
  MuQ-large-msd-iter/
    config.json
    model.safetensors
  MuQ-Eval-A3a/
    model_state_dict.pt
outputs/
logs/
.cache/
```

可通过下载脚本显式准备资产，也可以手动放到上述目录。不用修改源码中的路径。
下载脚本固定 revision、核对远端对象大小和可用的 LFS/Git 摘要，记录 SHA-256；已有文件校验失败时拒绝覆盖。
先尝试 `hf-mirror.com`，再尝试 Hugging Face 官方地址，全程直连，不使用代理或 token。直连失败时停止，提示手动准备。
下载脚本依赖 `curl`；已准备资产后的离线实验不依赖它。

两个重要历史校验值：

```text
MusicEval-full.zip
b9ced113cede47ab71ec8a9f7d41fb9a2901994d63a4adb67d10f9e659336aa9
作者 A3a model_state_dict.pt
aaec85582cbea0d726d0511f571c58435c251d08e1c3231bfc722343e6a30868
```

`prepare_assets.py` 不联网，只提取 ZIP 中的原始元数据，校验三套 split 的文件名、真实系统 ID、MOS 和数量，并检查基础模型文件存在。
官方 train/validation/test 分别为 1,923/412/413 条；训练只用 train。test 实际为 30 系统，不强加训练集的 31 系统断言。
检查点内容严格加载在实际模型入口执行；准备脚本本身不进行 GPU 推理或训练。

## 配置规则

每个主要入口都有 `--config` 和 `--paths`。合并顺序是默认存储布局、实验配置、存储覆盖。
`--paths` 只能包含 `storage`，其合法键与 `configs/paths.example.yaml` 相同。
这将实验方法参数与文件位置分离，改盘符或挂载目录不必改网络结构及超参数。

- 配置文件参数及存储中的相对路径以源码根目录解析。
- 音频清单中的 `pattern` 相对 `music_root` 匹配，不能逃出目录。
- 命令行直接给出的音频、结果目录和输出文件路径按调用者当前目录解析；跨目录调用时推荐用绝对路径。
- 不要求源码目录含 `.git`；来源记录会回退到固定上游提交说明。
- 产物内的 WAV/图片路径相对实验目录，个人源路径相对 `music_root`；模型与数据身份不依赖这些绝对挂载位置。
- `resolved_config.yaml` 是本机运行记录，含解析后的实际路径，不应直接加入公开仓库。

## GPU、日志与恢复

CUDA bf16 是参考训练/推理配置。主要单卡入口先选设备，再初始化 CUDA；不能用一个设备生成 MuQ 的旋转位置缓存后再迁移到另一个设备训练。
运行入口显示命令、环境、输出和 GPU，单卡选择可用 `--gpu`，A1 调度用 `--gpus`。
默认单进程；增加并行必须显式设置 A1 `--max-parallel`，上限 4，每卡一个进程。
只在调用者允许的 `CUDA_VISIBLE_DEVICES` 范围内选择卡，空字符串表示禁用 GPU，不会被忽略。

A1 每折 worker 日志写入 `storage.logs_root`。其余入口输出标准输出/错误，可由使用者重定向至日志目录。
非有限损失、异常梯度、OOM 均停止当前运行，不改变学习率、batch 或精度自动重试。
不要认为 6 GiB 分配器上限覆盖全部设备开销，也不要把单次空闲检查当作显存预留保证。

A1 `--resume` 只复用已通过完整验收的折，不恢复中断优化器；不完整折不自动删除。
个人训练拒绝覆盖已有模型，暂无优化器断点续训。
课堂 `--reuse-scores` 严格核对身份后复用完整候选结果，不是复用任意 CSV。

## 常见问题

| 情况 | 处理方式 |
|---|---|
| 离线找不到模型/数据 | 核对固定版本、目录和 `--paths`；先显式准备资产，不自动开放联网 |
| 个性化歌曲匹配为 0 或多条 | 修改音乐清单的 pattern，使每首恰好匹配一个 Ogg；不要只改扩展名 |
| 不能选满切片 | 增加可用录音时长或明确设计新的自定义协议，不自动降低能量/隔离标准 |
| 期望有人声但切到伴奏 | 本方案只有能量与首尾过滤，没有人声识别；试听切片清单并承认局限 |
| 作者模型严格加载失败 | 检查仓库 ID、revision、文件摘要和依赖版本，不使用 `strict=False` 绕过 |
| 复制源码后旧产物身份不符 | 旧 A1 清单缺少新逐音频摘要，旧个人格式含机器路径；发布版不通过放松验证自动迁移它们 |
| 网页音频或图缺失 | 保留完整生成目录及相对资源，而非只携带 HTML |
| 课堂分数没有达到目标 | 保留真实结果；不改分、不反复针对展示集调参或替换固定展示例 |

所有训练前都应核对配置、输出位置和可用设备。不要让新实验覆盖旧结果。
