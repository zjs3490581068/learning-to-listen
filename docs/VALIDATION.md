# 发布验证与 GitHub 步骤

## 验证边界

源码整理验收阶段不重新训练、不下载资产、不分发已有音乐与权重；Git 提交与推送是验收后单独执行的发布步骤。
测试在现有实验环境运行，不等价于已在所有平台完成全新安装；GitHub Actions 工作流随代码提供，但尚未在远端触发。
历史实验结果见各实验文档，不把整理时的短验证当作重新复现完整实验。

默认离线测试：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
python scripts/check_release.py --check-help
```

默认测试不需要真实数据或权重，不联网。真实数据测试显式启用：

```bash
MUQEVAL_INTEGRATION_PATHS=configs/paths.local.yaml python -m unittest discover -s tests -v
```

这两项集成测试需要已准备好的官方数据、原始标注及 A1 划分清单；不是自动下载步骤。
不具备条件时显示 skipped，而不是伪造通过。

## 检查内容

- 真实系统 ID、非法身份拒绝、五折可重复/覆盖/互斥/均衡、预测顺序、检查点协议。
- 噪声 SNR、低通频响、削波阈值、嵌套断续、长度、PCM-24 往返及共享增益。
- 音频真实字节完整解码、能量切片、不足数量拒绝、展示泄漏、偏好期望值、冻结统计、验证缓存可反向。
- 路径覆盖、不同工作目录、无 Git、可见 GPU、原/自定义音乐配置、缓存迁移与数据内容校验。
- 源码 AST、文档内部链接、所有入口帮助命令、禁止私有路径、资产或凭据文件进入发布目录。
- 网页生成与可选浏览器检查用合成测试信号，不将它们当作真实课堂音频或主观听审。

## 本次实际执行结果（2026-11-18）

| 项目 | 状态与范围 |
|---|---|
| 离线单元及网页测试 | 58 项中 56 项通过，2 项真实资产检查按默认规则跳过；没有下载 |
| 两个真实数据测试 | 另行只读运行并通过，直接读取已有 Arrow 缓存，不重建大缓存 |
| A1 音频内容身份 | 1,923 条训练音频逐条计算并复核 SHA-256；未向原实验写入新清单 |
| 个人数据检查 | 已核对原 150 条及 5 个展示 WAV 的身份、摘要与隔离；十二首录音重新确定切片，与历史全部相同 |
| A1 兼容性 | 发布版严格加载原 fold0，合成 10 秒输入与原仓库输出最大差 0 |
| 作者 A3a 兼容性 | 严格加载，合成输入与原仓库输出最大差 0 |
| 新 LoRA 初始化 | 新增后输出最大差 0，可训练参数 1,445,638 |
| 单次诊断反向 | LoRA 32、池化 4、MI 头 6 个梯度张量通过审计，冻结参数无梯度；优化器更新为 0 |
| 两个离线网页 | 使用合成 WAV 生成并运行 Chromium，1440/768/390 宽视口通过，播放、切换、显隐正常，无外部请求；额外覆盖自定义展示数量 |
| 全新环境安装/下载 | 本次没有执行，使用既有环境及本地资产 |
| 重新训练/远端 CI/推送 | 本次没有执行 |
| 主观听审 | 本次没有执行 |

短模型检查 batch=1、bf16、GPU 7、6 GiB 分配器上限，观测峰值约 1.549 GiB；
这是合成输入的兼容性检查，不是 batch=16 的完整训练资源测量。

可对自己的本地模型重做严格加载及零增量诊断：

```bash
python scripts/verify_models.py --paths configs/paths.local.yaml --gpu 0 --backward
```

可加 `--a1-checkpoint` 检查 A1，或 `--reference-repo` 对比可信的原始源码目录。
`--backward` 只做一次诊断反向，没有优化器更新，不是正式微调。
单次检查不能证明所有输入或不同硬件逐位一致。

## 发布到 GitHub

本项目仓库地址为 `https://github.com/zjs3490581068/learning-to-listen.git`。
以下命令用于首次发布尚未初始化 Git 的源码目录；已克隆仓库不必重新初始化或添加远端。
发布前先运行发布检查并审阅文件清单；发布到自己的仓库时替换远端地址，不在地址中放 token：

```bash
git init -b main
git add README.md README.en.md LICENSE .gitignore requirements.txt requirements-browser.txt train.py generate_predictions.py run_evaluation.py src scripts configs templates tests docs .github
git status --short
git diff --cached --stat
git commit -m "Publish MuQ-Eval reproduction and classroom experiments"
git remote add origin https://github.com/zjs3490581068/learning-to-listen.git
git push -u origin main
```

发布检查不能代替人工复核。不要 `git add` 数据盘目录，不上传完整检查点、音乐或本机 resolved 配置。
生成试听压缩包也只供使用者在其授权范围内使用，不能随源码一并公开。
检查脚本面向纯源码目录，直接扫描磁盘，不把 `.gitignore` 当作排除依据。
本地运行产生 data、模型、缓存或字节码后，应在干净源码副本检查发布，或一开始就使用外部存储路径。
