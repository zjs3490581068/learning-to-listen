# 来源、许可与引用

## 上游与修改

- 原项目：[dgtql/MuQ-Eval](https://github.com/dgtql/MuQ-Eval)。
- 提取基准提交：`990325ba8512a24b04ad2bb58149c02a494da975`。
- 原论文：*MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music Generation Evaluation*，arXiv:2603.22677。
- 发布版包含该基准之上的本地实验修改，不是上游提交的原样复制，也不声称由原作者维护。

保留网络结构与参数名、MuQ 编码器包装、池化、预测头、损失、训练和统计代码的必要部分。
新增/整理的部分包括真实 TTM 数据协议、最佳检查点评估一致性、课堂退化与网页、个人音乐切片、
实际最后四层注意力 LoRA、个人偏好入口、便携路径/设备配置、回归测试与文档。
移除了无关基线及其他实验入口。科学协议变化在对应实验文档中明确说明，不能把个人偏好扩展当作论文主结果。

## 许可边界

`LICENSE` 保留上游 MIT 原文及原版权声明，不捏造上游未提供的作者归属。
此许可证针对代码，不能推导出数据、歌曲或预训练模型也可任意使用。

本机固定版本的模型/数据卡记录：

| 资产 | 所读取来源的许可说明 | 发布策略 |
|---|---|---|
| MuQ 基础模型 | 模型卡区分代码 MIT 与权重 CC-BY-NC-4.0 | 不打包权重，使用者单独确认许可 |
| MusicEval | 数据卡标注 CC-BY-NC-4.0 | 不打包数据，使用者从原来源获取 |
| 作者 A3a | 其模型卡元数据标注 MIT | 不打包，仍须考虑基础模型及其他相关条件 |
| 十二首个人歌曲 | 本项目不授予音乐使用或传播权 | 不打包，使用者自行合法取得 |

以上是固定来源材料的记录，不是对组合产物的法律授权或商业可用性判断。
下载器记录固定 revision，使用者还应查阅对应来源的完整条款；新下载的不同版本不能自动继承这里的记录。
同样不能因为仅有 10 秒片段，就假定可以公开分发生成的试听包。

## 引用原论文

沿用原仓库提供的 BibTeX：

```bibtex
@article{zhu2026muqeval,
  title={MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music Generation Evaluation},
  author={Zhu, Di and Chen, Zihan and Li, Zixuan},
  journal={arXiv preprint arXiv:2603.22677},
  year={2026}
}
```

发布者可在正式发布时补充自己的仓库地址与贡献者信息；本次整理不擅自创建远端或填写个人身份。
