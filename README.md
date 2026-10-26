# GameScaling Agent Lab

**目标：GameScaling 模型训练、多模态输入与低延迟决策。** 本项目与 [sitp-ue5-spatial-lab](https://github.com/Ziran-Ziran/sitp-ue5-spatial-lab) 独立开发、独立评测、独立发布，共享小型工程工具，不共享运行时或必需模型。

## 研究边界与当前能力

- **已有实现：**截图到官方策略的桥接、动作过滤、有限会话、执行历史反馈；新增可审计 RGB 轨迹转换、保存图评测、PEFT LoRA 训练入口、优化器/RNG 检查点恢复及重任务互斥限时包装器。
- **已验证范围：**离线合同与控制器回归、CPU 恢复一致性和同硬件确定性 4 步 GPU 恢复；本地官方 0.8B 已完成合成冒烟及自有玩具环境真实专家记录上的 128 步 LoRA。冻结实际回合中 reach 2/4→4/4、hazard 0/4→4/4、keydoor 0/4→3/4；规则简单且推理期间暂停，详见[反馈与适配结果](docs/feedback.md)。
- **反馈学习：**原始 0.8B 上已实际完成同初始化/同 128 更新预算的首键 SFT 与一步奖励学习。冻结主评测两者均 reach 16/16、hazard 12/16；钥匙门迁移为 6/8 与 7/8，且共同出现四个原模型成功后失败的 hazard 样本，见[方法、成本和失败](docs/feedback-learning.md)。
- **尚待验证：**商业游戏任务收益、原能力保持、广泛跨游戏泛化、顺序强化学习、官方模型额外模态理解及低延迟调度；受控玩具结果不能替代这些结论。

长期愿景是能在持续运行的陌生游戏中有效行动的通用玩家。当前研究重点是训练方法、额外模态的表示与融合，以及任务能力—响应延迟—算力成本的权衡，不依赖 SITP、SLAM 或 UE5。

## 离线开始

推荐 Python 3.12；先确认本机 Git、Python 路径和版本。只克隆本仓库即可运行以下检查，不需要兄弟仓库、GPU、游戏、模型或 UE。

```powershell
git clone https://github.com/Ziran-Ziran/gamescaling-agent-lab.git
cd gamescaling-agent-lab
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v

.venv\Scripts\python.exe scripts/validate_project.py
.venv\Scripts\python.exe scripts/sync_engineering.py
.venv\Scripts\python.exe scripts/check_repository.py
git diff --check
```

Linux/macOS 将解释器路径换成 `.venv/bin/python`。完整控制器测试以 Windows 为准；缺少官方 gamescaling 包时部分集成测试会跳过，不能据此宣称模型接口已验收。

`experiments/*.json` 是**设计阶段的研究配置**，尚未绑定数据或训练/导航执行器；`validate_project.py` 只检查配置与数据边界，不启动实验。

独立可运行的训练/评测 CLI 与这些设计配置分开：见[离线轨迹、RGB 评测与 LoRA 训练](docs/training.md)。需要已有本地模型和项目专用训练环境，重任务先通过共用锁；没有训练授权与真实反馈时只运行明确标注的合成冒烟。

额外模态提供音频因果窗口/缺失掩码、小型 CPU 冒烟和[实际 0.8B 音频频谱双图 LoRA 实验](docs/audio-vision.md)。后者实际训练 128 步，但八种主评测条件均 8/16、动作全为 A，没有建立融合收益；它也不是原生波形或真实游戏音频理解。

后续[冻结视觉读出诊断](docs/vision-readout.md)已完成四组各200次 CPU 更新及2048个实际消融回合。联合 MLP 在两套对齐条件各32/32、交换音频后各0/32，单模态各16/32；这支持人工任务的联合表示可读性。该入口跳过原语言/动作解码器，不把新头结果当作 LoRA 已学会融合。

## 可选的本地场景

已提供[两项目本地场景接入](docs/local-scenes.md)：共用源素材登记，各自维护独立 UE 实验副本；素材与本机绑定不上传 Git，也不构成训练或导航已验收。

## 工程入口

- [架构与模块边界](docs/architecture.md)、[开发与平台限制](docs/development.md)
- [研究问题与实验设计](docs/research-plan.md)、[阶段路线图](docs/roadmap.md)
- [输入、时间、模态和评测协议](docs/data-contract.md)
- [固定策略推理执行与延迟对照](docs/latency.md)
- [自行生成的无界面任务与反馈账本](docs/feedback.md)
- [一步实际奖励学习与监督对照](docs/feedback-learning.md)
- [分拆来源与本地数据处理](docs/migration.md)、[共享工程工具维护](docs/shared-engineering.md)
- [贡献指南](CONTRIBUTING.md)、[发布范围](docs/repository-policy.md)、[授权说明](NOTICE.md)

原始混合项目保留为历史来源，本仓库不复制其私人日志、论文资料、资产、数据和模型；公开源码按固定提交导出。
