# GameScaling Agent Lab

**目标：GameScaling 模型训练、多模态输入与低延迟决策。** 本项目与 [sitp-ue5-spatial-lab](https://github.com/Ziran-Ziran/sitp-ue5-spatial-lab) 独立开发、独立评测、独立发布，共享小型工程工具，不共享运行时或必需模型。

## 研究边界与当前能力

- **已有实现：**截图到官方策略的桥接、动作过滤、有限会话、执行历史反馈、保存图评估及 Windows 离线回归测试。
- **尚待实现与验证：**SFT/LoRA 训练器、在线强化学习、额外模态编码器及融合、低延迟策略调度；目前仍是 RGB 推理基础设施，未证明任意游戏通用能力。
- **本轮完成：**源码分拆、来源清单、独立 CI、发布白名单、研究协议和配置检查；不代表新增算法、训练或游戏现场验收完成。

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

## 可选的本地场景

已提供[两项目本地场景接入](docs/local-scenes.md)：共用源素材登记，各自维护独立 UE 实验副本；素材与本机绑定不上传 Git，也不构成训练或导航已验收。

## 工程入口

- [架构与模块边界](docs/architecture.md)、[开发与平台限制](docs/development.md)
- [研究问题与实验设计](docs/research-plan.md)、[阶段路线图](docs/roadmap.md)
- [输入、时间、模态和评测协议](docs/data-contract.md)
- [分拆来源与本地数据处理](docs/migration.md)、[共享工程工具维护](docs/shared-engineering.md)
- [贡献指南](CONTRIBUTING.md)、[发布范围](docs/repository-policy.md)、[授权说明](NOTICE.md)

原始混合项目保留为历史来源，本仓库不复制其私人日志、论文资料、资产、数据和模型；公开源码按固定提交导出。
