# 开发与复现

推荐 Python 3.12。离线控制器依赖在 requirements/controller.txt；requirements/dev.txt 仅引用该文件。完整控制器以 Windows 为准，Linux CI 只运行不依赖 Windows API 的工程测试。

官方 GameScaling 客户端、权重、GPU 推理运行时独立配置，不随仓库提供。当前控制器仍保留历史 WSL/上游路径默认值，集中在 backends.py、wsl_bridge.py、evaluate_saved.py；policy_worker.py 的规则仍针对原神短距离探索，不能当成通用任务 API。迁移不自动启动或重新配置任何服务。

训练使用单独的 `requirements/training.txt` 和项目专用 venv，入口为 `python -m scripts.training.data`、`synthetic`、`evaluate`、`saved_probe`、`train`，详见 [training.md](training.md)。现有共享推理环境保持只读，新版 Transformers/PEFT 在训练 venv 中覆盖；不在该 overlay 中运行旧版 vLLM。按 research-plan 固定基线、数据许可、划分与预算；真实 torch/CUDA/驱动/模型文件哈希记录在本地运行清单，控制器依赖不能当训练锁文件。

自有玩具任务提供 `collect_headless`、`feedback` 和 RGB `rollout` 入口，见[环境反馈与适配](feedback.md)。教师引擎真值只用于审计/采集，模型回合只接收 RGB、固定规则和实际历史；推理期间暂停，不能把本地动作窗口称为在线实时游戏验收。

`bandit_train` 使用相同 RGB 首键 logits 比较专家分类监督和实际一步奖励目标，复用 LoRA/检查点/互斥包装器；方法、固定预算、真实训练和冻结失败见[反馈学习对照](feedback-learning.md)。奖励分支不读取教师动作作目标，但两组共享专家来源的 reset 语料，不能省略采集成本或称为顺序 RL。

现有回环端点和模型身份校验保留。不要为方便接入开放公网或去掉 allowlist；云端脚本的 trust-remote-code 需要运行前审查。游戏输入不是这次工程拆分授权的内容。

运行 README 中的离线测试与工程检查。依赖官方 gamescaling 包的测试在缺失时明确 skip；Windows 真实输入、截图、服务、模型和训练均需独立现场验收。

## 工程配置与共享工具

`python scripts/validate_project.py` 校验设计配置、评测划分及特权信息声明，不读取真实数据，不检查实际模型行为。
`python scripts/sync_engineering.py` 检查本仓库的共享文件与锁定哈希一致；更新流程见 shared-engineering.md。
`python scripts/check_repository.py` 只检查 Git 暂存区实际内容。全新仓库应在暂存源码后运行；正常克隆已存在完整 index。

这些检查都不启动训练、GPU、云资源、UE、游戏输入或模型下载。实验输出留在忽略目录 output/，不得把旧工程的绝对路径当本机事实。

本地蒙德城区素材接入与独立 UE 副本见 [场景接入说明](local-scenes.md)。
