# 设计配置

这些 JSON 只描述研究条件，`status: design`、`runner_implemented: false`、`dataset_bound: false` 表示尚无执行器或绑定数据。synthetic_fixture_* 是划分隔离检查用的标识，不代表已有游戏或场景数据。

`python scripts/validate_project.py` 校验字段、领域、模态声明、训练/评测信息边界及划分互斥；不会转换数据、训练或导航。实际实现执行器时，须另加明确的入口、数据来源与许可清单、依赖锁定、测试和实测记录，再通过 PR 修改状态。
