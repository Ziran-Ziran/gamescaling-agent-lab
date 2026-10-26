# 项目分拆与来源

2026-10-26 根据用户确定的独立研究方向创建本仓库：GameScaling 模型训练、多模态输入与低延迟决策。

- 来源：https://github.com/Ziran-Ziran/gamescaling-npc-lab
- 固定提交：`46e75f16af30d5eb52ec6833a6dff3b3703ddc61`，逐文件 source_sha256 见 engineering/provenance.json。
- 迁移内容：controller/、控制器 tests/ 和 requirements/controller.txt，以及经调整的协作基础设施。
- 历史 Git 数据不复制，避免将旧混合项目的全部领域历史带入独立仓库；来源提交可回溯。
- 已改文件：README、AGENTS、CONTRIBUTING、NOTICE、CI、发布规则及文档均针对新边界重写；共享仓库规则测试重构为无领域依赖。GameScaling controller/README.md 去除旧耦合描述。其余迁移源文件初始保持字节一致。

原始本地混合目录及 GitHub 仓库保持原样，其他聊天可以继续完成既有实验。旧输出、论文解读、UE 资产、二进制、WSL 安装和权重没有移动、删除或上传；新的克隆只含源码，不能直接复用旧绝对路径假定依赖就绪。

以后新增领域工作进入对应新仓库。仍在旧工程完成的改动需要显式按文件/补丁迁入，经测试和 PR 审查，不自动同步、不把旧目录作为运行时依赖。本轮固定快照之后的旧工程改动未自动包含在内。

两项目只共用版本化工程副本，见 shared-engineering.md；不共享模型、地图内存或强制数据协议。
