# 上传范围与 Git 规则

公开内容采用 `.gitignore` 与 repository-policy.json 双白名单；提交检查读取 index 中的策略与文件，不用工作树内容冒充将发布的内容。

允许：本项目源码、合成隔离测试、requirements、experiments 设计 JSON、小型工程工具与来源哈希、协作模板和明确列出的公开文档。controller 是既有控制实验；不包含 SLAM 或 UE 源码。

禁止：output/tmp/inputs/research、模型权重、原始采集、游戏截图、真实轨迹、日志、账户配置、凭据、论文全文、未知授权资产及 UE Content。只保留本地不等于删除或不再有用。

新增共享目录需同步白名单与检查规则，经 PR 审查；不使用 git add -f 绕过规则。不增加未经确认的许可证，不共享账户。

提交前先显式暂存路径，运行 `python scripts/check_repository.py --staged-only`、`git diff --cached --check` 和 `git diff --cached`；检查器只输出可疑类别，不回显秘密。它不是完整的泄漏检测器。

GitHub 目标：main 通过 PR、相关 CI 全通过、分支需保持更新，禁止强推和删除，使用 squash merge；单维护者不要求自己批准，管理员保留恢复能力。规则必须在服务端实际配置并回读，不能仅凭本文宣称生效。
