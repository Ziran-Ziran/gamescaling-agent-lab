# 小型工程工具共用

两仓库均可单独克隆运行，不用 submodule、符号链接或兄弟目录 PYTHONPATH。共享范围只包括工程检查脚本、它们的测试、编辑器格式、Git 属性、安全规范和通用 Issue/PR 模板；领域依赖、研究协议与 CI 不共用。

engineering/manifest.json 是版本与 SHA-256 清单；GameScaling Agent Lab 为维护来源，双方使用同一 bundle version。哈希用于发现漂移，不是外部签名或供应链信任证明。

```powershell
# 仅检查本地副本完整性
python scripts/sync_engineering.py
# 显式选择一份经过审查的源克隆，只检查差异
python scripts/sync_engineering.py --source-root <reviewed-source-clone>
# 确认源清单和本地旧副本无损后，显式同步
python scripts/sync_engineering.py --source-root <reviewed-source-clone> --apply
```

同步只操作内置固定文件列表；拒绝路径跳转、符号链接、损坏源哈希和未经清单更新的本地改动，预检全部通过后才写入。不会拉网络、修改发布策略或改领域源码。

维护者更新流程：在来源仓库 PR 修改共享文件与相应测试；将 bundle version 升版，按固定文件列表更新清单哈希；通过 CI 后，再在另一仓库显式同步并 PR。消费者可以暂缓升级，仍独立可运行。不得为让检查通过而删除文件或放宽敏感数据规则。
