# 贡献指南

先读 README、docs/roadmap.md 与 docs/repository-policy.md。先通过 Issue 明确研究假设、输入输出和验收，再为一项工作开一个分支与 PR。

```powershell
git switch -c codex/your-topic
# 修改后运行 README 的离线测试与工程检查
git add <明确审查过的源码路径>
python scripts/check_repository.py --staged-only
git diff --cached --check
git diff --cached
git commit -m "feat(scope): describe the change"
git push -u origin codex/your-topic
```

main 通过 PR 集成，使用 squash merge；不强推或删除 main。单维护者阶段不要求自己批准；第二位维护者加入后再设置至少一名审查者。协作者通过 GitHub 权限或 Fork/PR 参与，不共享账户与密钥。

行为变化补充有意义的回归测试；PR 写清假设、固定变量、结果和未测范围。不得按评测真值挑选参数、隐藏失败或把跳过当通过。

开源授权尚未确定，原创内容保留权利；新增第三方材料前先确认来源。真实输入必须有独立授权及急停、失焦、超时和按键释放验收；CI 只做离线测试。新配置使用参数或明确支持的环境变量，不添加机器用户名、服务器地址和凭据。保留现有实验代码中的机器依赖时，在开发指南写明限制。
