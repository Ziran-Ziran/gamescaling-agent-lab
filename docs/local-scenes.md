# 两项目共用的本地场景

提取的蒙德城区精细静态版可作为两条研究线的可选素材。场景不随 Git 分发，不是项目运行离线测试的依赖，也不把 SITP 与 GameScaling 重新接为一个运行时。

## 共用方式

维护一份本地素材目录和目录外的 catalog；两仓库的接入工具读取同一 catalog，各复制一份约 123 MiB 的 UE Content 到自己被忽略的 `local/scenes/<scene-id>/workspace/`。原始 FBX、贴图、摆放与来源证据仍共用源素材包，实验地图、材质修改和 UE 生成文件分别保存。

两边默认场景 ID 为 `mondstadt-city-7-1-0`，UE 5.8 工程为 `workspace/UE5/MondstadtCityDetailed.uproject`，地图为 `/Game/MondstadtCity/Mondstadt_City_Detailed`。默认接入精细城区版，不自动替换为早期远景简化版。

```powershell
# 任一仓库执行一次，显式选择本机素材和本地共享 catalog。
python scripts/local_scene.py register --catalog <outside-repository-catalog.json> --source-package <detailed-city-package>
# 分别在两个仓库根目录执行；各自建立独立副本。
python scripts/local_scene.py prepare --catalog <outside-repository-catalog.json>
# 在各自仓库检查绑定与副本完整性，不启动 UE。
python scripts/local_scene.py inspect
```

路径按本机显式登记，不把用户名或另一台电脑的路径写入公开配置；换电脑后重新登记。绑定信息在各仓库 `local/scenes/<scene-id>/binding.local.json`。`inspect` 输出当前本机工程路径和修改/缺失列表；修改列表允许保留研究编辑，缺失文件返回非零。

## 完整性与修改

工具校验源 `delivery_checksums.json` 覆盖的 UE 项目、地图和全部静态资产；源清单变化时要求新版本 ID。已存在的场景 ID 不静默改绑，已有工作副本不覆盖本地修改。只复制独立文件，不创建硬链接或符号链接；不复制 Saved、Intermediate、缓存、原项目的私人配置或 Android 文件服务设置。副本中生成只指定默认地图的干净 DefaultEngine.ini。

catalog 和 binding 含本机路径，素材、UE 工作副本以及这些登记文件一律留在本地。发布检查明确拒绝 `local/`，不能用 git add -f 上传。CI 使用合成小文件测试，不需要或下载原神素材。

## 两边分别能做什么

- SITP：作为后续语义识别、可通行表示与路径规划实验的场景基础；采集组件、语义标签、代理与导航闭环需独立接入。
- GameScaling：作为后续交互、轨迹采集、训练或多模态消融的可选环境素材；目前没有自动生成任务、奖励、玩家控制或引擎状态适配器。

当前素材是静态城区重建，不等同于可完整游玩的原神。源工程曾做保存/重载结构检查，但原版着色器、动态对象、任务与战斗逻辑不完整；碰撞、NavMesh、Play、UE 画面和数据采集均需分别验收。复制及哈希检查不等于这些能力已通过。本次接入不会启动 UE、游戏输入、模型或训练。
