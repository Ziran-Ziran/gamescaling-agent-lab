# 无界面任务与环境反馈

`scripts.training.headless` 提供本项目自行生成的 7×7 点玩家网格环境，只用 Pillow 几何图形，不调用桌面、真实客户端、UE、SITP 或 SLAM。它是受控玩具任务，不能代表商业游戏能力。当前已实现反馈和专家采集，尚未据此训练策略或运行 RL。

## 可观察输入与训练教师

蓝圆为玩家、绿色为目标、红色为致命区域、深色为墙。钥匙门规则中，黄色钥匙必须先收集，紫门始终覆盖绿色目标；左上角黄色指示器显示持有钥匙。底部条显示剩余动作预算。所有影响路径选择的网格、物体与库存均在 RGB 中显示。

三种规则共用引擎：`headless-reach-v1` 仅到达目标；`headless-hazard-v1` 增加致命区域；`headless-keydoor-v1` 增加钥匙和封闭目标门。默认课程从相邻目标开始，只有四个方向，环境复杂度很低。不能将这些共享引擎的规则变体称为陌生商业游戏泛化。

BFS 专家读取引擎状态，明确标为 `teacher_inputs: privileged_engine_state_for_BFS`。原始账本含训练期特权真值，用于审计物理与奖励；转换后的学生数据只含 RGB、固定规则提示及实际已执行动作历史，不含坐标、未来状态、奖励或教师路径。模型评测必须另行运行，不以专家成功率代替。

## 动作、时钟与奖励

坐标是网格单位，x 向右、y 向下；玩家是无碰撞半径的点，位置可为小数，障碍按所在网格判定。输入只接受 WASD，鼠标与滚轮为零；相反按键抵消，斜向速度归一化。200ms 动作块含六个子步骤，整块持有一个方向移动一格。每步先按 x 再按 y 检查墙和未解锁门，接触致命区域后本块余下物理冻结，终止在块末返回。

CLI 使用真实 `perf_counter_ns` 和 sleep 执行约 200ms 墙钟动作窗口，允许 ±20ms 完成容差。单调时间记录采样、动作开始/结束和反馈返回。物理只在六个虚拟子步骤推进，**推理期间环境暂停**；这不是持续运行的实时游戏，也没有逐子步骤时延测量。注入测试时钟的单测数据强制标为 `synthetic`，不能冒充真实计时采集。

`headless-grid-feedback-v1` 每块奖励 −0.01，到达目标另加 1，接触致命区域另减 1；拿钥匙本身无奖励。目标/死亡使用 `terminated`，耗尽步数使用 `truncated`；终止优先于时间上限。同一块不能同时成功与死亡。结束后禁止继续执行。

每帧绑定实际完整动作、其哈希、前后状态哈希、奖励事件、终止原因和时钟。审计复算奖励、无 sleep 重放六步物理、验证连续状态链，并从前状态重新渲染图像对比实际 PNG 的解码像素哈希。动作 hash、位置上界或孤立奖励值不能单独替代这种对齐检查。该本地一致性审计并非外部可信采集签名。

## 采集与独立复核

在项目隔离环境安装已有开发依赖后：

```bash
python -m scripts.training.collect_headless --output output/headless-expert \
  --train 64 --validation 8 --test 8
python -m scripts.training.feedback --source output/headless-expert/episodes.jsonl \
  --data-root output/headless-expert --output output/headless-expert/reaudit.json
python -m unittest discover -s tests -p 'test_headless*.py' -v
```

输出目录必须新建。采集只写完整并通过反馈审计的 episode；失败时保留中断文件，不能把它们并入有效清单。默认种子为训练 reach 0–63、验证 hazard 1000–1007、测试 keydoor 2000–2007。游戏按规则划分，episode 不混组，转换器还检查跨组解码像素重复。输出含原始轨迹、PNG、奖励审计、来源哈希、划分及可直接交给现有 [LoRA 训练器](training.md) 的 `converted/`。

2026-10-27 的首轮实际 WSL 采集：80 个专家回合、96 个动作窗口，80 个达到目标，反馈采集约 29.0 秒；转换约 1.1 秒另外计。这里只证明该小环境的专家采集和合同一致性，**尚无模型任务成功或强化学习结论**。媒体、轨迹、检查点、原始结果和本机路径均留在忽略目录。
