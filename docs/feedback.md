# 无界面任务与环境反馈

`scripts.training.headless` 提供本项目自行生成的 7×7 点玩家网格环境，只用 Pillow 几何图形，不调用桌面、真实客户端、UE、SITP 或 SLAM。它是受控玩具任务，不能代表商业游戏能力。已完成实际 0.8B LoRA 训练及冻结 RGB 回合对照，未运行 RL。

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

2026-10-27 的首轮实际 WSL 采集：80 个专家回合、96 个动作窗口，80 个达到目标，反馈采集约 29.0 秒；转换约 1.1 秒另外计。专家成功只证明该小环境的采集和合同一致性，不能代替下文独立模型回合评测。媒体、轨迹、检查点、原始结果和本机路径均留在忽略目录。

## RGB 策略回合评测

`scripts.training.rollout` 复用本地 GameScaling RGB 后端。策略接口只构造固定规则提示、当前 RGB 与完整实际动作历史；它收不到环境对象、坐标、奖励、BFS 路径或终止真值。记录原始生成文本，严格格式失败时执行并计数零输入 fallback；格式有效后的鼠标/非 WASD 过滤另计。历史只追加环境验证已完成的实际动作。

```bash
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  "$PY" -m scripts.training.rollout --model "$MODEL" \
  --output output/headless-before --episodes-per-game 4 --seed-start 3000 \
  --max-steps 8 --history-len 0 --max-new-tokens 64
```

`PY`、`GPU_LOCK` 和 `MODEL` 的本机确认方式见[训练入口](training.md)；模型必须已有本地快照。适配后使用相同参数、新的输出目录并增加 `--adapter output/headless-lora/checkpoint-000128/adapter`。默认三个规则的起始种子分别为 3000、4000、5000，每组四回合、上限八步；全部计划回合必须完成，才能形成成功率汇总。若更改规则或预算，必须在比较前重新冻结。

本轮在看到冻结测试结果前固定 rank 8、lr 0.0001、seed 20261027、128 次总更新、零历史与确定性训练；仅用 reach 的 64 条真实专家执行记录训练。先运行四步测梯度/显存，再恢复到 128：

```bash
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 240 -- \
  "$PY" -m scripts.training.train --model "$MODEL" \
  --dataset output/headless-expert/converted --output output/headless-pilot \
  --steps 4 --save-every 4 --rank 8 --lr 0.0001 --seed 20261027 --history-len 0
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  "$PY" -m scripts.training.train --model "$MODEL" \
  --dataset output/headless-expert/converted --output output/headless-lora \
  --resume output/headless-pilot/checkpoint-000004 --steps 128 --save-every 32 \
  --rank 8 --lr 0.0001 --seed 20261027 --history-len 0
```

输出分开保存执行前的 `decisions.jsonl`、逐个已完成的 `transitions.jsonl`、开始/失败/结束的 `episode-status.jsonl`、完整回合的 `episodes.jsonl`。`run-status.json` 原子更新计划/完成/失败/活动回合数；推理异常保留已执行证据，异常退出不生成完整结果。硬中断可能留下 running 状态，不能当作完成；执行前拟动作也不能当作实际动作。

完整运行的 `summary.json` 按规则分别给出成功率、回报、步数、格式/fallback/过滤比例与时延，重复审计完整反馈。时延分项包含渲染、PNG 编码/写入/哈希、解码、处理器编码、设备传输、推理、过滤、实际执行。`policy_call` 包含处理与推理，不能与其子项重复相加；采样到执行结束不含采样前渲染，整圈时间还包含反馈检查。没有网络传输和排队的阶段填 null。首请求冷启动未剔除，P95 使用最近秩；小样本与暂停式环境不能推出在线尾延迟或过期率。

## 首轮固定预算结果与失败

2026-10-27，官方 `THUSI-Lab/GameScaling-0.8B-S3` 快照 `1b89640dcf2ef51fb53ad9338a0725e9622dd200`，单张 RTX 5070 Ti Laptop GPU，bf16、Transformers 5.14.1、PyTorch 2.10.0/cu128、PEFT 0.21.2。模型权重 SHA256 为 `d553c0796833691e32277a937db649eaac39bae9a0028a63c964c12a261c2dfa`。数据源 SHA256 为 `820b451a4512ab293ec6828ac5a693dd527a53da1867ea67d6de3e3ebf03ce7f`；源含训练 64、验证 8、测试 24 个实际完成的动作窗口。只有训练 reach 数据进入梯度更新。

确实执行了 4 + 124 次优化器更新，恢复前后身份一致，梯度有限且非零；峰值 CUDA 分配 3293.27MiB。四步试跑约 19.6 秒，恢复后 124 步约 97.0 秒，各自包含模型加载。首步/末步训练损失约 0.121785 / 0.00002442；这两个训练样本不同，不能当作同一样本的学习曲线。最终 adapter 权重 SHA256：`cc8a220cb138a05971e7d593547c2a1553ecfe0f3fc42a740d773e22b57aa174`。

前后各 12 个计划回合全部完成，无运行失败；相同模型、软件/硬件、生成参数、提示、初始画面、种子与步数预算已逐项核对。两组各有 12 张唯一初始画面，与全部专家记录无精确像素重复，但布局仍是同一个简单相邻目标课程。没有根据测试结果挑选检查点。

| 规则与种子 | 原模型成功 | 适配后成功 | 原平均回报 | 适配后平均回报 |
|---|---:|---:|---:|---:|
| reach 3000–3003，同训练规则新种子 | 2/4 | 4/4 | 0.4475 | 0.9900 |
| hazard 4000–4003，未用于训练的规则 | 0/4 | 4/4 | −0.0800 | 0.9900 |
| keydoor 5000–5003，未用于训练的规则 | 0/4 | 3/4 | −0.0800 | 0.7050 |

共执行 85 个原模型动作与 26 个适配动作，所有输出格式严格有效，fallback 和安全过滤均为零。每组样本仅四回合；规则共用引擎和主要视觉编码，不代表独立商业游戏或广泛跨游戏泛化。reach/hazard 适配后均一步到达，目标本来就相邻。

适配后 keydoor seed 5001 失败：取得钥匙后绕向左上角，最后在墙边连续向上，八步上限结束；没有隐藏失败或修改测试集。静态保存帧辅助评测每组前四帧，动作匹配由训练 0/4、验证 0/4、测试 1/4 变为 4/4、4/4、2/4，不能与上述实际回合成功率混用。原能力保持仍缺少带任务真值的独立游戏评测。相同官方 Doom 静态图/提示探针仍输出六组 LB，但鼠标 x 由原模型 +10 变为 −10；这是输出漂移证据，不是带任务真值的游戏遗忘或保持评测。
