# 带真实任务回报的 RGB 策略保留评测

静态 Doom 图上的动作漂移没有实际任务回报，不能衡量遗忘或保持。本实验固定五个已有策略，在同一批新种子的自有 RGB 玩具环境中执行动作、核对奖励，逐回合比较原模型成功后的退化。它不训练或选择新检查点。

## 冻结模型与边界

共同基座为 `THUSI-Lab/GameScaling-0.8B-S3` revision `1b89640dcf2ef51fb53ad9338a0725e9622dd200`；权重 SHA256 `d553c0796833691e32277a937db649eaac39bae9a0028a63c964c12a261c2dfa`，完整模型指纹 `ed679f4172207776e6aa8e63915c3645721f62a38aef7c1ce3fd82a96241c87c`。

| 策略 | 既有训练 | adapter 权重 SHA256 |
|---|---|---|
| original | 无新增训练 | 无 |
| text-sft128 | [完整动作文本监督](feedback.md)，128更新 | `cc8a220cb138a05971e7d593547c2a1553ecfe0f3fc42a740d773e22b57aa174` |
| class-sft128 | [首键分类监督](feedback-learning.md)，128更新 | `7d727dffcc16b1ec9d6c46b651fe12ac7f14259ff457bb4181dd64bc6f00311d` |
| bandit128 | [实际一步奖励学习](feedback-learning.md)，128更新 | `749b147d9eb0c95913b22a569afb1c1c920495e58d89abb40cb2771a1e10d31d` |
| audio-lora128 | [双图音频派生视觉监督](audio-vision.md)，128更新 | `bfa8ec85542d284f3ef123b64aa3dfd18a09918e2581884d2992c76d6cf2be56` |

`retention_compare.py` 也固定 adapter 配置文件的摘要与完整指纹，不能只验证权重文件。数据、目标、训练种子、标签/交互成本不同，因此这是描述性的策略保留矩阵，不是等预算训练方法优劣对照。

五组都使用 RGB、相同可见规则提示、零历史、greedy 五类首键预测和安全整窗口动作模板。audio-lora128 的 RGB-only 是显式的模态/观测干预，不是它的 aligned 双图输入。独立冻结视觉读出头跳过原解码器，不属于这些原策略 adapter，不混入此矩阵。

硬件为 RTX 5070 Ti Laptop，bf16，Torch 2.10.0+cu128、Transformers 5.14.1、PEFT 0.21.2；使用原先 Torch 回退实现。每个实际动作200ms，推理期间世界暂停，没有真实桌面或游戏键鼠输入。

## 样本与审计

先冻结全部种子、五个最终 adapter 和所有预算，再开始任何新回合。主评测为 reach 30000–30015、hazard 31000–31015，各16回合、一步上限；次评测为 keydoor 32000–32007，8回合、八步上限。每策略40回合，共200个计划实际回合，最多480次动作，不能按结果剔除种子或改变上限。规则共享同一个7×7玩具引擎，新的调色与种子不等于新的独立商业游戏。

`rollout_compare` 重查完整状态、当前推理源码摘要、模型/环境、默认可见提示、固定 reset、逐帧 RGB/动作/状态/奖励和全部结果。首键原始 logits 必须产生记录的 argmax、模板和实际执行动作；诊断 softmax 只允许跨平台 libm 的极小舍入差，原始 logits/选类/动作不作容差比较。实际执行、观测到动作开始/结束的时间必须与原始单调时钟相符。

`retention_compare` 在此基础上约束五组具体模型与套件协议，并逐回合比较初始 RGB、状态、提示、动作合同和其余运行参数。审计引擎真值不作为策略输入；完整原始账本、RGB、权重、日志和本机路径不上传。

除成功率、平均回报和动作数外，报告相对原模型的回报胜/平/负、原成功→适配失败及原失败→适配成功的具体回合。保留比例的分母是原模型成功回合数；若分母为0，明确返回 `null`/不具信息性，不能用两组都失败证明能力保持。

## 复现入口

根据当前设备设置已有本地模型、Python、共享锁及四个 adapter 目录。源模型不下载；每个 GPU 任务独立受锁和900秒上限保护，正常完成并审计后再启动下一项。保存原始失败/未完成状态，不能把未执行的计划当成实际结果。

```bash
# original 不传 --adapter；其余四组加 --adapter "$ADAPTER"，使用独立输出目录。
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 900 -- \
  $PY -m scripts.training.rollout --model "$MODEL" --decoder first-key \
  --games reach hazard --seed-start 30000 --episodes-per-game 16 \
  --max-steps 1 --history-len 0 --max-new-tokens 64 --output output/retention-original-primary
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 900 -- \
  $PY -m scripts.training.rollout --model "$MODEL" --decoder first-key \
  --games keydoor --seed-start 30000 --episodes-per-game 8 \
  --max-steps 8 --history-len 0 --max-new-tokens 64 --output output/retention-original-secondary
# 实际 keydoor 种子偏移为 +2000，即32000–32007。
```

五组都完成后，两套分别在 CPU 上审计，以下以主评测为例；次评测换 `--suite secondary` 和对应目录。比较入口不初始化模型/GPU。

```bash
$PY -m scripts.training.retention_compare --suite primary \
  --original output/retention-original-primary \
  --text-sft128 output/retention-text-sft128-primary \
  --class-sft128 output/retention-class-sft128-primary \
  --bandit128 output/retention-bandit128-primary \
  --audio-lora128 output/retention-audio-lora128-primary \
  --output output/retention-primary-comparison.json
# 通用的两组同协议比较也可独立使用：
$PY -m scripts.training.rollout_compare --before output/retention-original-primary \
  --after output/retention-text-sft128-primary --output output/retention-text-paired.json
```

## 冻结结果与退化

2026-10-27，五组两套全部完成，共200实际回合、386次实际动作，执行失败0；每个独立任务正常退出并审计后才进入下一项。共享锁曾被另一任务占用，队列等待后取得锁，没有删除或接管他人锁。所有默认提示、初始画面、实际动作、时钟、反馈和完整汇总均经 CPU 重审。

另对三个实际 TRAIN 语料（每个64行）重跑来源审计：主评测32张、次评测8张初始 RGB 均唯一，五策略的对应初始像素完全相同，与三个语料训练画面的精确像素重合全部为0。这只能排除精确图像重复，不能排除规则或布局相似。所有评测任务中的峰值 CUDA 分配至多1705.14MiB。

| 策略 | reach 成功/16；平均回报 | hazard 成功/16；平均回报 | keydoor 成功/8；平均回报 | keydoor 动作数；平均步数 |
|---|---:|---:|---:|---:|
| original | 8；0.4900 | 5；0.0525 | 0；−0.0800 | 64；8.00 |
| text-sft128 | 16；0.9900 | 16；0.9900 | 7；0.8325 | 34；4.25 |
| class-sft128 | 16；0.9900 | 12；0.4900 | 6；0.7050 | 36；4.50 |
| bandit128 | 16；0.9900 | 12；0.4900 | 8；0.9650 | 28；3.50 |
| audio-lora128（RGB 干预） | 7；0.4275 | 4；−0.0100 | 0；−0.0800 | 64；8.00 |

reach/hazard 每组各16次动作、每回合一步。相对同种子原模型的逐回合回报胜/平/负如下，未用平均分掩盖退化：

| 策略 | reach 胜/平/负 | hazard 胜/平/负 | keydoor 胜/平/负 |
|---|---:|---:|---:|
| text-sft128 | 8/8/0 | 11/5/0 | 7/1/0 |
| class-sft128 | 8/8/0 | 11/1/4 | 6/2/0 |
| bandit128 | 8/8/0 | 11/1/4 | 8/0/0 |
| audio-lora128（RGB 干预） | 3/9/4 | 7/1/8 | 0/8/0 |

| 策略 | reach 原成功保留；原失败→成功 | hazard 原成功保留；原失败→成功 |
|---|---:|---:|
| text-sft128 | 8/8；8 | 5/5；11 |
| class-sft128 | 8/8；8 | 1/5；11 |
| bandit128 | 8/8；8 | 1/5；11 |
| audio-lora128（RGB 干预） | 4/8；3 | 1/5；3 |

分类 SFT 和 bandit 的 hazard 平均回报都比原模型高0.4375，但各自仅保留1/5原成功回合；原成功→失败为 `31000, 31004, 31008, 31012`，均属于 `headless-hazard-v1`。这四个 reset 中原策略向 W 到达绿色目标，分类 SFT/bandit 向 S 进入红色危险区。音频 LoRA 同样损伤这四个 hazard 回合，但动作是 D、以一步上限结束；它另损伤 reach 的 `30000, 30004, 30008, 30012`。完整回合标识和新增成功列表都保存在 CPU 比较报告中。

text-sft128 在这两个单步套件中没有原成功退化，但这只覆盖13个原模型成功样本和自有简单规则。keydoor 的原成功分母为0，四个保留比例全部为 `null`、`retention_informative=false`；0/8对0/8不能证明音频 LoRA 保持了能力。bandit 的8/8、文本SFT的7/8和分类SFT的6/8也是单个训练种子、八个新 reset 的描述量，训练数据/目标/成本不同，不能据此排名方法。

本结果补充了有任务回报的玩具 RGB 保留证据，不能代替 Doom、商业游戏或广泛原能力的遗忘验收。尤其 audio-lora128 使用了不同于其双图训练的 RGB-only 观测条件；它说明该干预下的已测损伤，不单独定位训练目标、输入模态或内部机制的因果责任。
