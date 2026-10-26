# 一步环境奖励学习与监督对照

本实验使用自有网格的实际动作与奖励，比较相同 RGB 观测上的专家监督和 contextual bandit。每回合只决策一次，动作执行约 200ms，随后成功或到达明确显示的一步上限。**没有长时程信用分配，不称为顺序 RL、商业游戏收益或原能力保持。** 环境和奖励合同见[反馈说明](feedback.md)，五类动作限制见[解码实验](latency.md)。两组实际 128 步训练及冻结回合评测已完成，主评测得分相同，均存在未训练规则上的退化。

## 固定输入与目标

自行生成 horizon1 语料，训练 reach 种子 0–63、验证 hazard 1000–1007、源测试 keydoor 2000–2007。80 回合/80 动作窗口实际录下，64 个训练与 8 个验证专家回合成功；8 个钥匙门回合一步内不可完成，按真实结果截断，不能计作成功。钥匙门源划分只保留首动作和游戏隔离，不用它定义主任务成功率。

训练只用 64 张唯一 reach 初始 RGB。两条分支共享这份由专家执行记录冻结的初始语料和相同两轮打乱顺序；不能说奖励分支没有专家来源或数据准备成本。学生输入仅为 RGB、固定可见规则、零历史；引擎状态与教师标签留在源审计中。奖励分支实时重置同一 seed，检查实际 RGB 像素与对应冻结训练图相同，再作决策。

同一原始 0.8B 基座、全新且相同初始化的 q/v LoRA：rank 8、alpha 16、dropout 0、bf16，AdamW lr 0.0001、weight decay 0.01、梯度裁剪 1、batch 1、seed 20261028、确定性算法、128 次总更新。两者都用一次 forward 的 W/A/S/D/noop 五类 logits，温度 2；这是首键分类目标，不是原来的完整动作文本损失。

- SFT 用实际完成的专家动作类别，损失为 `-log_softmax(logits/2)[expert_class]`。
- Bandit 从相同温度的 `Categorical` 采样，执行实际动作并审计奖励，损失为 `-(actual_reward - 0.19) * log_prob(sampled_class)`。教师动作不会进入此采样或目标；奖励是脱离计算图的 Python 标量，来自实际状态转移。

常数 baseline 0.19 在训练前固定，等于均匀五类的奖励先验 `(0.99 + 4×(-0.01))/5`，与本次采样动作/状态无关；没有奖励模型或隐含 EMA 状态。正确动作优势为 +0.8，错误动作为 −0.2；熵正则为零。每步的实际奖励、采样类、原 logits、损失和梯度均保留，负的 bandit loss 本身不表示错误。

两者各用 128 次观测、forward/backward 和优化器更新。共享的 64 次专家采集成本另记；SFT 使用 128 次离线标签，bandit 增加 128 次在线实际动作。相同更新数不等于相同总交互、标签或墙钟成本。所有原始媒体、轨迹、采样记录与权重留本地。

## 运行与恢复

先按[训练环境](training.md)核实并设置 `PY`、`MODEL`、`GPU_LOCK`。转换清单绑定原执行环境的绝对路径，WSL 生成的数据应在 WSL 中加载；跨环境移动需要重新审计/转换，不能直接把 WSL 路径当 Windows 路径。

```bash
$PY -m scripts.training.collect_headless --output output/bandit-corpus \
  --train 64 --validation 8 --test 8 --max-steps 1

for ARM in sft bandit; do
  $PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 240 -- \
    $PY -m scripts.training.bandit_train --arm "$ARM" --model "$MODEL" \
    --dataset output/bandit-corpus/converted --output "output/class-$ARM-pilot" \
    --steps 4 --save-every 4 --rank 8 --lr 0.0001 --seed 20261028 \
    --temperature 2 --baseline 0.19
done
```

每个 pilot 正常完成并确认有限梯度/显存后，再恢复到预定预算：

```bash
for ARM in sft bandit; do
  $PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
    $PY -m scripts.training.bandit_train --arm "$ARM" --model "$MODEL" \
    --dataset output/bandit-corpus/converted --output "output/class-$ARM-128" \
    --resume "output/class-$ARM-pilot/checkpoint-000004" \
    --steps 128 --save-every 32 --rank 8 --lr 0.0001 --seed 20261028 \
    --temperature 2 --baseline 0.19
done
```

检查点包含实际 trainables、优化器和 Python/Torch/CUDA RNG。身份另绑定 arm、基座/语料/初始化参数哈希、温度/baseline、奖励/动作合同与实际源码依赖，包含提示构造、数据、动作过滤。不能把某个分支检查点当另一个分支恢复，或在改输入协议后静默续训。bandit 的每次 reset 用唯一 update tag，历史仍为空，图像像素不因标签改变。

采集沿用逐决策/已完成动作/回合状态日志；完成实际动作后即使训练失败，该动作与反馈也保留，不能冒充已完成优化器更新。重跑时新建输出目录，报告失败额外消耗。CPU 小模型采样恢复检查不能代替实际 GPU 对照，训练损失也不能代替下述冻结任务结果。

## 冻结评测与边界

训练前固定主评测为 reach 12000–12015、hazard 13000–13015，各一步；前者是训练规则的新种子，后者是共用引擎的未训练规则，不混称独立游戏泛化。次评测为 keydoor 14000–14007、八步，是超出一步训练的规则/时长迁移。原始模型、SFT128、bandit128 都用相同首键解码、初始画面和预算，不按测试结果选检查点。

```bash
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 240 -- \
  $PY -m scripts.training.rollout --model "$MODEL" --decoder first-key \
  --games reach hazard --episodes-per-game 16 --seed-start 12000 --max-steps 1 \
  --history-len 0 --output output/class-original-primary
# 适配两组分别添加 --adapter output/class-ARM-128/checkpoint-000128/adapter，使用新输出目录。
# 次评测改为 --games keydoor --episodes-per-game 8 --seed-start 12000 --max-steps 8。
```

`--games` 保留全局种子偏移，单选 keydoor 仍加 2000，不重新编号。模型未获知任何源教师动作、奖励、状态或未来画面。模板格式合法是构造保证；报告全部计划/完成/失败、实际成功/回报和交互成本。静态 Doom 输出漂移只能作探针，缺少任务真值时不能证明遗忘或保持。

## 实际训练证据与成本

2026-10-27，两条分支都实际完成 128 次更新，基座为先前固定的 0.8B S3 快照，RTX 5070 Ti Laptop、bf16、Torch 2.10.0/cu128、Transformers 5.14.1、PEFT 0.21.2。源语料 SHA256 为 `471de4e783272695ad24825ef502ecc56539d92aadda991007ca54ac5bdce8dd`。128 个对应样本的顺序、RGB 像素、初始化、源码及优化器预算已逐项核对；初始 LoRA 参数指纹相同，都是 `6d90e7d472c0150216475c35ab577110c75bd440dfe04b847c14b0be8e6f4187`。

| 训练项 | 首键 SFT | 奖励学习 |
|---|---:|---:|
| 优化器更新 / 输入观测 | 128 / 128 | 128 / 128 |
| 有限、非零梯度更新 | 128 | 128 |
| 梯度目标中专家标签使用 | 128 | 0 |
| 新增实际交互 | 0 | 128 |
| 更新区间总墙钟 | 92.14s | 140.35s |
| 含各段模型加载的总耗时 | 120.75s | 185.51s |
| 峰值 CUDA 分配 | 3195.07MiB | 3193.58MiB |

两者共同的语料准备另有 80 次实际专家动作，其中 64 次用于训练；不能在上表新增交互为零的一列中把这项成本省略。监督分支分两段，奖励分支为验证恢复而分三段，加载次数不同；墙钟并非配对算法效率结论。额外连续四步的 GPU 恢复验证还花了四次交互，独立记录，不混入主训练预算或隐藏消耗。

奖励分支每 32 次训练交互的成功数依次为 9、23、32、32；原始采样、环境奖励和全部 128 个完整回合重新审计，损失按记录 logits/实际奖励复算通过。这是训练曲线，不是冻结评测得分。SFT adapter 权重 SHA256 为 `7d727dffcc16b1ec9d6c46b651fe12ac7f14259ff457bb4181dd64bc6f00311d`；奖励 adapter 为 `749b147d9eb0c95913b22a569afb1c1c920495e58d89abb40cb2771a1e10d31d`。

实际 GPU **2+2 步与连续 4 步**对照中，采样/奖励/logits/loss/梯度序列、24 个 trainable 张量、优化器和 Python/Torch/CUDA RNG 全部逐位一致，最大参数差为 0；没有把这项四步证据扩写成完整 128 步跨设备复现。

## 冻结任务结果与失败

原始、SFT128、bandit128 各自完成全部 32 个主评测回合和 8 个次评测回合，无执行失败或遗漏；三个模型的种子、初始 RGB/状态、提示、基础权重、动作合同和相应回合上限逐项一致。权重在评测期间冻结，未按测试表现选点或调参。每条实际动作/奖励都通过状态重放、时间、媒体和摘要审计。

SFT 次评测另有一次 adapter 目录参数写错的启动失败，发生在模型加载前，实际动作数为零；失败状态保留。使用正确检查点路径和新输出目录重试，冻结参数与八个计划种子不变，未删除或替换任何已执行回合。

| 冻结任务 | 原始成功 / 平均回报 | SFT128 | Bandit128 |
|---|---:|---:|---:|
| reach，新种子，一步，16 回合 | 8/16 / 0.4900 | 16/16 / 0.9900 | 16/16 / 0.9900 |
| hazard，未训练规则，一步，16 回合 | 5/16 / 0.0525 | 12/16 / 0.4900 | 12/16 / 0.4900 |
| keydoor，规则/时长迁移，八步，8 回合 | 0/8 / −0.0800 | 6/8 / 0.7050 | 7/8 / 0.8350 |

主评测中 SFT 与 bandit 的 32 个实际动作全部相同。相对原始模型，reach 回报为 8 胜/8 平/0 负；hazard 为 11 胜/1 平/**4 负**。hazard 的 13000、13004、13008、13012 四个种子上，原始模型向上到达绿色目标，两组适配都向下进入红色危险格，回报从 +0.99 变成 −1.01。总分改善同时存在规则迁移退化；不能声称已解决危险识别或保持原能力，当前记录不确定该共同错误的机制。

八步次评测中 bandit 相对 SFT 为 1 胜/7 平/0 负，唯一胜出种子是 14007，平均回报差 +0.1300；相对原始为 7 胜/1 平/0 负。SFT 相对原始为 6 胜/2 平/0 负。单个额外成功和八个样本不足以证明奖励学习优于监督；没有独立训练重复或统计显著性证据。训练只用一种 reach 规则，评测仍共享简单网格引擎，不能称为广泛跨游戏泛化。

所有动作期间真实等待约 200ms，但世界在推理期间暂停。这里的任务收益不等于持续运行环境的实时收益，也不能用分类器安全模板的格式合法率推断完整动作生成能力。先前 128 步完整文本 SFT 的结果与本表采用不同种子、损失和解码器，不作直接方法排序。

同一官方 Doom 保存图/提示的完整文本解码探针中，原始模型为鼠标 `(10,-1)`、六段 `LB`；首键 SFT 后为 `(-10,-1)`、六段 `LB`，bandit 后为 `(-10,1)`、六段 `LB`。两者均合法但都不再与原始动作一致，说明这次训练目标以外的输出发生漂移。没有实际 Doom 回报或独立保持集，不能把漂移量当遗忘分数，也不能声称原能力保持。
