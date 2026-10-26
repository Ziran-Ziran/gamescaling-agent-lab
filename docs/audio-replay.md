# 同一音频父适配器上的 RGB 重放对照

[此前两组200步原解码器适配](audio-batch.md)在人工对齐音频上获得16/16，却仅保留3/7原 reach 成功、2/6原 hazard 成功。本研究针对这个已观察到的具体损伤，事先指名 paired-context200 为共同父状态，比较继续音频训练与增加 RGB 专家目标；不按新评测挑选父模型或检查点。旧128步负结果和200步损伤继续保留。

## 冻结数据、目标与预算

基座、硬件及项目隔离环境沿用 audio-batch：官方0.8B revision `1b89640dcf2ef51fb53ad9338a0725e9622dd200`、权重 SHA256 `d553c0796833691e32277a937db649eaac39bae9a0028a63c964c12a261c2dfa`，同一 RTX5070Ti Laptop/Torch2.10.0+cu128/Transformers5.14.1/PEFT0.21.2。原冻结 cohort SHA256 `4058e462f2eb55dcfb2278447e65977a0d3a8251635c5efbfb471417ac0a3122`，父 adapter 权重 `e017302b535a028872b5899493259ee7429020ffb5da53b0933f1736e6f7fa44`。

只从父检查点复制全部24个 float32可训练张量；基座仍bf16。父 state.pt SHA256 `6c564babbbff1ff6cc5c0862d5cb2c94883337bfe95cb35eb773be1e00cc07f3`，24张量指纹 `4056bac919e2dd3388ed48222c3e7d30336ad6a15bd224888c72ce6cbc62a217`。复制不恢复父优化器或 RNG。两组都从新seed20261101、fresh AdamW开始，rank8/alpha16/q_proj+v_proj/dropout0/lr0.0001/wd0.01/温度2/梯度裁剪1；自身恢复才加载本阶段优化器与全部 RNG。NumPy/Pillow、实际图像处理器和训练依赖均绑定到父身份，不能在改变 STFT 输入后继续旧身份。

两组固定64次优化器更新。每更新共同的两个音频 TRAIN 例为相同 RGB、相反音调/相反 A/D 标签，每个 CE 都除4：

- `audio-only-control` 只有这两个音频项，仍各 CE/4。
- `rgb-replay` 再加两个标准 RGB reach 专家 TRAIN 例，也各 CE/4；W/S 与 A/D 组合交替。

共同音频有32对/64行，每32步以局部 RNG 洗牌一遍；64步正好两遍，每个样本使用2次。全局第8、16……64步同时遮蔽两例音频：128标签中有效112（A56/D56），缺失16（A8/D8）。两组逐样本、顺序和缺失曝光完全相同。重放组另使用128个 RGB 标签，四方向各32次，按每类16行独立循环洗牌。音频 manifest `19702bcd209526bab4793ffa4de9e89398116f7993dbea01aafa9705251dede5`、音频索引 `d99732d3c7827fb6721e30055da6fa5e491e544d4d8259f88ef1fba840324949`；RGB 专家 manifest `d58ddde6623581a86ef59502f1e7d32253cc4d4f6bc5a40a40b6b975f3b516d2`。两者均只训练 reach，来源实际专家转移重新审计，hazard/keydoor 及评测教师不进入训练。

额外 RGB 标签和计算成本不同，比较的是增加 RGB 目标及其对全局裁剪/AdamW 的整体效果，不是等成本算法排名。记录两个模态的 CE、加权总目标、裁剪前梯度范数及其推导诊断系数、原始五类 logits 和实际输入证据。每个微例立即 CE/4.backward 后释放图，完整2或4例后才裁剪/更新；中途失败清空梯度且不更新。推理只接收当前传感器与可见规则，教师动作只进入损失，reset/引擎真值/奖励/历史均不进入学生。

## 恢复、冻结与完整评测

实际短跑先做每组4步、恢复到8步，以及独立连续8步，跨过第8步缺失边界；比较24参数、优化器、Python/Torch/CUDA RNG 和实际输入/logits/CE/梯度。主路径为4+4+56=64更新，额外连续8步单列，不能混入128/256个主路径标签预算。先测显存和耗时，再决定继续；单个主任务受整个进程树720秒硬上限与共享锁约束。

全阶段在首个 GPU 短跑前预留50分钟给两组训练、恢复控制、全部评测、CPU审计、文档/测试/审查和CI。主任务前须再用两组实测暖步×剩余56×1.3+120秒加载/保存余量检查预算。截止前无法完整完成时不启动；07:40起不新开重任务。实际未来音频/ RGB 调度每作业最多等待60秒，并将每个剩余作业的全部队列余量计入总阶段预测；首轮质量总预算为6×(90+60)+8×(50+60)+900=2680秒，包含后续完整RGB矩阵及15分钟审计/交付余量。

两组完整64步和逐例训练/24导出/父初始化审计通过后，才生成新 cohort，冻结 parent200、control、replay 和 RGB 原基座身份；任何新质量指标都在其后，固定全部策略一起报告：

- 音频：hazard47000–47001 与 reach48000–48001，各仅2个独立上下文×两布局×两音调=8观测/条件；aligned、missing、swapped、rgb-only、rgb-blank。三策略×两套×五条件共240个实际单步回合，8行不等于8个独立场景。
- RGB：原基座与上述三策略，reach50000–50007、hazard51000–51007各8个单步回合；keydoor52000–52003共4个八步回合。四策略共80回合、最多192动作。

总共320回合、最多432动作。CPU 重新审计来源/实际默认 reset与提示、传感器因果窗口/干预、原始 logits→动作、时钟/回报和汇总，分开报告相对父模型与原基座的变化。原成功为0时保留比例为 null、无信息。预检渲染的全部新初始 RGB 与两种 TRAIN 的像素交集均0；实际回合仍要再次检查。小样本、一个训练种子、人工频谱视觉编码与暂停世界的边界不会因结果改善而消失；不声明原生音频、商业游戏泛化、持续实时或顺序 RL。

## 复现入口

这些专属研究审计入口固定本次保留产物的身份；权重与原始轨迹不随仓库发布，只有保存了这些本地产物才能复核同一 cohort。通用数据采集/转换/训练入口见[训练说明](training.md)。重新采集会改变实际时间与清单哈希，跨电脑改变绝对来源路径也会改变 manifest 字节身份；应登记为新运行并先审计，不能修改旧冻结哈希冒充同一产物，也不承诺跨硬件逐位一致。

以下为 Bash/WSL 示例。先核实当前设备并设置已有的 `PY`、`MODEL`、`AUDIO_DATA`、`RGB_DATA`、`PARENT`、`PARENT_COHORT` 和 `GPU_LOCK`。不能从文档猜测本机路径。下面以 control 为例，replay 使用独立输出路径和 `--arm rgb-replay`：

```bash
set -euo pipefail
train_replay() {
  limit=$1; shift
  "$PY" scripts/resource_guard.py --lock "$GPU_LOCK" --seconds "$limit" -- \
    "$PY" -m scripts.training.audio_replay_train --arm "$ARM" --model "$MODEL" \
    --audio-dataset "$AUDIO_DATA" --rgb-dataset "$RGB_DATA" --parent "$PARENT" \
    --cohort "$PARENT_COHORT" "$@"
}
for ARM in audio-only-control rgb-replay; do
  RUN="output/replay-$ARM"
  train_replay 240 --steps 4 --save-every 4 --output "$RUN-pilot4"
  train_replay 240 --steps 8 --save-every 4 --resume "$RUN-pilot4/checkpoint-000004" --output "$RUN-resume8"
  train_replay 240 --steps 8 --save-every 4 --output "$RUN-continuous8"
  "$PY" -m scripts.training.audio_replay_audit --segments "$RUN-pilot4" "$RUN-resume8" \
    --continuous "$RUN-continuous8" --output "$RUN-resume-audit.json"
done
# 两组恢复/资源/完整剩余预算均通过后，再逐组继续；不按指标增减更新。
for ARM in audio-only-control rgb-replay; do
  RUN="output/replay-$ARM"
  train_replay 720 --steps 64 --save-every 16 --resume "$RUN-resume8/checkpoint-000008" --output "$RUN-64"
  "$PY" -m scripts.training.audio_replay_audit --segments "$RUN-pilot4" "$RUN-resume8" "$RUN-64" \
    --model "$MODEL" --audio-dataset "$AUDIO_DATA" --rgb-dataset "$RGB_DATA" \
    --cohort "$PARENT_COHORT" --output "$RUN-training-audit.json"
done
"$PY" -m scripts.training.audio_replay_audit \
  --control-segments output/replay-audio-only-control-pilot4 output/replay-audio-only-control-resume8 output/replay-audio-only-control-64 \
  --replay-segments output/replay-rgb-replay-pilot4 output/replay-rgb-replay-resume8 output/replay-rgb-replay-64 \
  --model "$MODEL" --audio-dataset "$AUDIO_DATA" --rgb-dataset "$RGB_DATA" \
  --cohort "$PARENT_COHORT" --output output/replay-cohort.json
"$PY" scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  "$PY" -m scripts.training.audio_replay_eval --model "$MODEL" --cohort output/replay-cohort.json \
  --policy parent200 --adapter "$PARENT/adapter" --suite transfer --output output/replay-parent-transfer
# 同样执行control/replay与IID全部五条件；全部完成后才跨策略比较。
"$PY" -m scripts.training.audio_replay_compare --model "$MODEL" --cohort output/replay-cohort.json \
  --parent200 output/replay-parent-transfer --audio-only-control output/replay-control-transfer \
  --rgb-replay output/replay-rgb-transfer --output output/replay-transfer-comparison.json
"$PY" scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  "$PY" -m scripts.training.rollout --model "$MODEL" --games reach hazard --seed-start 50000 \
  --episodes-per-game 8 --max-steps 1 --decoder first-key --history-len 0 --max-new-tokens 64 \
  --output output/replay-rgb-original-primary
# 四策略都跑：其余加对应--adapter；secondary使用--games keydoor --episodes-per-game 4 --max-steps 8。
"$PY" -m scripts.training.audio_replay_retention --cohort output/replay-cohort.json --suite primary \
  --original output/replay-rgb-original-primary --parent200 output/replay-rgb-parent-primary \
  --audio-only-control output/replay-rgb-control-primary --rgb-replay output/replay-rgb-replay-primary \
  --output output/replay-rgb-primary-comparison.json
```

## 实际训练与恢复结果

两组实际64更新及全部逐例输入/CE、父24参数初始化、检查点和最终24张量导出审计均已完成。每组64个梯度都有限且非零；两组128音频标签的逐样本有效/缺失曝光完全相同。实际4+4对连续8步包括第8步缺失，全部24参数、优化器、Python/Torch/CUDA RNG、实际输入/logits/损失/梯度逐位一致，最大参数差0。

| 产物/成本 | audio-only-control | rgb-replay |
|---|---:|---:|
| 最终 adapter 权重 SHA256 | `c98497aa5a41469693fe6203d833a60cd3aad34a391467387b6f66d7cce94076` | `e0c7836632bef4aa149bc5a351a193856060e7792e4a289c2b03af72c220699e` |
| 主路径标签数 | 128音频 | 128音频+128RGB |
| 主路径进程内墙时合计 | 166.052秒 | 230.070秒 |
| 更新计算合计 | 95.429秒 | 180.747秒 |
| 更新计算经验 P50 | 1.394秒 | 2.759秒 |
| 独立连续8步控制墙时 | 28.688秒 | 38.989秒 |
| 峰值 CUDA 分配 | 3540.70MiB | 3540.70MiB |
| CPU逐例 CE 最大绝对差 | 0.0000001192 | 0.0000002384 |

主路径为4+4+56，墙时包含各进程的数据审计、初始化和保存，不包含外部队列及另行 CPU 复核；更新计算子项不含后续日志/检查点写入。独立连续8步每组额外16音频标签，重放组还额外16RGB标签；两组总训练更新含控制为144，主路径为128，不把控制计入主训练。顺序运行且重放有额外计算，不作速度排名。

最终 cohort SHA256 `f87b8bf24af3c5b6850789e28a5e818d71d72d029f3b69e71cbc2e0490003e4b` 已在任何本轮新质量指标前冻结。control 配置 SHA256 `2dc33d71032002e5ed0d6000e5ec8555d1de33062a4570a9b4900168d1a8f316`、replay 为 `0682c3286ced6c1a0200c5b5d4df8c0468abcaef4f7b48d14ef01d43902e5911`；唯一 JSON 差异是相同目标模块集合的列表序列化顺序，不混淆文件身份。

实际追加训练与恢复完成；任务结果在下面单独验收，不能从低训练 CE 推出收益。

## 实际240局音频对照

全部六个作业正常完成，240个实际单步回合/动作、0失败记录；全部来源输入、默认 reset、提示/处理/token、原始 logits→实际动作、时钟、反馈和汇总通过完整 CPU 审计。每套只有2个独立上下文，以下每格为成功数/8（平均回报）。

### hazard 迁移

| 条件 | parent200 | audio-only-control64 | rgb-replay64 |
|---|---:|---:|---:|
| aligned | 8/8 (0.99) | 8/8 (0.99) | 6/8 (0.74) |
| missing | 4/8 (0.49) | 4/8 (0.49) | 4/8 (0.49) |
| swapped | 0/8 (-0.01) | 0/8 (-0.01) | 2/8 (0.24) |
| rgb-only | 4/8 (0.49) | 4/8 (0.49) | 4/8 (0.49) |
| rgb-blank | 4/8 (0.49) | 4/8 (0.49) | 4/8 (0.49) |

### reach 同规则新上下文

| 条件 | parent200 | audio-only-control64 | rgb-replay64 |
|---|---:|---:|---:|
| aligned | 8/8 (0.99) | 8/8 (0.99) | 8/8 (0.99) |
| missing | 4/8 (0.49) | 4/8 (0.49) | 4/8 (0.49) |
| swapped | 0/8 (-0.01) | 0/8 (-0.01) | 0/8 (-0.01) |
| rgb-only | 4/8 (0.49) | 4/8 (0.49) | 4/8 (0.49) |
| rgb-blank | 4/8 (0.49) | 4/8 (0.49) | 4/8 (0.49) |

迁移的两个损伤位于context47000/47001、colors=1/tone=1：父模型/对照为 A→goal，重放为 D→horizon。对应交换条件的两次成功来自colors=1/tone=0，输入干预后与上述情况相同，不解释为抗交换干扰改善。重放组在迁移对齐条件由8/8降到6/8，同时交换条件由0/8升到2/8；该具体损伤不能被同规则新上下文的8/8掩盖。单模态/缺失条件仍为4/8，不表示已恢复标准 RGB 任务；必须在另一套原规则提示和观测合同下检查。训练目标新增 RGB 的代价及其对裁剪/AdamW 的影响一起保留：实际裁剪更新数为control5/64、replay16/64，不能把不同目标的总损失直接排名。

首次标准 RGB 执行器因共享重任务锁忙，等待60秒后以75退出，未启动GPU子进程或产生回合目录。失败日志保留；资源释放后使用独立的零排队、可重审已完成前缀并跳过其GPU作业的恢复执行器，全部80局正常完成。没有通过覆盖失败产物或省略策略得到结果。

## 实际80局标准 RGB 保留

80个实际回合、173动作全部完成并逐条审计；与音频矩阵合计320回合、413动作。以下每格为成功数/分母（平均回报）。

| 策略 | reach | hazard | keydoor | 原 reach 成功保留 | 原 hazard 成功保留 |
|---|---:|---:|---:|---:|---:|
| original | 4/8 (0.4900) | 2/8 (-0.0100) | 0/4 (-0.0800) | 基准 | 基准 |
| parent200 | 4/8 (0.4900) | 2/8 (-0.0100) | 0/4 (-0.0800) | 2/4 | 0/2 |
| audio-only-control | 4/8 (0.4900) | 2/8 (-0.0100) | 0/4 (-0.0800) | 2/4 | 0/2 |
| rgb-replay | 8/8 (0.9900) | 6/8 (0.4900) | 4/4 (0.9675) | 4/4 | 0/2 |

重放组恢复了全部4个原 reach 成功，回报胜/平/负4/4/0、平均变化+0.5；父模型/音频对照虽然总成功同基座4/8，仍损失50000、50004两例，只保留2/4。

hazard 的原成功种子51000、51004被三个适配策略全部损失，保留均0/2。父模型/音频对照结局是 horizon，重放组则踩 hazard 死亡。重放组总成功6/8、回报胜/平/负6/0/2、平均变化+0.5，不能掩盖这两个原成功全部失败；这不是修复了 hazard 保留。

keydoor 重放组4/4、平均0.9675，共13个实际动作/平均3.25步；其他三策略均0/4、平均−0.08、各32动作。由于基座没有成功，所有保留比例为 `null`、无信息；这里是小样本任务收益，不是保持原能力。

本次新增 RGB 目标提高了上述 RGB 套件的平均回报，但没有消除 hazard 原成功损伤，并损失两个音频迁移成功。只支持本次单训练种子、两种模态合同和玩具任务中的具体权衡；不能概括为整体遗忘已解决或跨游戏泛化。

实际回合的初始画面已重新核验：音频两套各8观测/4个唯一RGB，RGB主套16个、次套4个唯一初始RGB；对应策略初始像素相同，且与音频和RGB两套TRAIN的精确像素交集均0。此检查不证明独立世界，也不将每个布局/音调当作独立场景。
