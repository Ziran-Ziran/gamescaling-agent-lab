# 原解码器的平衡小批次音频适配对照

[此前双图 LoRA128](audio-vision.md)在八种消融条件中都选择 A，没有建立融合收益；[冻结视觉联合 MLP](vision-readout.md)能读出同一人工任务的信息，但跳过原语言/动作解码器。本实验回到原模型的五类首键预测，测试更长、严格平衡的训练，并在相同样本/缺失曝光下比较成对与普通平衡组批。不会用新实验覆盖此前负结果。

## 冻结预算与输入

基座仍是 `THUSI-Lab/GameScaling-0.8B-S3` revision `1b89640dcf2ef51fb53ad9338a0725e9622dd200`，权重 SHA256 `d553c0796833691e32277a937db649eaac39bae9a0028a63c964c12a261c2dfa`。复用已审计、授权的64个 reach 初始观测，RGB manifest `19702bcd209526bab4793ffa4de9e89398116f7993dbea01aafa9705251dede5`，音频 manifest `d99732d3c7827fb6721e30055da6fa5e491e544d4d8259f88ef1fba840324949`。没有新数据资产或模型下载。

两组都从原模型新建 q_proj/v_proj LoRA：seed20261031、rank8、alpha16、dropout0、AdamW、lr0.0001、weight decay0.01、温度2、全局梯度裁剪1。基座 bf16，PEFT 的24个实际可训练张量为 float32，检查点与导出逐张量核实。每组固定200次优化器更新，每次4个单例前后向，各 CE/4 后立即 backward、释放图，最后只裁剪/更新一次。共800次标签使用；额外的连续4步恢复对照另计16次标签使用。模型只接收当前 RGB、因果 STFT 图片和缺失掩码，教师动作只进入损失/采样器，没有目标、频率标签、reset、引擎真值或奖励输入。

运行于同一 RTX 5070 Ti Laptop、Torch2.10.0+cu128、Transformers5.14.1、PEFT0.21.2、torchvision0.25.0、四个CPU线程，保留 Torch 回退。新恢复身份另外绑定实际图像处理器类型、规范化配置和 torchvision 版本；不修改之前实验的冻结依赖。

## 唯一组批干预

64行构成16个完整上下文，每个上下文是两种颜色位置×两种音调：相反音调有完全相同的 RGB 和相反标签；相同音调、镜像位置也有相反标签。两组每个小批次都严格含2A/2D。

每个 epoch 先共享洗牌16个上下文，再分成两个8上下文/32行块。每8次更新中，固定最后一个上下文的4行作共同的缺失更新，另外28行音频有效：

- `paired-context`：七个有效批次各包含一个完整上下文。
- `balanced-unpaired`：将相同28行按 A/D 桶确定性洗牌，再分成七个2A/2D批次；第八批缺失样本与前者完全相同。

200更新正好覆盖25个这样的块，即12.5个64行 epoch。包括末尾半个 epoch 在内，每个样本的有效/缺失曝光次数两组逐项相同；仅有效批次的组合变化。缺失共有25次更新、100个标签（A50/D50），有效700标签（A350/D350）。每次恢复以全局更新数决定日程，只在完整四个 micro 更新和优化器更新后保存检查点。

这是比旧128次单例训练更大的预算；与旧结果的差异不能只归因于配对、平衡或任何单一变化。两组的新对照固定了共同初值、样本曝光、缺失分配和更新数，但只有一个训练种子，不提供一般性方法优劣结论。

## 实际恢复与资源预检

两组各自实际完成2步、恢复再2步、独立连续4步。CPU 比较器严格比较全部24个训练张量、优化器、Python/Torch/CUDA RNG、有效输入和原始 logits/损失/梯度序列，均逐位相同、参数最大差0。前4步没有计划缺失更新，因此不宣称已逐位验收跨缺失边界恢复。

成对组两步短试使用8个标签，峰值 CUDA 分配3540.70MiB；第一步5.367秒、第二步2.913秒。两组连续短跑的暖步最大值约2.989/3.038秒。按剩余196步×暖步最大值×1.3再加120秒加载/保存/IO余量，预测约882/894秒。它只是小样本资源预测；主任务仍受整个进程树1100秒硬上限和共享锁保护，预测本身不能替代实际完成证据；实际200步结果另列如下。

## 冻结评测与审计入口

两组实际200更新已全部完成，主路径共400更新/1600标签；每组200个梯度均有限且非零。完整训练/cohort 在独立 CPU 进程重新审计后，JSON 与原报告完全一致，CUDA 未初始化。共同初值指纹为 `da7ebd2fbf865ac6355ea6429c9001654f41b353fdd4ba0caaa820bfceaf8660`，最终权重如下：

| 产物/成本 | paired-context | balanced-unpaired |
|---|---:|---:|
| 最终 adapter 权重 SHA256 | `e017302b535a028872b5899493259ee7429020ffb5da53b0933f1736e6f7fa44` | `08dec8d88ae273b30880a6bfc4dad1831346177802f11dd1549934beff35e9ad` |
| 主路径进程内墙时合计 | 651.904秒 | 619.000秒 |
| 更新计算时间合计 | 576.686秒 | 575.201秒 |
| 更新计算经验 P50 | 2.859秒 | 2.844秒 |
| 独立连续4步控制的进程内墙时 | 26.283秒 | 22.641秒 |
| 峰值 CUDA 分配 | 3540.70MiB | 3540.70MiB |
| CPU 重算逐例 CE 最大绝对差 | 0.0000004768 | 0.0000001788 |

主路径包括实际2+2+196更新；进程内墙时来自各分段状态的开始/结束，包含初始化、数据审计和保存，不含外部队列等待、跨进程 CPU 复核及独立控制。更新计算子项不包括其后日志/状态/检查点写入，不能替代完整成本。顺序运行的成本差没有配对速度结论。配置文件 SHA256 分别为 `2dc33d71032002e5ed0d6000e5ec8555d1de33062a4570a9b4900168d1a8f316` 和 `0682c3286ced6c1a0200c5b5d4df8c0468abcaef4f7b48d14ef01d43902e5911`。实际 JSON 唯一差异是 `target_modules` 列表序列化顺序相反；两者目标集合相同，仍分别绑定原始文件哈希。

在任何新任务指标之前保存的完整 cohort SHA256 为 `4058e462f2eb55dcfb2278447e65977a0d3a8251635c5efbfb471417ac0a3122`；两组800标签的有效/缺失曝光和24个最终导出张量均已绑定到该文件。上述训练完成和低 CE 不是未见上下文能力结论。

所有新200步权重完成并审计后，`audio_batch_audit` 才生成含四个固定策略身份的 cohort：原基座、此前负结果的 audio-LoRA128、两组新的200步 adapter。该文件在任何新任务指标产生前保存，不按结果选择检查点。

新评测为 hazard 35000–35003 的规则迁移，以及 reach 36000–36003 的同规则新上下文；每套仅4个独立上下文，各×两布局×两音调为16观测/条件；16行不是16个独立场景。四个策略、两套、八条件（aligned/rgb-only/missing/swapped/future/noise/silence/rgb-blank）共1024个实际单步回合。hazard 的正确动作仍为 A/D，不检验复杂避障。实际200ms传感器和动作窗口、暂停世界的边界不变。

CPU 审计重新检查来源 RGB/WAV、因果窗口和干预，重建实际提示/图像处理/输入 token 数，核对原始 logits→argmax→模板→实际动作、默认 reset、反馈与全部汇总。缺失、交换、遮挡等条件必须一起报告；仅 aligned 分数提高不足以证明模型使用音频。训练审计逐例重算800个温度化 CE（CPU/GPU浮点绝对容差0.000002）、实际源输入和每样本曝光，并检查每个参数的优化器步数与完整检查点。最终 safetensors 的全部24个张量须与已审计状态逐位相同，LoRA配置和基座快照也核对；cohort 只能使用这个已绑定的导出身份。原始日志、媒体、权重和本机路径留在忽略目录。

根据当前设备设置已有的 `PY`、`MODEL`、`DATA` 和 `GPU_LOCK`，不推测本机路径。以下使用成对组；普通平衡组将 arm 换为 `balanced-unpaired` 并使用独立输出目录。

```bash
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 240 -- \
  $PY -m scripts.training.audio_batch_train --arm paired-context --model "$MODEL" \
  --dataset "$DATA" --steps 2 --save-every 2 --output output/batch-paired-pilot2
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 240 -- \
  $PY -m scripts.training.audio_batch_train --arm paired-context --model "$MODEL" \
  --dataset "$DATA" --steps 4 --save-every 2 --resume output/batch-paired-pilot2/checkpoint-000002 \
  --output output/batch-paired-resume4
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 240 -- \
  $PY -m scripts.training.audio_batch_train --arm paired-context --model "$MODEL" \
  --dataset "$DATA" --steps 4 --save-every 2 --output output/batch-paired-continuous4
$PY -m scripts.training.audio_batch_audit --segments output/batch-paired-pilot2 output/batch-paired-resume4 \
  --continuous output/batch-paired-continuous4 --output output/batch-paired-resume-audit.json
# 两组恢复/资源预检均通过后，保持原预算，逐组正常完成再推进。
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 1100 -- \
  $PY -m scripts.training.audio_batch_train --arm paired-context --model "$MODEL" --dataset "$DATA" \
  --steps 200 --save-every 25 --resume output/batch-paired-resume4/checkpoint-000004 \
  --output output/batch-paired200
$PY -m scripts.training.audio_batch_audit --segments output/batch-paired-pilot2 \
  output/batch-paired-resume4 output/batch-paired200 --model "$MODEL" --dataset "$DATA" \
  --output output/batch-paired-training-audit.json
```

两组200更新完成后，冻结并运行全部新条件：

```bash
$PY -m scripts.training.audio_batch_audit --model "$MODEL" --dataset "$DATA" \
  --paired-segments output/batch-paired-pilot2 output/batch-paired-resume4 output/batch-paired200 \
  --unpaired-segments output/batch-unpaired-pilot2 output/batch-unpaired-resume4 output/batch-unpaired200 \
  --output output/batch-cohort.json
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 900 -- \
  $PY -m scripts.training.audio_batch_eval --model "$MODEL" --cohort output/batch-cohort.json \
  --policy original --suite transfer --output output/batch-original-transfer
# 对其余三个固定策略传对应 --adapter；两套都运行，不能按分数只做部分条件。
$PY -m scripts.training.audio_batch_compare --model "$MODEL" --cohort output/batch-cohort.json \
  --original output/batch-original-transfer --legacy128 output/batch-legacy128-transfer \
  --paired-context output/batch-paired-context-transfer \
  --balanced-unpaired output/batch-balanced-unpaired-transfer --output output/batch-transfer-comparison.json
```

这里“实际模型训练完成”“额外模态带来收益”“规则迁移”和“原能力保留”分别验收；没有原生波形、真实游戏泛化、持续实时或顺序 RL 结论。

## 实际1024回合质量结果

全部8个作业正常完成，1024个实际单步回合/动作、0失败记录；四策略来源输入、实际动作和反馈及两套跨策略比较均已完整 CPU 审计。各条件在相同初始 RGB、WAV 和 reset 上比较，训练期间未见这些评测种子。以下每格为成功数/16（平均回报）。

### hazard 规则迁移

| 条件 | original | legacy128 | paired-context200 | balanced-unpaired200 |
|---|---:|---:|---:|---:|
| aligned | 0/16 (-1.01) | 8/16 (0.49) | 16/16 (0.99) | 16/16 (0.99) |
| rgb-only | 0/16 (-1.01) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |
| missing | 0/16 (-1.01) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |
| swapped | 0/16 (-1.01) | 8/16 (0.49) | 0/16 (-0.01) | 0/16 (-0.01) |
| future | 0/16 (-1.01) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |
| noise | 0/16 (-1.01) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |
| silence | 0/16 (-1.01) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |
| rgb-blank | 0/16 (-1.01) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |

### reach 同规则新上下文

| 条件 | original | legacy128 | paired-context200 | balanced-unpaired200 |
|---|---:|---:|---:|---:|
| aligned | 0/16 (-0.01) | 8/16 (0.49) | 16/16 (0.99) | 16/16 (0.99) |
| rgb-only | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |
| missing | 2/16 (0.12) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |
| swapped | 0/16 (-0.01) | 8/16 (0.49) | 0/16 (-0.01) | 0/16 (-0.01) |
| future | 2/16 (0.12) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |
| noise | 0/16 (-0.01) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |
| silence | 0/16 (-0.01) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |
| rgb-blank | 0/16 (-0.01) | 8/16 (0.49) | 8/16 (0.49) | 8/16 (0.49) |

两组新的原解码器适配在两套对齐条件均16/16、交换音调均0/16；移除或遮蔽任一传感器均8/16。这组干预支持模型在本人工任务中利用 RGB 与音频联合信息，不能仅由固定单一动作解释。缺失输入中的相反教师标签本就不可辨，50%不能当成缺失实现失效；未来窗口被因果规则屏蔽，其50%也不能称为读取了未来。静音是有效传感器输入，仍与缺失分开审计。

两种新组批的各条件总成功数完全相同，没有观察到成对组批优于普通平衡组。动作不必逐条相同：迁移 noise 的成对相对普通平衡组回报胜/平/负为1/14/1，IID 的 rgb-only/noise 为1/14/1、missing/future 为8/0/8、silence 为4/8/4；因此相同总分并不表示逐回合或动作全等。与旧128步负结果相比同时改变了更新数、有效批量、标签曝光和缺失平衡，不能把新收益单独归因于任何一个变化。规则迁移中的正确动作仍只有 A/D，不是复杂避障、商业游戏或广泛泛化证据。

### 本次质量运行的时间边界

以下仅为 aligned 条件、每策略每套16个实际回合的经验 P50，单位 ms；不是将全部128条件回合合并，也不是交错配对的速度实验。

| 套件/策略 | 推理 | RGB观测到动作开始 | RGB观测到动作结束 | 音频窗口起点到动作结束 |
|---|---:|---:|---:|---:|
| transfer/paired-context | 222.77 | 376.09 | 576.29 | 777.20 |
| transfer/balanced-unpaired | 239.55 | 384.91 | 585.14 | 787.00 |
| iid/paired-context | 251.77 | 389.73 | 589.93 | 792.12 |
| iid/balanced-unpaired | 255.45 | 401.48 | 601.62 | 803.08 |

真实200ms传感器窗口与实际200ms动作窗口均计入相应端到端边界；世界在推理时暂停。对齐成功不能替代持续实时验收，也不能把推理子项当作截图到执行延迟。

## 预先冻结的次要 RGB 保留评测

在任何新质量指标产生前另存保留协议：同四个最终 cohort 策略，在 reach40000–40015、hazard41000–41015 上各16个单步回合，keydoor42000–42007 上8个八步回合，共160个计划实际回合、最多384次动作。先完成上述全部音频消融，再运行这里的所有策略，不按音频分数选择。它与首次[五策略保留矩阵](retention.md)使用不同种子，不混合分母。

仍用已有 RGB `rollout --decoder first-key --history-len 0`，各套 `--seed-start 40000`，其余参数对应主套 `--games reach hazard --episodes-per-game 16 --max-steps 1`、次套 `--games keydoor --episodes-per-game 8 --max-steps 8`。音频 adapter 的标准 RGB 规则提示与单图输入是显式观测/任务干预。`audio_batch_retention` 复用原始动作/时钟/奖励审计，并将四个实际 adapter 绑定到同一 cohort，输出原成功损伤种子及保留比例；原成功为0时仍为 `null`、不具信息性。

```bash
$PY -m scripts.training.audio_batch_retention --cohort output/batch-cohort.json --suite primary \
  --original output/batch-rgb-original-primary --legacy128 output/batch-rgb-legacy128-primary \
  --paired-context output/batch-rgb-paired-context-primary \
  --balanced-unpaired output/batch-rgb-balanced-unpaired-primary --output output/batch-rgb-primary-comparison.json
```

## 实际 RGB 保留结果

预先声明的160回合全部正常完成，共384次实际动作；各来源输入、默认提示、原始 logits、动作、真实时钟、反馈和汇总均完整审计。以下每格为成功数/分母（平均回报）。

| 固定策略 | reach | hazard | keydoor | 原 reach 成功保留 | 原 hazard 成功保留 |
|---|---:|---:|---:|---:|---:|
| original | 7/16 (0.4275) | 6/16 (0.1150) | 0/8 (-0.0800) | 基准 | 基准 |
| legacy128 | 8/16 (0.4900) | 4/16 (-0.0100) | 0/8 (-0.0800) | 3/7 | 2/6 |
| paired-context | 8/16 (0.4900) | 4/16 (-0.0100) | 0/8 (-0.0800) | 3/7 | 2/6 |
| balanced-unpaired | 8/16 (0.4900) | 4/16 (-0.0100) | 0/8 (-0.0800) | 3/7 | 2/6 |

三个音频 adapter 均在 reach 损失原成功种子40000、40004、40008、40012；hazard 损失41000、41004、41008、41012。reach 总成功7→8仍只保留3/7，hazard 总成功6→4只保留2/6。逐例回报胜/平/负分别5/7/4和6/2/8，相对原模型平均回报变化+0.0625和−0.125。这些损伤都是原策略 W 到适配策略 D，结局为 horizon，并非踩 hazard 死亡。两组新的融合适配并未修复标准 RGB 决策损伤，不能用对齐音频收益掩盖这个结果。

keydoor 四策略均0/8、平均−0.08；保留比例为 `null` 且不具信息性，不称为保持了原能力。音频任务与标准 RGB 的规则提示、图像数量和观测合同不同，这是显式任务/观测干预下的具体损伤，不外推为原始商业游戏能力遗忘比例。

独立检查实际初始解码像素：两套音频各16观测/8个唯一 RGB，RGB主套32个唯一初始RGB、次套8个；四策略对应输入完全一致，与本次音频64行 TRAIN 的32个唯一 RGB 精确交集均0。该检查排除这些初始画面的逐像素复用，不证明玩具规则或任务分布独立，也不替代按游戏划分。
