# 冻结视觉表示的独立读出实验

本实验在原始 GameScaling 0.8B 的**冻结视觉编码器**后训练新的任务专用 CPU 分类头，检查 RGB 与音频频谱图的联合信息能否被读出。它跳过语言/动作解码器，不能把结果称为此前 [LoRA 音频实验](audio-vision.md) 已修复，也不是原生音频理解或通用玩家能力。

## 输入和训练合同

复用已经审计的虚拟音频任务与专家来源，源 SHA256 为 `ee4b13454ed4660ec03ea9b7104f3a00b1ca642b129184d5f2f6e07e6f98e29c`。只提取 TRAIN 的 64 个 reach 初始观测；验证、测试及新评测上下文不参与训练、标准化或选点。RGB 与 WAV 的因果时间、实际动作、终止和游戏划分仍由原专属审计重算。

基座为 `THUSI-Lab/GameScaling-0.8B-S3`，快照 `1b89640dcf2ef51fb53ad9338a0725e9622dd200`，上游源码提交 `5c8c0322868171a53c9e20dface316b8afe9120d`，模型权重 SHA256 为 `d553c0796833691e32277a937db649eaac39bae9a0028a63c964c12a261c2dfa`。本次运行环境为 RTX 5070 Ti Laptop、bf16、torch2.10/cu128、transformers5.14.1、torchvision0.25.0、NumPy2.2.6、Pillow13。模型、处理器与实际依赖/源码哈希绑定在缓存及检查点身份中。

`get_image_features` 每次只接一张可观测图片，不接文本、文件名、reset、目标、标签或引擎状态。实际输入网格为 `[1,16,16]`，视觉合并后为 8×8×1024 token；按空间 2×2 自适应平均池化，得到 channel-major 的 4096 维 CPU float32 向量。原始视觉权重全冻结，语言层不运行。RGB 与 STFT 分别编码，不把两张图伪装为时间历史。

64 个原始训练输入全部再配一个缺失音频版本，形成 128 个全批量样本：present64、missing64，各自 A/D 标签各32。相同 RGB 的相反目标始终保留，缺失时不能凭空推断有效颜色。黑色 STFT 的冻结特征作为明确声明的缺失占位，同时传布尔 present；有效静音保持 present=true。每个坐标的均值/标准差仅在这 128 个训练样本上拟合，std 下限 1e-4，随头保存。

| 组 | 输入门控 | 分类头 | 可训练参数 |
|---|---|---|---:|
| rgb-mlp | 音频特征和 present 置零 | Linear(8193,32)→GELU→Linear(32,5) | 262373 |
| audio-mlp | RGB 特征置零 | 同上，同初始参数 | 262373 |
| joint-mlp | RGB、STFT、present | 同上，同初始参数 | 262373 |
| joint-linear | RGB、STFT、present | Linear(8193,5) | 40970 |

固定种子20261030，CPU float32、4线程、确定性算法，全批量128、AdamW lr0.001/wd0.01、clip1、200次更新，不早停或按测试结果选点。每组共25600次带标签样本使用，不能与此前 LoRA 的128次 batch1更新称为相同成本。新训练不产生额外环境交互，但不能省略共享的专家采集及特征提取成本。

实际缓存编码了36种像素输入：32种 RGB、3种 STFT（相位/量化导致像素不同）、1个黑色占位；按真实内容哈希去重，没有硬编码“两种频率等于两种图片”。最终缓存 SHA256 为 `36f367f9242f6f34941d4610a3fbbfc87756daa30e554640e3be9022c4bedf85`。

首次缓存的处理器身份含 tuple，JSON 落盘变为 list，导致首个 CPU 训练在更新前被严格身份校验阻断。已保存失败，加入往返序列化回归、统一 canonical JSON 身份，再完整重新提取 v2；没有跳过哈希检查。另一次启动因数据目录名错误在更新前退出，修正命令后使用同一数据和预算。

## 复现入口

先按 [training](training.md) 设置当前设备上的 `PY`、`MODEL`、`GPU_LOCK`，核实本地权重，不自动下载模型。GPU 锁必须覆盖整个进程树；每次检查正常完成后才启动下一项。

```bash
$PY -m scripts.training.audio_cue_data --output output/audio-cue-corpus
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  $PY -m scripts.training.vision_readout_data --model "$MODEL" \
  --dataset output/audio-cue-corpus/converted --output output/vision-features

# 先做2+2与连续4步恢复控制，再固定200步；训练进程在导入torch前隐藏CUDA。
$PY -m scripts.training.vision_readout_train --features output/vision-features \
  --dataset output/audio-cue-corpus/converted --arm joint-mlp \
  --steps 2 --save-every 2 --seed 20261030 --output output/readout-pilot2
$PY -m scripts.training.vision_readout_train --features output/vision-features \
  --dataset output/audio-cue-corpus/converted --arm joint-mlp \
  --resume output/readout-pilot2/checkpoint-000002 --steps 4 --save-every 4 \
  --seed 20261030 --output output/readout-resume4
$PY -m scripts.training.vision_readout_train --features output/vision-features \
  --dataset output/audio-cue-corpus/converted --arm joint-mlp \
  --steps 4 --save-every 4 --seed 20261030 --output output/readout-continuous4
# 核对恢复组与连续组的参数、优化器和RNG相同，再恢复到200步。
$PY -m scripts.training.vision_readout_train --features output/vision-features \
  --dataset output/audio-cue-corpus/converted --arm joint-mlp \
  --resume output/readout-resume4/checkpoint-000004 --steps 200 --save-every 50 \
  --seed 20261030 --output output/readout-joint
# rgb-mlp、audio-mlp、joint-linear 各用独立目录，从同一固定种子运行200步。
```

训练入口每次重新审计源数据和缓存，保存全部128×5原始 logits、loss、梯度范数、模型/优化器/CPU RNG，以及标准化和占位特征。实际2+2与连续4步的四个参数张量、优化器、Python/Torch RNG、完整 logits/loss/梯度序列逐位一致，最大参数差0；两者均无 CUDA RNG。它是 CPU 读出头恢复证据，不替代基座 LoRA 恢复实验。

四组实际200步均已完成，逐步重算损失与标签、确认有限非零梯度和相同数据/标准化/预算；CUDA 未初始化。最终训练集 present/missing 正确数依次为 joint-mlp64/64与32/64，其他三组均32/64与32/64。该表只说明训练拟合，不能当未见场景分数。联合头权重文件 SHA256 为 `4734d42b735bbe30c9b0631e73f0fb7f3aa037fa6102989ea129f5680b34e208`。

## 固定评测与证据重放

主评测使用新 hazard 上下文25000–25007，每个上下文2种颜色位置×2种音，共32回合/条件。训练前另固定同规则、新上下文 reach26000–26007，用来区分输入读出与未训练危险规则迁移；后者不是新游戏划分或跨游戏泛化证明。两套均完整运行四组、八种输入条件，共2048个实际回合，不依结果选择汇报项。

条件为 aligned、RGB-only、missing、swapped、future、noise、silence、RGB-blank；均复用原始 WAV 干预与因果掩码。动作五类 W/A/S/D/noop，0生成token、0预测语言token、1个类别；安全模板执行真实200ms动作，源环境反馈单独审计。输入仍为自有暂停式虚拟任务，不接麦克风、真实游戏或桌面。

```bash
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 900 -- \
  $PY -m scripts.training.vision_readout_eval --model "$MODEL" \
  --readout output/readout-joint --suite transfer --output output/readout-joint-transfer
# --suite iid 运行另一套；四个头都运行两套，各自使用新输出目录。
$PY -m scripts.training.vision_readout_compare \
  --rgb-mlp output/readout-rgb-transfer output/readout-rgb \
  --audio-mlp output/readout-audio-transfer output/readout-audio \
  --joint-linear output/readout-linear-transfer output/readout-linear \
  --joint-mlp output/readout-joint-transfer output/readout-joint \
  --output output/readout-transfer-comparison.json
```

每帧保存编码后的原始 CPU 特征、掩码、logits、像素与特征/头输入哈希。CPU 比较器重新检查实际 WAV/STFT/RGB、重放环境动作/回报、加载固定头逐位重算 logits，并核对实际动作，要求所有计划回合完成、四组数据/预算/标准化/初值和初始输入相同。特征证据写入的成本单列且纳入观测到动作的实际总延迟。文件、原始图像、特征和权重只留在忽略目录。

## 实际冻结结果

2026-10-27，四组、两套、八条件全部完成，共2048个实际回合/2048次动作，执行失败0。每组每套256回合，全部包含，不按结果换点或增加训练。以下每格为“成功数/32 / 平均回报”，每回合上限一步。

主评测：未训练的 hazard 规则，新上下文25000–25007。

| 条件 | RGB MLP | 音频 MLP | 联合线性 | 联合 MLP |
|---|---:|---:|---:|---:|
| aligned | 16/32 / 0.4900 | 16/32 / 0.4900 | 0/32 / −0.0100 | 32/32 / 0.9900 |
| rgb-only | 16/32 / 0.4900 | 16/32 / 0.4900 | 0/32 / −0.0100 | 16/32 / 0.4900 |
| missing | 16/32 / 0.4900 | 16/32 / 0.4900 | 0/32 / −0.0100 | 16/32 / 0.4900 |
| swapped | 16/32 / 0.4900 | 16/32 / 0.4900 | 0/32 / −0.0100 | 0/32 / −0.0100 |
| future | 16/32 / 0.4900 | 16/32 / 0.4900 | 0/32 / −0.0100 | 16/32 / 0.4900 |
| noise | 16/32 / 0.4900 | 16/32 / 0.4900 | 0/32 / −0.0100 | 16/32 / 0.4900 |
| silence | 16/32 / 0.4900 | 16/32 / 0.4900 | 0/32 / −0.0100 | 16/32 / 0.4900 |
| rgb-blank | 16/32 / 0.4900 | 16/32 / 0.4900 | 8/32 / −0.2600 | 16/32 / 0.4900 |

次评测：同训练规则的 reach，新上下文26000–26007；它不是另一个游戏划分。

| 条件 | RGB MLP | 音频 MLP | 联合线性 | 联合 MLP |
|---|---:|---:|---:|---:|
| aligned | 16/32 / 0.4900 | 16/32 / 0.4900 | 12/32 / 0.3650 | 32/32 / 0.9900 |
| rgb-only | 16/32 / 0.4900 | 16/32 / 0.4900 | 16/32 / 0.4900 | 16/32 / 0.4900 |
| missing | 16/32 / 0.4900 | 16/32 / 0.4900 | 16/32 / 0.4900 | 16/32 / 0.4900 |
| swapped | 16/32 / 0.4900 | 16/32 / 0.4900 | 14/32 / 0.4275 | 0/32 / −0.0100 |
| future | 16/32 / 0.4900 | 16/32 / 0.4900 | 16/32 / 0.4900 | 16/32 / 0.4900 |
| noise | 16/32 / 0.4900 | 16/32 / 0.4900 | 14/32 / 0.4275 | 16/32 / 0.4900 |
| silence | 16/32 / 0.4900 | 16/32 / 0.4900 | 16/32 / 0.4900 | 16/32 / 0.4900 |
| rgb-blank | 16/32 / 0.4900 | 16/32 / 0.4900 | 8/32 / 0.2400 | 16/32 / 0.4900 |

联合 MLP 在两套对齐条件中相对两种单模态各16胜/16平/0负，平均回报差+0.50；相反音频使其选择相反方向而降为0/32，缺失、未来、噪声、静音或遮住RGB均回到16/32。它支持**该人工任务在冻结视觉表示上的联合信息可读性**，不是此前 LoRA 原解码器的融合收益。主评测 aligned 的联合头 A/D 各16次，RGB头恒D；音频头虽也 A/D 各16，但缺少位置输入仍只成功一半。

线性头在固定预算下既未完美拟合训练集，且 hazard aligned 实际32次全为 noop，因此0/32、回报−0.01；遮掉 RGB 时出现8次成功但总回报−0.26。reach aligned 也只有12/32。这些失败保留，不能由本次准确率差直接推出唯一内部机制或一般性的架构优劣。

单回合训练 reset 的 GPU 冒烟已完成，峰值1676.20MiB；该冷启动回合不计入正式样本。正式实际对齐回合中，联合头 hazard 的推理 P50 为107.75ms，观测到动作开始265.51ms、到结束465.70ms、传感器窗口开始到动作结束666.44ms；reach 对应108.26、215.43、415.53、616.31ms。文件保存与特征证据写入包含在实际观测到动作延迟中，世界仍暂停；这些是顺序质量评测的观测值，不是配对速度提升或实时验收。

冻结视觉特征、新任务头、人工频谱规则、单个训练种子和共享玩具引擎都限制结论。没有商业游戏收益、广泛跨游戏泛化、原生音频能力或原模型遗忘保持结论；原始语言/动作解码器在这里未运行，也未被更新。
