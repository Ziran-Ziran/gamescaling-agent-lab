# RGB 与音频频谱图的实际模型适配

该入口把因果虚拟音频转成固定频谱图，作为当前观测的第二张图，复用原始 GameScaling 0.8B 的视觉编码器和 q/v LoRA。模型收到的是 **RGB + 音频派生图像 + 显式缺失掩码**，没有原生波形编码器、语音能力或真实游戏音频理解的结论。此前 418 参数 CPU 融合冒烟见 [training](training.md)，不能替代这里的实际预训练模型训练。

## 任务与可识别性

蓝色点玩家看到黄色与青色两个候选目标，只有一个有效。200ms、16kHz、mono PCM16 虚拟传感器的 500Hz 音表示黄色，2000Hz 表示青色；两者 RMS 相同。颜色的左右位置在每个上下文中互换，同一 RGB 又配相反音频/实际目标。相反音频对和颜色互换组始终留在同一个游戏划分内。

完整平衡组中，RGB 单独无法知道哪个颜色有效；音频单独不知道该颜色在左还是右。因此主评测中，缺失音频或遮掉 RGB 的确定性策略，最多能在每对相同输入/相反目标里成功一半；若选择了别的方向还会更低。该控制可检查是否利用两种输入，但人工音色规则不代表一般声音理解。

三个划分使用实际不同的任务条件：训练 `audio-cue-reach-v1` 的两个目标在左右相邻格，一步上限；验证 `audio-cue-hazard-v1` 上下有真实致命红格，一步上限；测试 `audio-cue-slalom-v1` 目标各距两格且左右邻格被墙挡住，需要绕行，八步上限。后者会超出只含 A/D 标签的训练动作分布，失败必须保留。它们共享简单玩具引擎，不是独立商业游戏泛化集。

## 观测、时间与真值边界

`audio-cue-dual-color-v1` 是新的观测投影；其内层物理、六子步动作和奖励仍按 `headless-grid-feedback-v1` 重放。专属审计同时重置源世界、验证状态链和实际动作/回报、重投影双候选 RGB、核对虚拟 WAV 与 reset 的一致性，再检查成对像素相同和颜色位置平衡。不能把旧版仅渲染一个绿色目标的 RGB 审计当作新投影验收。

虚拟音频通过本地确定性函数生成，不接麦克风、扬声器或桌面。程序实际等待 200ms 传感器窗口，样本在内存可用后获取 RGB，再落盘证据。窗口 start/end、available、RGB captured、动作 start/end 同属单调时钟；落盘完成时间不冒充传感器可用时间。世界在推理和文件操作期间暂停。

复用 `audio.read_audio`：窗口必须在 RGB 前结束且已可用，距 RGB 不超过 50ms；错时钟、未来、尚未可用和过期均在打开文件前返回显式缺失。有效静音仍是 `present=1`。STFT 使用 256 点 Hann、hop128，截取 0–4000Hz，固定满幅参考的 −80..0dB 映射为 256×256 灰度 RGB 图；没有按每个样本峰值归一化，也不把测得频率或目标写进模型文本。

两张图片位于**同一条当前 user 观测**，不是伪造的历史帧。文本只有固定玩家规则和 present=0/1；文件名、episode/reset ID、奖励、教师动作、目标真值与引擎状态均不进入编码输入。RGB 转换器负责原有动作/终止/游戏划分检查，独立 audio index/manifest 再绑定源哈希、WAV 和时间；加载时重新审计而非只信 sidecar 哈希。RGB 与原始 WAV 的内容都禁止跨游戏划分重复。

首次实际采集在一个动作后被因果审计拦下：WSL 挂载盘写入和首次渲染让音频相对 RGB 过期约 102.7ms。原失败轨迹保留；修复采集顺序、加入慢写入时钟回归后，在新目录重采，未放宽 50ms 时效合同。新语料共 88 个完整专家回合、112 个动作，88 次 goal；训练 64 个 reach 回合、验证 16 个 hazard 回合、测试 8 个 slalom 回合/32 动作。源 SHA256 为 `ee4b13454ed4660ec03ea9b7104f3a00b1ca642b129184d5f2f6e07e6f98e29c`。三个划分最大音频年龄分别为 2.645、2.608、2.463ms；这仍是专家采集成功，不能当模型成绩。

## 运行和固定预算

先按 [training](training.md) 核实当前设备、设置 `PY`、`MODEL`、`GPU_LOCK`；只使用已有本地权重和项目环境。

```bash
$PY -m scripts.training.audio_cue_data --output output/audio-cue-corpus
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 240 -- \
  $PY -m scripts.training.audio_vision_train --model "$MODEL" \
  --dataset output/audio-cue-corpus/converted --output output/audio-pilot2 \
  --steps 2 --save-every 2 --seed 20261029
```

先检查实际双图 forward、有限非零梯度和显存。两步恢复到四步，与相同初始化的连续四步比较全部参数、优化器和 RNG；通过后恢复到固定总预算 128。训练是首键五类交叉熵、温度 2，rank8/alpha16/dropout0、bf16、AdamW lr0.0001/wd0.01、clip1、batch1、seed20261029。每第八个全局更新固定掩掉音频，继续记录该歧义输入/标签；恢复不重新计数。

```bash
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 240 -- \
  $PY -m scripts.training.audio_vision_train --model "$MODEL" \
  --dataset output/audio-cue-corpus/converted --output output/audio-resume4 \
  --resume output/audio-pilot2/checkpoint-000002 --steps 4 --save-every 4 --seed 20261029
# 检查正常完成后，单独运行连续四步控制，不能同时占用 GPU。
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 240 -- \
  $PY -m scripts.training.audio_vision_train --model "$MODEL" \
  --dataset output/audio-cue-corpus/converted --output output/audio-continuous4 \
  --steps 4 --save-every 4 --seed 20261029
```

```bash
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  $PY -m scripts.training.audio_vision_train --model "$MODEL" \
  --dataset output/audio-cue-corpus/converted --output output/audio-128 \
  --resume output/audio-resume4/checkpoint-000004 \
  --steps 128 --save-every 32 --seed 20261029
```

检查点绑定 RGB/audio manifest、初始化、实际源码、NumPy/Pillow 等频谱依赖版本和确定性配置。每步记录实际输入像素/提示哈希、音频掩码、原始五类 logits、目标、loss、梯度和显存；不能把只创建配置称为模型适配。

## 冻结评测

训练前固定新 hazard 上下文 23000–23003，每个上下文两种颜色位置、两种音频，共 16 回合/条件，原始基座与固定适配点各测一次。aligned、RGB 单图、显式缺失、相反原始音频交换、未来音频掩码、原始噪声 WAV、有效静音、遮掉 RGB 八种条件都实际执行动作、审计真实回报。噪声在相同上下文的相反目标组中相同；不会给两种标签分别生成可识别噪声。

```bash
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  $PY -m scripts.training.audio_vision_eval --model "$MODEL" \
  --suite primary --output output/audio-original-primary
# 适配组添加 --adapter output/audio-128/checkpoint-000128/adapter，新建输出目录。
# 次评测 --suite secondary：slalom上下文24000–24001，共8回合，aligned/missing、八步上限。
$PY -m scripts.training.audio_vision_compare \
  --before output/audio-original-primary --after output/audio-adapted-primary \
  --output output/audio-primary-comparison.json
```

动作仍限制为 W/A/S/D/noop 整窗保持，零生成 token、一次类别预测；合法格式来自安全模板。每个条件报告计划/完成/失败分母、成功/回报、动作成本、音频实际输入掩码和包含采集/执行的延迟。训练期真值留在审计账本，受控 waveform 干预只生成输入，不向模型传目标或改变奖励。无效运行保留，正式评测不选检查点或调整预算。

CPU 比较器重新重放所有实际反馈、重算干预后的 WAV/STFT/RGB 与缺失条件、核对选类和实际动作，再比较两组初始输入、模型、源码、依赖、种子/回合/条件和计划完成数。缺回合、失败状态、条件成绩/动作数/掩码数/四项延迟 P50 的篡改或不相同的实际输入都会拒绝比较；只更改 adapter 是明确的训练干预。次评测动作导致后续画面不同，不要求两模型走过同一路径，只要求初始条件相同。

2026-10-27 的实际双图 GPU pilot 已通过：两步训练和恢复两步均有限、非零梯度，峰值 CUDA 分配约 3542.64MiB。2+2 与连续四步的 24 个 trainable 张量、优化器、Python/Torch/CUDA RNG、输入像素/提示/音频掩码、loss/logits/梯度序列逐位一致，最大参数差 0。前四步没有遇到每第八步的缺失训练，不能把短程恢复证据扩写成全部条件或跨设备确定性。

实际 128 步训练已完成，全部有限非零梯度、两次完整打乱 epoch，逐步复算 CE、音频/RGB 输入与掩码一致。初始 LoRA 参数指纹为 `49cfd9cd954ceff9ab658db18f759e19c641cf5acad5c115352cea1289b4de83`，最终 adapter 权重 SHA256 为 `bfa8ec85542d284f3ef123b64aa3dfd18a09918e2581884d2992c76d6cf2be56`；基座仍为固定的官方 0.8B S3 快照。更新区间总耗时 99.80s，含三次加载的合计 141.48s，峰值 3542.64MiB；另有连续四步恢复控制的额外开销，未混入 128 次预算。

16 次缺失音频更新的标签分布为 A11/D5，112 次有音频更新为 A53/D59，合计仍各 64。缺失是与标签无关的预定 step 规则，但这次随机打乱后的缺失子集并不平衡，不能把它声称为严格平衡的缺失训练。四个 32 步区间平均训练损失为 0.9811、0.7350、0.7150、1.0545；未形成单调下降，不用 loss 证明收益。

## 主评测：没有建立融合收益

原始模型与 adapter 各完成全部 128 个主评测回合，八种条件各 16 回合，无执行失败或遗漏。CPU 比较器重新核对完整原始反馈、条件干预、初始 RGB/状态和实际模型输入，配对检查全部通过。

| 输入条件 | 原始成功 / 平均回报 | LoRA128 成功 / 平均回报 |
|---|---:|---:|
| 对齐 RGB + 音频频谱 | 0/16 / −1.01 | 8/16 / 0.49 |
| 仅 RGB 单图 | 0/16 / −1.01 | 8/16 / 0.49 |
| 显式缺失音频，空频谱 | 0/16 / −1.01 | 8/16 / 0.49 |
| 交换成相反音色的原始 WAV | 0/16 / −1.01 | 8/16 / 0.49 |
| 未来音频，实际被掩码 | 0/16 / −1.01 | 8/16 / 0.49 |
| 原始噪声 WAV | 0/16 / −1.01 | 8/16 / 0.49 |
| 有效静音，present=1 | 0/16 / −1.01 | 8/16 / 0.49 |
| 遮掉 RGB，仅保留音频 | 0/16 / −1.01 | 8/16 / 0.49 |

adapter 在这 128 个回合全部选择 A，原始模型均进入红色危险格。分数提升只表明这次训练把行为推向左右候选方向，**没有超过缺失条件的可识别性上限，也没有建立音频/画面融合或理解收益**。这次实际训练成功和收益失败必须分开报告。

内部 logits 并非完全不受输入影响：交换音频相对对齐的 A−D margin 绝对变化均值 0.046875、最大 0.125，缺失相对对齐均值 0.414063、最大 0.5，但没有改变任何实际动作。不能把“动作不变”扩写成“编码器对音频绝对零响应”；这些小幅 score 变化也不能充作任务理解证据。没有据此修改超参数、延长当前预算或挑选较好检查点。

对齐条件的实际循环 P50 也没有达到 200ms 响应目标：原始模型推理 222.27ms、RGB 观测到动作开始 470.28ms/结束 670.48ms，从音频窗口起点到动作结束 872.15ms；adapter 相应为 234.44、457.10、657.27、858.55ms。这是两次顺序质量评测中的描述量，首次冷请求保留，不能按两列差值宣称延迟优化；采集、文件持久化/核验、双图编码和实际执行的边界都包含在对应整段时钟中。

## 迁移与下一步

绕墙次评测中，两模型各完成 16/16 计划回合（对齐 8、缺失 8）、128 次实际动作，均无执行失败。四组均为 **0/8 成功、平均回报 −0.08、平均八步**，没有建立绕行迁移能力。训练只有 A/D 的相邻目标，绕墙需要 W/S；该已知分布变化没有被此次 LoRA 解决。所有失败、时钟和实际路径保留，未缩小分母。

同一官方 Doom 保存图/提示的完整文本静态探针中，鼠标仍为 `(10,-1)`，但六段按键从原始的 `LB` 变成 `A LB`，动作一致性为 false。它与此次首键 A 偏好相符，但只有一次静态输出，不能当作实际 Doom 遗忘分数或证明原能力保持。探针未进入训练，也未用于调参。

目前已经实现且跑通双图训练、恢复和可审计消融；实际模型完成训练不等于额外模态收益。下一项有价值的问题是区分“固定视觉表示中是否包含足够的联合信息”和“当前动作 token 的 LoRA 目标是否没有提取出来”。可以在新冻结预算、独立新种子下做冻结视觉特征的线性/非线性读出诊断，并将新头和原始解码器明确分开；不能拿它悄悄替换本次负结果或称为修复了此 LoRA。

后续[冻结视觉读出](vision-readout.md)已实际完成四组训练和2048回合，联合 MLP 支持该人工任务上的信息可读性，但不修复本 LoRA。[带回报的 RGB 保留矩阵](retention.md)也已补充本 adapter 的 RGB-only 干预结果：reach 原成功保留4/8、hazard保留1/5，keydoor原模型0/8导致保留分母不具信息性；这些是自有规则上的损伤，不能将静态 Doom 漂移扩写成真实 Doom 遗忘。

后续独立的[平衡四例累积对照](audio-batch.md)在更大预算下重新训练原动作解码器，并完整报告正负条件和标准 RGB 保留。新实验不覆盖这里128步的负结果，也不能把差异单归因于组批。
