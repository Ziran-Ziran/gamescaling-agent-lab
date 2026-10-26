# 固定 RGB 策略的推理执行对照

本入口比较同一个本地基座快照与 LoRA adapter 在 PEFT 未合并和 `merge_and_unload(safe_merge=True)` 两种执行方式下的保存图推理。它不训练新权重，不运行游戏或桌面输入。使用[官方 PEFT 合并接口](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraModel.merge_and_unload)，bf16 合并可能带来舍入差异，必须检查数值和动作，不能假设逐位等价。

```bash
# PY、MODEL、GPU_LOCK 由当前设备实际检查结果显式设置；ADAPTER 为本地检查点。
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  $PY -m scripts.training.latency --model "$MODEL" --adapter "$ADAPTER" \
  --dataset output/color-smoke/converted --output output/lora-merge-benchmark \
  --samples 2 --repeats 3 --max-new-tokens 64
```

两份模型同时驻留同一 GPU，固定存储快照、adapter 文件哈希、bf16、输入、零历史、空思考、greedy 和 token 上限。预算不足时不要启动第二份模型；先检查可用显存。预热两种路径各一次并单独保存，不把冷启动静默混进平均值；正式请求按 pair 交替先后顺序。输出保留所有请求、失败、生成 token 数和动作匹配。

逐阶段记录磁盘 RGB 解码、processor 编码、CPU→GPU、模型生成、解析及 offline_total。没有截图、网络、排队测量或执行器，相关字段为 null。200ms 只作为保存图处理预算；其超预算率不是实际在线帧过期率。任务真值仅为冻结合成规则，匹配率不能当游戏任务成功。

对照额外测量 teacher-forced logits 的最大/平均绝对差及目标损失。只有两种路径在这些静态样本上均无错误、动作合法且匹配真值，才给出质量门控的成对中位加速比；任一失败都会抑制该比值。输出文本一致率把失败当不一致。小样本 P95 使用 nearest-rank，只描述本次观测，不能估计线上尾延迟。

所有原始输出、数值差、模型和机器清单保留本地。该对照不证明真实游戏低延迟、跨游戏任务保持或 200ms 部署可行；部署前还需要实际采样到执行、在线成功率、奖励和反馈验收。

## 首次固定硬件观测

2026-10-26，使用 [训练说明](training.md) 中同一个 0.8B S3 快照、32 步合成适配 adapter、RTX 5070 Ti Laptop、bf16、Transformers 5.14.1 / PEFT 0.21.2。冻结测试前两张 RGB 图，正式 6 对请求（每图 3 次）；两个模型全程驻留，warmup 单独保存。基座和 adapter 内容哈希相同，合并使用独立模型副本。

| 指标 | PEFT 未合并 | 安全合并 |
|---|---:|---:|
| 正式请求 / 错误 | 6 / 0 | 6 / 0 |
| 合成动作匹配 | 6/6 | 6/6 |
| offline_total P50 | 2,071.7ms | 1,991.5ms |
| offline_total 经验 P95 | 2,259.7ms | 2,193.5ms |
| inference P50 | 2,055.0ms | 1,974.9ms |
| 200ms 处理预算超限 | 100% | 100% |

成对中位加速比为 **1.029**，输出文本一致率 100%，峰值总 CUDA 分配约 3,444MiB。这只是一次小样本观测，没有置信区间或线上任务试验，不能据此断言稳定的性能提升；更没有达到 200ms 目标。主要瓶颈仍是模型生成，单纯合并这个小 adapter 的收益有限。

两图 teacher-forced logits 最大绝对差分别为 0.34375 和 0.375，平均绝对差约 0.03555 / 0.03273；目标损失分别从 0.000164 / 0.000115 变为 0.000161 / 0.000112。数值并不逐位等价；本次两个合成目标动作保持，不能推广为其他画面或任务保持。因此没有把合并路径作为默认部署策略。

## 限定动作空间的首键解码

自有网格课程中，很多动作都是整块持有一个方向。`scripts.training.action_head.FirstKeyBackend` 保持原模型和 adapter 权重，向已经空思考的 RGB 提示追加固定的鼠标零输入/动作首组前缀，然后一次 forward 从 W/A/S/D/空组五个候选 token 中选最大 logit，再构造六组相同按键的安全动作。它不逐 token 生成整段文本，记录 `generated_tokens: 0`、`predicted_tokens: 1` 和模板来源。

这改变了策略的动作空间：没有鼠标、攻击、斜向、混合按键或块内时长变化，不能作为官方通用协议的替代。模板格式合法是构造保证，不是模型格式能力改善。候选使用真实 tokenizer 校验尾部往返、前缀严格延伸、各类唯一单 token；任何不兼容直接报错。文本 attention mask 补 1、多模态类型 mask 补 0，保留图像 tensor/grid，原 batch 不变。

每个选择保留五类原始 logits、全词表最大 token、其是否属于候选集，以及明确命名为 `restricted_softmax_not_calibrated` 的归一化分数。该分数只在这五类之间归一化，不是校准后的置信度；全词表最大 token 不在集合中时仍是受限选类，不能声称与自由生成等价。

先冻结新的任务集合，两个独立新输出目录分别运行相同权重的实际回合：

```bash
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  $PY -m scripts.training.rollout --model "$MODEL" --adapter "$ADAPTER" \
  --decoder full --output output/decoding-full --episodes-per-game 4 --seed-start 6000 \
  --max-steps 8 --history-len 0 --max-new-tokens 64
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  $PY -m scripts.training.rollout --model "$MODEL" --adapter "$ADAPTER" \
  --decoder first-key --output output/decoding-head --episodes-per-game 4 --seed-start 6000 \
  --max-steps 8 --history-len 0 --max-new-tokens 64
$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  $PY -m scripts.training.decoding_benchmark --model "$MODEL" --adapter "$ADAPTER" \
  --rollout output/decoding-full --output output/decoding-paired --repeats 3 --max-new-tokens 64
```

配对 benchmark 只取完整源回合的初始 RGB，重新审计来源反馈，要求模型、adapter、软件/硬件与源回合相同。一个 GPU 常驻模型分别调用两种解码，各一次 warmup 单独保存，正式 pair 交替先后。主时延为保存图解码到处理、模型、选类、序列化及解析过滤的完整 `offline_total`；两种 `inference` 子项包含最终文本解码的边界不同，不直接据此取主要加速比。

记录所有失败和请求，错误阻止给出成对加速比；动作不一致保留并单独报告，不能以速度代替任务质量。200ms 仍只是处理预算，保存图不含渲染/传输/排队/执行。实际回合另计真实 200ms 执行和奖励，世界仍在推理时暂停。正式计时期间不并行运行 CPU 测试或其他重任务。

### 首轮受限解码观测

2026-10-27，在相同 RTX 5070 Ti Laptop、bf16、0.8B S3 基座和[128 步玩具任务 adapter](feedback.md) 上，固定权重、零历史与 greedy；没有为解码实验训练新权重。先验证真实多模态 forward，再冻结 reach 6000–6003、hazard 7000–7003、keydoor 8000–8003，两种策略分别完成全部 12 回合，无运行失败。各规则都是 4/4 成功；reach/hazard 平均回报 0.99、一步，keydoor 平均回报 0.965、3.5 步。两种路径各执行 22 个动作，回报和步数逐回合一致。

正式计时取这 12 张唯一初始画面，各重复三次，共 36 对；同一 GPU 常驻模型，各一次 warmup 排除并保存。所有 pair 的模型、adapter、RGB 和处理器一致，两种路径的建议及实际过滤后动作全部一致，完整生成在这批画面中恰好都落在五类限制内。

| 指标 | 完整文本生成 | 受限首键预测 |
|---|---:|---:|
| 正式请求 / 错误 | 36 / 0 | 36 / 0 |
| offline_total P50 | 1,991.5ms | 163.4ms |
| offline_total 经验 P95 | 2,236.0ms | 197.9ms |
| 200ms 处理预算超限 | 36/36 | 1/36 |

成对中位处理加速比 **12.103**，峰值 CUDA 分配约 1,717MiB。它是 12 张画面的重复观测，不是 36 个独立任务样本；没有统计显著性或线上尾延迟结论。受限路径仍有一次超过 200ms。模板构造并没有提升格式学习，也没有证明其他动作或画面保持。

这些新种子全部成功，不应与之前适配对照的 keydoor 3/4 混成能力提升：新集合上的完整解码同样 4/4。小样本、共享简单规则与推理暂停限制仍在；真实执行另花约 200ms，不能把约 163ms 的处理时间说成采样到动作完成或持续实时游戏验收。完整解码仍为默认，首键路径只显式用于可接受其动作限制的实验。

实际回合还存在 PNG 落盘、逐帧日志和动作间隔，负载节奏与连续配对计时不同。以下为各自 22 个动作的回合观测，首请求未剔除，两个运行顺序执行；不能把它们当作上述交替配对分布：

| 实际回合 P50 | 完整解码 | 首键路径 |
|---|---:|---:|
| 渲染 | 0.25ms | 0.32ms |
| PNG 编码/写入/哈希 | 22.9ms | 29.7ms |
| 模型解码/预测子项 | 2,265.6ms | 220.5ms |
| 采样到实际动作开始 | 2,306.4ms | 278.0ms |
| 实际执行窗口 | 200.2ms | 200.2ms |
| 采样到实际动作结束 | 2,506.6ms | 478.2ms |

各分项的中位数不可直接相加。回合中两条路径的采样到动作开始都为 22/22 次超过 200ms；处理/日志开销与执行尚未形成 200ms 整链路。配对与回合耗时差异的具体硬件/负载原因未做归因实验，不能自行归结为某个内核或电源状态。仍需连续世界的任务与过期观测验收。

## 冻结视觉头的精确频谱缓存

[频谱特征缓存](spectral-cache.md)对另一个任务专用视觉头进行了固定权重/硬件的96对离线请求、32对实际回合测量。像素与编码器身份精确匹配、音频因果检查在缓存之前，RGB始终重编码；所有配对特征、logits、动作一致，实际两组均32/32成功。

离线主处理 P50 为131.47→94.70ms、配对中位时间比1.405；实际观测到动作开始 P50 为210.69→195.02ms、配对中位比1.080，但实际 P95 为243.26→258.71ms，缓存组仍12/32超过200ms。实际到动作结束 P50 为395.26ms，传感器窗口开始到动作结束596.89ms。保留尾延迟变差的结果，不用离线速度替代整链路改善。这不是前述原语言解码器的缓存，也没有真实游戏音频命中率或持续实时验收。

## 保存计时的 CPU 排队重放

`scripts.training.queue_replay` 用上述完整36对原语言解码计时做离散事件重放；不再启动模型。先重算全部72请求的配对摘要，检查输入/交替顺序、无缺失或额外请求、无错误及动作一致，并核对当时四个源码文件的哈希。保留 config、requests、summary 字节身份和当前重放源码身份；实际计时的模型、adapter、硬件记录随报告保留，不冒充本次 CPU 模拟环境。

虚拟合同在首个结果前冻结：36观测每200ms到达；单 worker 串行执行“原记录 offline_total 处理→动作开始→固定200ms动作→动作结束→下个处理”。FIFO 保留全部队列；latest 仅选当前已到达的最新帧，丢弃其他待处理帧，不打断已开始的处理或动作。到达恰逢释放时可选，输入结束后仍排空最后一帧。没有预测未来服务耗时来选帧。

开始期限200ms与结束期限400ms在固定200ms动作下数学等价，是同一期限的两种时钟表达。准时分母始终是全部36个到达，丢帧不算准时；分位数仅基于实际选中的虚拟事件。以下为同一已测硬件/模型 trace 的条件模拟结果，**不是新的实际在线延迟**：

| 原计时路径 / 调度 | 处理 / 丢帧 | 排队P50 / P95，ms | 采样到开始P50 / P95，ms | 准时开始 / 全36 | 排空结束，秒 |
|---|---:|---:|---:|---:|---:|
| 完整 / FIFO | 36 / 0 | 34123.69 / 67755.44 | 36096.13 / 69912.56 | 0/36 | 79.356 |
| 完整 / latest | 5 / 31 | 38.41 / 1813.82 | 2038.41 / 4056.79 | 0/36 | 11.257 |
| 首键 / FIFO | 36 / 0 | 2885.60 / 5678.45 | 3046.08 / 5850.27 | 1/36 | 13.206 |
| 首键 / latest | 21 / 15 | 100.01 / 199.82 | 272.81 / 366.18 | 4/36 | 7.693 |

结束年龄的每个事件及对应分位数均比开始增加200ms，准时分子相同。完整解码 latest 的最后一帧在到达停止后仍等待当前动作，所以排队尾部可以超过一个到达间隔；没有删掉该尾部来改善 P95。首键 latest 虽减少积压，仍只有4/36满足这个假设期限，丢帧也可能损害决策与任务收益；重放没有模拟状态分叉，不报告游戏成功率。

```bash
# 复核既有计时用的源码快照；归档代码只用于读取哈希，不执行。
mkdir -p output/decoder-source
git archive c5e3bf07a2d3fe2733138726e6211adea43b4d93 \
  scripts/training/decoding_benchmark.py scripts/training/action_head.py \
  scripts/training/hf_backend.py scripts/training/rollout.py \
  | tar -xf - -C output/decoder-source
python -m scripts.training.queue_replay \
  --benchmark output/decoding-paired \
  --source-root output/decoder-source/scripts/training \
  --output output/queue-replay.json
```

该示例的归档提交仅对应本次已保存计时；新 benchmark 应使用自己 config 记录的源码快照。原 trace 配置 SHA256 为 `ddea601226d3fe324819fe33c2da023ab746c46abb6f36a6b8c23feb124caad1`、请求 `9d684bdaa6dbe4db12e805702297a1831c6492839712c7c51272c0376957ce8e`、摘要 `4a0aa13723ed7cdd2760b7917caa96160b0d5d1633725c0f8cf7c5da38bc0fac`。当前工作树之后已有两个模块修改，首次重放因身份不符在输出前拒绝；从 Git 取到四文件均精确匹配的快照后才得到上述结果，旧计时没有改写。

服务成本被假定为独立于新节奏的已有实测值，截图、网络、新日志与连续世界未模拟，真实运行还会受热状态、缓存和并发影响。这里没有新 GPU 测速、因果加速比或持续实时验收；下一步应在有真实反馈的连续环境测量最新帧策略的任务损伤、实际过期率与覆盖率。

## 已保存候选分数的事后诊断

收尾时另对[RGB重放保留套件](audio-replay.md)的四策略全部64个单步回合（每策略reach8/hazard8）复算五类logits、选类及保存的归一化分数；不纳入多步keydoor，不产生新推理或动作。首个诊断结果前固定0.5/0.9/0.99三个描述阈值，全部行保留。以下每格为“分数达阈值的行数 / 其中成功 / 其中失败”，总分母每策略均16：

| 固定策略 | 全部任务成功 | ≥0.5 | ≥0.9 | ≥0.99 |
|---|---:|---:|---:|---:|
| original | 6/16 | 5 / 3 / 2 | 0 / 0 / 0 | 0 / 0 / 0 |
| parent200 | 6/16 | 16 / 6 / 10 | 5 / 4 / 1 | 2 / 2 / 0 |
| audio-only-control | 6/16 | 16 / 6 / 10 | 4 / 4 / 0 | 1 / 1 / 0 |
| rgb-replay | 14/16 | 16 / 14 / 2 | 16 / 14 / 2 | 16 / 14 / 2 |

重放的两个已知hazard死亡回合，51000选择S：logit26.5对W15.875，top2间隔10.625，候选分数0.99997553；51004选择S：27.25对W17.625，间隔9.625，分数0.99993389。大候选间隔和高归一化分数仍与具体错误动作并存。

这是已知失败的事后描述，不是新发现、独立校准集或阈值安全性验证；没有拟合温度、挑阈值或执行回退策略。候选分数只在五个token内归一化，不能直接解释为任务成功概率或部署置信度。小样本、共享布局及不同适配器分数尺度也不支持用该表排名置信度；若研究拒绝/重试策略，需另冻结校准和连续任务反馈评测。
