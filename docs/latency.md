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
