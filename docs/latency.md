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
