# 离线轨迹、RGB 评测与 LoRA 训练

这些入口独立于 controller、SITP、SLAM 和 UE。导入模块不会启动模型、下载或操作桌面。训练使用现有 PyTorch、Transformers 和 PEFT；当前模型后端限定为本地 `qwen3_5` GameScaling 快照，模型文件必须已经存在，禁止自动下载和远程代码。

## 运行边界

先按 AGENTS 核实当前设备、解释器、显卡、进程、官方上游提交及模型快照。驱动显示的 CUDA 支持上限与 PyTorch 编译用的 CUDA 版本分别记录。`requirements/training.txt` 是已选直接依赖；每次运行还记录实际包版本、硬件、模型文件哈希和数据清单哈希。完整传递依赖与本机路径只保存于本地实验目录。

推荐 Linux/WSL Python 3.12。已有可用 PyTorch 时可以在项目专用 venv 中只读复用它，并将新版 Transformers/PEFT 安装到专用 venv。不要升级共享的推理环境：旧版 vLLM 的 Transformers 约束可能与训练环境冲突。本后端不运行 vLLM。

所有模型工作都必须由 `scripts/resource_guard.py` 包装。显式传入两对话共用的锁路径；Windows 与 WSL 使用指向同一文件的路径。独占创建失败返回 75，已有锁保持原样；不要按年龄删锁。包装器持锁直到进程树清理结束，单次上限 1,200 秒。命令不得另建 session、逃出 job 或后台脱离包装器。强制关闭整个主机/WSL 后可能遗留锁，此时先查进程与所有权，不自动接管。

## 轨迹转换

原始格式为每行一个完整 episode 的 JSONL，具体字段与时间约束见 [数据合同](data-contract.md)。数据和图像放在忽略目录 `output/`。显式填写按游戏划分的 JSON：`train`、`validation`、`test` 分别为非空 game_id 数组。

```bash
python -m scripts.training.data \
  --episodes output/capture/episodes.jsonl \
  --splits output/capture/splits.json \
  --data-root output/capture \
  --output output/capture/converted --history-len 5
```

转换器检查来源和训练授权、图像哈希、路径边界、时钟、实际执行时间、六子步动作与终止标记。只保留实际完整执行的动作；建议动作不会进入目标或历史。未完成动作计入逐帧排除统计并重置连续历史，episode 最终结果另存清单。禁止游戏/episode 跨集合，以及解码 RGB 像素完全重复的跨集合泄漏；重新编码的同像素 PNG 也会被拒绝。它不能发现有损压缩、裁剪、重命名来源或谎报游戏标识，真实采集仍需可靠来源登记及场景去重审计。

加载器再次审计原始 episode，并逐行比较转换结果，防止只修改转换数据及其哈希就绕过实际动作合同。原始文件、划分清单和图像必须保留在原位置。输出目录必须不存在，避免静默覆盖冻结的数据版本。

## 可复现的合成冒烟流程

以下为 Linux/WSL shell。变量由当前机器的实际检查结果填写；不从其他设备复制默认路径。`MODEL` 为已有快照目录，`UPSTREAM` 为经过核实的官方仓库，`GPU_LOCK` 指向共用重任务锁。示例假定 `PY` 已指向项目训练 venv 的 Python。

```bash
export PY=.venv-train/bin/python
# 显式设置 MODEL、UPSTREAM、GPU_LOCK 后执行。
$PY -m scripts.training.synthetic --official-repo "$UPSTREAM" \
  --output output/color-smoke --count 8 --seed 20261026

$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  $PY -m scripts.training.evaluate --model "$MODEL" \
  --dataset output/color-smoke/converted --output output/rgb-before \
  --split all --max-samples 8 --max-new-tokens 64

$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 1200 -- \
  $PY -m scripts.training.train --model "$MODEL" \
  --dataset output/color-smoke/converted --output output/lora-pilot \
  --steps 4 --save-every 2 --rank 8 --lr 0.0001

$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 1200 -- \
  $PY -m scripts.training.train --model "$MODEL" \
  --dataset output/color-smoke/converted --output output/lora-resumed \
  --resume output/lora-pilot/checkpoint-000004 --steps 8 --save-every 2

$PY scripts/resource_guard.py --lock "$GPU_LOCK" --seconds 600 -- \
  $PY -m scripts.training.evaluate --model "$MODEL" \
  --dataset output/color-smoke/converted --output output/rgb-after \
  --adapter output/lora-resumed/checkpoint-000008/adapter \
  --split all --max-samples 8 --max-new-tokens 64
```

每个重任务还可指定 `--latest-start 2026-10-27T07:40:00+08:00`；这个选项放在包装器的 `--` 之前。截止时间随实际任务更新。

合成生成器创建红/蓝色块到 W/S 的静态规则样本，以官方 prompt 模板为基础，三个集合使用不同人工 game_id 和唯一图像。时间与动作都是生成值，没有执行器或环境反馈。相同规则和视觉分布用于三组，不能用它证明跨游戏泛化；每组不相邻的唯一图像也不能替代真实多游戏评测。

## 训练、恢复与评测含义

LoRA 只更新文本 full-attention 中的 q_proj/v_proj，默认 rank 8、alpha 16、dropout 0；视觉骨干和其他参数冻结。batch size 1，AdamW，梯度范数上限 1。损失只覆盖当前 assistant 的动作补全；系统提示、RGB token、历史动作均被掩码。仅计算动作区间的 logits，避免为整段提示分配大词表 logits。当前官方 GameScaling 后端只接受 RGB；下面的小模型音频试验是独立冒烟，不包含 RL。

检查点包含可训练权重、优化器、Python/PyTorch/CUDA RNG、步数与模型/数据/超参数/版本身份。完成后写哈希清单；哈希或身份不匹配拒绝恢复。可复现数据顺序按 seed 和 epoch 生成。标准 PEFT adapter 随检查点保存，基座权重不复制。CPU 测试验证下一步恢复结果与不中断训练相同；GPU 恢复证据见下面的固定硬件对照，不能推广到其他环境。

训练默认启用进程级确定性：CUDA handle 创建前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，开启 PyTorch deterministic algorithms 和 cuDNN deterministic，并关闭 cuDNN benchmark。设置写入检查点身份，不能在恢复时静默切换。旧版未记录这些设置的检查点需显式 `--no-deterministic`；此模式不承诺逐位重现。确定性模式遇到不支持的 GPU kernel 会报错，而不是悄悄回退到非确定性训练；不同设备、驱动和框架版本仍需独立复现。

评测输出逐样本生成、严格动作合法率、动作精确匹配和按目标 token 加权的损失。所有失败保留在分母。静态动作匹配不是游戏任务成功。适配前后需固定同一份验证/测试集；另需有独立原能力保持集才能报告遗忘，合成 W/S 结果不能代表原模型游戏能力保持。

计时区分磁盘图像解码、processor 编码、CPU→GPU、生成推理和解析；损失计算不计入决策耗时。截图、传输、排队和动作执行没有对应环节时输出 null，不伪造为 0。输出的 offline_total 是保存图到解析结束的本地时间，不能称作真实采样到动作执行延迟；小样本 P95 也不能估计线上尾延迟。

`python -m scripts.training.saved_probe --model "$MODEL" --image <saved-image> --prompt <utf8-prompt> --source-description <provenance> --output <new-local-directory>` 可单独测试已有保存图，无需伪造实际动作轨迹。适配后传入 `--adapter <local-adapter> --reference <before/result.json>` 比较相同图像/提示的动作一致性；图像/提示/生成参数变化会拒绝比较。没有任务真值，输出变化只代表静态输出漂移，不能称作遗忘或能力保持。模型标记不一定是单个 special token，停止条件按完整 `<|action_end|>` 字符串处理。

## 首次本地推理证据

固定官方上游 `5c8c0322868171a53c9e20dface316b8afe9120d` 与模型快照 `1b89640dcf2ef51fb53ad9338a0725e9622dd200`。2026-10-26 的本地 pilot 使用 0.8B S3、bf16、Transformers 5.14.1 / PyTorch 2.10.0 cu128、RTX 5070 Ti Laptop、256×256、零历史、关闭思考、greedy、64 token 上限；torch 线性注意力回退路径，未安装 FLA 快速内核。

三个合成集合各取前两张图，6/6 输出格式合法、0/6 满足指定 W/S 规则，均输出 no-op；测试两张图 offline_total 中位数约 2,367ms、最大值约 2,382ms，峰值已分配 CUDA 显存约 1,779MiB。这个很小的 pilot 只验证实际权重与推理链路，不能估计泛化或线上 P95。另对上游 Doom 保存图完成一次零历史输出，生成合法鼠标/开火动作；没有执行动作或任务真值。

首次 LoRA 合成短跑实际完成 4 个优化器步、保存第 2/4 步检查点，耗时约 17.4 秒（包含加载），峰值已分配 CUDA 显存约 5,743MiB；每步损失及梯度范数均为有限数。这个结果证明实际基座权重上的反向传播和保存链路已执行，不代表 GameScaling 游戏能力改善。全部原始输出、模型哈希清单、截图路径、adapter 和日志保留本地。

随后从第 4 步的真实 GPU 检查点恢复，沿相同数据次序完成至第 32 步（本次约 41.3 秒含加载，峰值约 5,743MiB，319,488 个可训练参数）。使用与适配前相同的每组两张冻结图：

| 合成集合 | 静态动作匹配，前 → 后 | 目标 token 加权损失，前 → 后 |
|---|---|---|
| train（2 张） | 0/2 → 2/2 | 0.08719 → 0.000137 |
| validation（2 张） | 0/2 → 2/2 | 0.10753 → 0.000230 |
| test（2 张） | 0/2 → 2/2 | 0.08752 → 0.000139 |

这些指标对应合成颜色规则。它们证明实际预训练权重上的 LoRA 能学习这个冒烟目标，不能证明真实游戏收益、未见游戏泛化或原能力保持。当前没有带授权/真实反馈的游戏轨迹，因此没有开展真实游戏适配训练。

适配前后的同一 Doom 保存图 probe 均输出合法开火动作，但鼠标位移从 `(10,-1)` 变为 `(0,0)`，因此严格动作一致性为 false。该静态输出已发生漂移；没有环境任务真值，不能把它解释成遗忘，也不能声称原能力保持。此 probe 不参与 LoRA 训练。

初始未开启确定性的 4+28 步与连续 32 步 GPU 对照没有逐位重现：数据/模型/随机状态一致，但最大 adapter 参数差约 0.000980。进一步检查发现首步前向损失相同、首步反向梯度已不同，差异发生在任何恢复之前。启用确定性设置后，两次独立 2 步控制实验的损失和梯度范数一致；这一区别说明仅固定 seed 和保存 RNG 还不够。原始失败和成功控制记录均保留本地。

确定性模式下再比较真实 0.8B GPU 训练的 **2+2 步恢复与连续 4 步**：24 个可训练张量、完整优化器状态、Python/PyTorch/CUDA 随机状态全部逐位一致，最大参数差为 0。峰值 CUDA 分配约 5,791MiB。该证据限定在相同模型/数据/超参数/框架和硬件上的短程复现；它不将先前非确定性 32 步结果改写成确定性结果，也不证明跨设备复现。

## 小模型音频融合冒烟

```bash
$PY -m scripts.training.audio_smoke --output output/audio-fusion-smoke \
  --seed 20261026 --steps 200
```

该命令仅用 CPU，内部预算 60 秒，不启动 GPU 或桌面。先在合成红/蓝 RGB 特征上训练小型分类器，再冻结 RGB 分支；音频为 200ms、16kHz 的本地生成正弦波，RGB 为与目标独立的灰度干扰图。固定 FFT 八频带能量、RMS、过零率共十维特征，训练 418 参数的 MLP 残差融合分支。掩码为 false 时残差严格归零；没有训练任何 GameScaling 音频编码器或 Qwen 模型。

合成游戏 ID、episode 和像素/音频文件哈希按集合隔离；三个集合仍共享同一人工频率规则，因此不能证明真实音频或跨游戏泛化。生成媒体、逐样本合同、数据哈希、模型 state 和结果全部保留本地。

2026-10-26 seed 20261026 的首次 CPU 运行共 200 步，约 15.1 秒，训练损失由约 0.8007 降至最后一步约 0.000486。冻结测试集 64 个样本的分类结果：

| 输入条件 | 合成分类准确率 |
|---|---:|
| RGB 单独 | 48.44% |
| RGB + 对齐音频 | 100% |
| 音频显式缺失 | 48.44% |
| 相反类别音频交换 | 0% |
| 未来音频窗口（全部被掩码） | 48.44% |
| 高斯噪声替换音频特征 | 56.25% |

最后一行是十维特征的数学扰动，不是原始波形噪声的鲁棒性实验。128 个经过变换的 RGB 预训练颜色样本在缺失音频时输出保持一致；这是冻结分支和残差掩码的设计不变性检查，不是独立游戏能力保持集。此试验验证了因果对齐、掩码和实际可学习融合路径，不能声称官方模型理解音频或真实游戏能力提升。

## 验证与发布

```bash
python -m unittest discover -s tests -p 'test_training*.py' -v
python -m unittest discover -s tests -p test_resource_guard.py -v
python -m unittest discover -s tests -p test_audio_contract.py -v
python scripts/validate_project.py
python scripts/sync_engineering.py
python scripts/check_repository.py
git diff --check
```

恢复测试需要可选 PyTorch；控制器完整测试以 Windows 为准。`experiments/*.json` 仍是设计配置，静态检查不启动上述独立 CLI。权重、adapter、真实轨迹、截图、原始输出、私人日志与本机绝对路径全部留在忽略目录。公开 PR 只包含源码、合成测试和通用命令。
