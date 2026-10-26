# 2026-10-27 夜间研究结果

本轮已经在真实 GameScaling 0.8B 权重上执行 LoRA 反向传播、保存、恢复和冻结评测。收益与损伤来自自行生成的静态颜色或暂停式网格任务；没有完成商业游戏适配、广泛跨游戏泛化或持续实时部署验收。项目不依赖 SITP、SLAM 或 UE5。

## A–F 实际完成范围

| 项目 | 已实现并执行 | 仍未验收 |
|---|---|---|
| A RGB 基线 | 固定官方快照、完整生成/受限首键、零历史；保存图与实际玩具回合的输入、动作、回报、时延 | 有授权的真实游戏任务基线 |
| B 轨迹与划分 | 实际动作对齐、终止/截断、来源声明、完整执行与历史、按游戏分组、精确解码像素泄漏；原始来源逐行重审；CPU搬迁见证 | 可靠真实来源登记、近重复/有损变化和场景族去重 |
| C SFT/LoRA | 真实基座加载、动作区间损失、24张量分类适配或完整动作 SFT、AdamW、检查点/RNG恢复、标准adapter导出 | 跨硬件逐位一致、任意模型后端 |
| D 有界适配 | 静态合成冒烟、实际网格 SFT128、一步奖励学习128、双图LoRA128及两组200、两组追加64；完整冻结对照 | 真实游戏收益和多种子算法排名 |
| E 额外模态 | 自行生成PCM16音频、同钟因果窗口/缺失掩码、固定STFT→第二图；实际适配与交换/移除/缺失干预 | 原生波形编码、真实游戏音频语义 |
| F 时延与调度 | 相同硬件/权重的交替解码、精确频谱缓存对照；完整计时trace的CPU队列模拟 | 连续世界下成功率/P95/过期率共同改善 |

“真实权重训练完成”与“真实游戏能力改善”分别验收。以下各套件使用不同冻结种子与分母，不能跨表合并成功率。

## 主要训练与任务结果

### RGB 与反馈基础

静态颜色规则在32步实际LoRA后，六张固定保存图动作匹配由0/6到6/6。这只证明管线冒烟。另一个实际128步文本SFT仅训练reach的64条专家转移，随后按冻结种子运行各四局：

| 自有规则 | 基座成功 | SFT128成功 | 平均回报，前→后 |
|---|---:|---:|---:|
| reach | 2/4 | 4/4 | 0.4475→0.9900 |
| hazard | 0/4 | 4/4 | −0.0800→0.9900 |
| keydoor | 0/4 | 3/4 | −0.0800→0.7050 |

这是共用简单引擎/视觉编码的玩具任务，世界在推理时暂停；失败的keydoor回合保留。完整动作格式是官方六子步200ms合同，受限首键另有显式动作空间约束。详见[反馈与SFT](feedback.md)。

实际一步bandit与分类SFT各128更新，冻结主套 reach/hazard 总成功相同，所有32个主套动作一致；两者仍损失四个原hazard成功。次要keydoor分别7/8与6/8，样本与交互/标签成本不同，不证明奖励算法更优；这不是顺序RL。详见[反馈学习](feedback-learning.md)与[独立RGB保留](retention.md)。

初次未启用确定性的GPU恢复对照最大张量差约0.000980，失败记录保留。开启确定性后，实际2+2对连续4步逐位一致；本轮后续两组追加训练还完成4+4对连续8步，跨过第8步音频缺失边界，24张量、优化器、Python/Torch/CUDA RNG、实际输入/logits/CE/梯度最大差0。不能推广为跨设备确定性保证。

### 人工音频融合及保留损伤

首次原解码器音频LoRA128在八条件上始终输出同一动作，每条件8/16，是负结果。之后的冻结视觉新头有单独对照；它与原解码器适配不能混为同一个模型。

冻结视觉读出另实际训练四个CPU头各200更新，并完成四头、两套、八条件共2048个实际回合。联合MLP两套对齐各32/32、交换各0/32，单模态头对齐各16/32。它支持人工联合表示的可读性，但跳过原语言解码器，不能称为LoRA已学会融合；详见[独立读出试验](vision-readout.md)。

再从相同原始基座训练两组平衡LoRA，各200更新、800标签，主路径之外的恢复控制单列。两个新适配器在迁移/IID两套各16观测的对齐音频上均16/16、交换均0/16、其他六条件均8/16；四策略共1024个实际回合完整审计。干预支持这个人工任务中的联合信息利用；两种组批同分，没有算法优胜结论。另160局标准RGB仅保留原reach成功3/7、hazard成功2/6，不能用融合收益覆盖损伤。详见[平衡音频训练](audio-batch.md)。

最后固定上述paired200为共同父状态，两组各追加64更新：共同音频每组128标签；RGB重放另加128 RGB标签，计算成本更大。主路径墙时约166.05/230.07秒、峰值显存均3540.70MiB。主路径和额外连续8步恢复控制分别记账。

三策略音频矩阵240回合，加四策略标准RGB80回合，共320实际回合、413动作，全部逐例审计：

| 最新套件 | parent200 | 音频对照64 | RGB重放64 |
|---|---:|---:|---:|
| 音频迁移 aligned | 8/8 | 8/8 | 6/8 |
| 音频IID aligned | 8/8 | 8/8 | 8/8 |
| 标准RGB reach | 4/8 | 4/8 | 8/8 |
| 标准RGB hazard | 2/8 | 2/8 | 6/8 |
| 标准RGB keydoor | 0/4 | 0/4 | 4/4 |
| 基座原reach成功保留 | 2/4 | 2/4 | 4/4 |
| 基座原hazard成功保留 | 0/2 | 0/2 | 0/2 |

基座在这套RGB上的分母为reach4/8、hazard2/8、keydoor0/4。重放虽提高hazard总成功，却仍损失两个原成功，且结局从父模型/对照的超时变为踩hazard死亡。keydoor基座无成功，保留比例为null；4/4是小样本任务收益，不是原能力保持。迁移音频同时损失两例；新增RGB目标并未解决整体遗忘。音频每套只有两个独立上下文，布局/音调组合不能当作八个独立场景。完整条件、回报、损伤种子与命令见[RGB重放对照](audio-replay.md)。

## 时延结果和测量边界

以下为固定模型/硬件的实测；不同分支不能拼成一个端到端速度结论。

| 对照 | 主要观测 | 限制 |
|---|---|---|
| 合并LoRA，6对 | offline P50 2071.7→1991.5ms；成对中位比1.029 | logits不逐位等价；均超200ms |
| 完整生成→首键，36对 | offline P50 1991.48→163.35ms；成对中位比12.103；动作一致 | 12张图×3次；五类整块动作，无鼠标/混合按键；没有截图/排队/执行 |
| 同一解码器实际回合 | 首键采样到开始P50 277.99ms，到结束478.21ms | 两条路径均22/22开始超200ms；顺序运行非速度配对 |
| 冻结视觉头的频谱缓存，96离线对/32实际对 | 离线成对中位比1.405；实际1.080，两组32/32成功 | 实际开始P95反而243.26→258.71ms；缓存组仍12/32超200ms |

CPU队列重放另有明确的虚拟假设：36次每200ms到达，处理后串行占用固定200ms动作。首键FIFO准时1/36；latest准时4/36但丢15帧。其分位数是所选虚拟事件，不是新测的在线延迟；没有状态分叉或任务收益。详见[时延、动作限制与调度](latency.md)、[缓存的完整对照](spectral-cache.md)。

## 版本与本地产物身份

本轮核实设备为RTX5070Ti Laptop 12GiB；Windows主机上的WSL Ubuntu、Python3.12.3、PyTorch2.10.0+cu128、Transformers5.14.1、PEFT0.21.2。驱动CUDA支持上限与torch编译版本分别登记；未安装FLA快速内核。基座bf16、24个分类LoRA可训练张量float32。下列为本次保留产物的内容哈希，不提供权重下载。

| 产物 | 固定版本/ SHA256 |
|---|---|
| 官方上游提交 | `5c8c0322868171a53c9e20dface316b8afe9120d` |
| GameScaling-0.8B-S3 snapshot | `1b89640dcf2ef51fb53ad9338a0725e9622dd200` |
| 基座 model.safetensors | `d553c0796833691e32277a937db649eaac39bae9a0028a63c964c12a261c2dfa` |
| 文本SFT128 adapter | `cc8a220cb138a05971e7d593547c2a1553ecfe0f3fc42a740d773e22b57aa174` |
| paired200 adapter | `e017302b535a028872b5899493259ee7429020ffb5da53b0933f1736e6f7fa44` |
| 追加音频control64 adapter | `c98497aa5a41469693fe6203d833a60cd3aad34a391467387b6f66d7cce94076` |
| 追加RGB replay64 adapter | `e0c7836632bef4aa149bc5a351a193856060e7792e4a289c2b03af72c220699e` |
| 最新cohort，评测前冻结 | `f87b8bf24af3c5b6850789e28a5e818d71d72d029f3b69e71cbc2e0490003e4b` |
| 音频语料RGB manifest | `19702bcd209526bab4793ffa4de9e89398116f7993dbea01aafa9705251dede5` |
| 音频索引manifest | `d99732d3c7827fb6721e30055da6fa5e491e544d4d8259f88ef1fba840324949` |
| RGB reach专家语料manifest | `d58ddde6623581a86ef59502f1e7d32253cc4d4f6bc5a40a40b6b975f3b516d2` |

原始轨迹、频谱/特征、模型/adapter、恢复状态与私人日志只留本地。来源许可为自行生成/采集的玩具资产；没有导入真实游戏训练轨迹或蒙德素材。各研究文档记录更细的数据/处理器/源码/依赖身份，实际运行应使用其固定源码版本。

## 可运行入口与复现

先检查当前设备、进程、工具、显卡、模型位置与共享锁。`PY`、`MODEL`、`GPU_LOCK`、`UPSTREAM` 必须按当前设备显式设置；不复制另一台电脑的默认路径。项目隔离环境使用已存在的本地权重，不自动下载模型。

```bash
python -m unittest discover -s tests -v
python scripts/validate_project.py
python scripts/sync_engineering.py
python scripts/check_repository.py
git diff --check
```

主要入口与完整命令：

- [RGB转换、训练、评测与恢复](training.md)：`scripts.training.data`、`train`、`evaluate`；先做小批量短跑与恢复对照，再增加预算。
- [实际反馈和专家采集](feedback.md)：`headless`、`rollout`；反馈真值不进入学生观测。
- [原解码器音频训练](audio-batch.md)及[同父重放](audio-replay.md)：包含控制、恢复、训练、cohort冻结、全部质量/保留矩阵和CPU审计命令。
- [时延](latency.md)：`decoding_benchmark`、`queue_replay`；模型实测必须持锁，CPU虚拟重放不启动GPU。
- [跨位置语料重审](training.md#搬迁已有-rgb-数据)：`relocate`；实际同机Windows/WSL80行一致，新清单与旧检查点身份分开。

重任务必须由 `scripts/resource_guard.py` 包住整个进程树、使用共享锁并限制每次不超过1200秒。锁忙时不杀其他会话或按年龄清锁。本轮标准RGB首执行器因锁忙退出75、零回合；后续独立恢复执行器审计并跳过已完成前缀，最终完整完成80回合，失败证据保留。不得把中断/running或只写配置称为完成。

训练与研究专属审计需要保留的本地产物；仓库不附原始权重/媒体。重新采集会改变时间、来源和清单身份，搬迁会改变位置清单身份，均不能冒充旧cohort。CPU测试不能替代实际GPU训练，确定性恢复不能替代任务收益，任务收益不能替代泛化。

## 验证与后续

最终代码阶段完整Windows套件302项、45项明确跳过（主要是未安装torch及一个符号链接权限测试）；相应torch/恢复/新训练审计测试已在隔离WSL环境执行，符号链接拒绝也在WSL实测。所有对应里程碑均通过项目配置、14文件工程同步、仓库白名单及diff检查，并经过独立源码/证据复算和三项CI。详细平台边界见[开发说明](development.md)。

另以固定上游包补跑官方接口离线测试9项全部通过，本轮320回合的413个实际动作全部通过官方严格解析及规范文本往返；核对了 `GameScalingPolicy`、`PolicyConfig`、`Turn` 和图像编码接口形状。该检查没有访问HTTP模型服务、运行模型或操作桌面，不替代现场集成验收。

主要已合并研究PR：[训练基础 #3](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/3)、[实际反馈 #11](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/11)、[SFT128 #13](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/13)、[首键时延 #15](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/15)、[一步奖励学习 #17](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/17)、[负音频结果 #19](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/19)、[冻结视觉头 #21](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/21)、[缓存 #23](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/23)、[RGB保留 #25](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/25)、[平衡音频 #27](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/27)、[RGB重放 #29](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/29)、[语料搬迁 #31](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/31)、[CPU调度 #33](https://github.com/Ziran-Ziran/gamescaling-agent-lab/pull/33)。

下一步应先取得有许可、含实际任务反馈的连续环境数据，冻结真实RGB基线/保留集；用新划分扩展训练任务，保留当前测试作最终审计，避免针对少量损伤种子反复调参。人工音频需要更多独立场景和真实传感器分布；最新帧调度需要实际过期/覆盖率和任务损伤一起验收。顺序RL应在环境状态、奖励、终止和干预边界清楚后单独开展。

可选多步slalom方案保留为未执行草稿：主320回合审查与发布完成时已过其07:10整轮预算门槛，未启动缩小或挑选策略的替代实验。蒙德静态场景未占用GPU；没有游戏/桌面输入、收费云资源、大模型下载或共享源素材修改。
