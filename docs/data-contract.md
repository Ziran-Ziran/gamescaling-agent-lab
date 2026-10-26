# 玩家训练与多模态输入合同

controller 的运行协议仍以现有源码为准。RGB 轨迹审计/转换及独立 SFT/LoRA CLI 已实现，见 [训练入口](training.md)；真实训练验收与额外模态能力需要独立证据，不能由合同检查推出。

每条轨迹需要 schema_version、episode_id、game_id/build、目标/规则、frame_id、采样单调时间与时钟来源、实际动作开始/结束时间、数据来源/许可、终止原因与质量标记。记录原始数据清单和哈希，不将记录本身提交 Git。

模态按独立通道声明：RGB、音频（采样率/窗口）、深度（单位/光轴或距离/无效值）、结构化 engine_state（字段、单位、坐标、可见权限）。每通道记录时间、缺失掩码和有效性；缺失不能伪装为全零有效观测。数字字段增加不等于模型已经学习该模态。

动作保留官方接口语义：约 200ms 动作块、六个约 33ms 子步骤、整块鼠标位移/滚轮；按键映射、屏幕尺寸、鼠标归一化单位与执行器版本必须记录。建议动作、过滤后动作、实际完成动作分开，历史只使用真实已执行动作；超时或部分执行不得冒充完成。

训练与评测分离：按 game 划分 train/validation/test，另记录 episode 与场景去重；不按相邻帧随机分割。训练期特权信息单独列出，player_observation 评测只允许 rgb/audio；engine_state 和 depth 只有显式 privileged 评测分支可用。学生不能无条件模仿依赖不可观测信息的教师。

实时指标包括采样到动作开始/结束、推理和排队时间、P50/P95、过期率、任务成功与算力成本；声明时钟校准、设备、模型/精度、历史长度、分辨率和并发。配置中的指标只是目标，不构成已经收集这些指标的实现声明。

训练前必须固定上游版本、数据授权、抽样与奖励定义、随机种子、优化器与预算、检查点和恢复策略、评测冻结集合。SFT 与 RL 分别记录模仿目标和环境反馈，普通推理不代表持续学习。

## 已实现的 RGB episode v1

`python -m scripts.training.data` 读取每行一个 episode 的 JSONL。必需顶层字段为：

- `schema_version: 1`、`episode_id`、`game_id`、`game_build`、`task`、`system_prompt`、`input_policy: player_observation`。
- `source` 包含 `id`、`license`、`training_allowed: true`、`kind`（`synthetic` / `recorded` / `replay`）；这是来源声明，不自动授予许可。
- `clock` 包含唯一 `id`、`kind: monotonic`、`unit: ns`。本版本不支持跨设备未标定时间。
- `action_contract` 包含 `version: gamescaling-200ms-v1`、`executor` 版本与正整数 `screen_size: [width, height]`。
- 非空 `frames` 数组；单个完整 episode 必须只出现在一行，不接受按帧打乱后的隐含 episode。

每帧包含严格递增的整数 `frame_id` 与 `captured_ns`；`rgb` 的 `path` 为数据根下的 POSIX 相对路径，`sha256` 校验文件字节。转换时增加含尺寸的解码 RGB `pixel_sha256`，用于识别同像素不同 PNG 编码的泄漏。`actual_action` 包含 `text`、`start_ns`、`end_ns`、布尔 `completed`。时间满足前一次实际动作结束 ≤ 当前采样 ≤ 当前实际动作开始 ≤ 结束。完整动作持续 200ms，允许 ±20ms 采集容差；更长延迟必须另行声明协议，不能悄悄沿用此标签。

动作目标使用精确 `<|action_start|>X Y Z ; g1 ; g2 ; g3 ; g4 ; g5 ; g6<|action_end|>`，XYZ 为整数，鼠标范围 ±1000、滚轮 ±5，每组最多四个无重复官方词表按键。训练标签不做容错、截断或 WASD 过滤；控制器的安全过滤与完整词表的训练标签是不同职责。模型建议可以作为原始附加字段保留，但转换器忽略它。

每帧必须显式填写 `terminated`、`truncated` 布尔值，两者不能同时为真。仅最后一帧必须有一个终止标记和非空 `terminal_reason`，其他帧 reason 为 null。未完成执行会被排除并清空连续历史；原始文件保留以审计失败/部分动作，转换清单记录排除帧及每个 episode 的最终结果，包括最后一帧未完成的情况。源轨迹与转换行逐行复核，训练前再次校验。

划分文件是具有 `train`、`validation`、`test` 三个非空 game_id 列表的 JSON。各游戏只属一组，episode_id 全局唯一，解码 RGB 像素哈希不得跨组；同一游戏中的相邻帧始终同组。默认保留最多五个因果历史，首次 RGB 冒烟使用零历史。合成来源的时钟、动作、终止与游戏 ID 必须明确标为合成，不作为环境任务反馈或跨游戏能力证据。

## 可选音频窗口 v1

独立的 `scripts.training.audio.read_audio` 校验玩家可观察的音频，不向现有 RGB GameScaling API 静默增加字段。它尚未集成进 RGB 轨迹转换器或官方模型；实际融合验收范围见 [训练说明](training.md)。

有效通道声明 `present: true`、相对 `path`、文件 `sha256`、`sample_rate`（8,000–48,000Hz）、`channels: 1`、`encoding: pcm_s16le`、`clock_id`、`window_start_ns`、`window_end_ns`、`available_ns`。当前只接受单声道 PCM16 WAV，样本数与窗口时长相符、最长一秒；记录中的采样率/通道/位宽必须与 WAV 头一致。

通道和 RGB 采样必须用同一 episode 单调时钟：`start < end <= available <= captured_ns`，默认窗口结束距 RGB 不超过 50ms。未来窗口、尚未可用、过期或时钟不一致返回 `present: false`、`missing_reason` 和空 waveform，不读取对应音频；格式、哈希、时长或路径损坏会报错。显式缺失通道为 `present: false` 加非空 `missing_reason`。

有效静音保留 `present: true`，不能与缺失混同。固定特征为八个归一化频带能量、RMS 与过零率，缺失通道和静音都可产生零特征，但布尔掩码不同。单样本窗口过零率定义为 0，禁止产生 NaN。训练和推理必须同时携带掩码；未来片段不会补作历史上下文。
