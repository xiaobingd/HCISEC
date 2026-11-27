# 复现：分层跨内容 ACC 身份表征实验

方案源提交：`65f31238c0e3eeed7e19bac79fd6f1bbe3d58ce9`。先读 `protocol.json` 和 `report.md`。本轮不改已冻结创新点一，不覆盖历史模型/结果。

## 输入与隐私

需要本地合法持有的 10 人 × 40 词原始硬件三轴 ACC、训练/开发角色真实 Audio 及 PCM 回调时间戳。公共仓库不含这些数据。源清单位于私有 `outputs/vr261119_10x40_raw_acc/continuous_acc_activity/recordings_private.json`，400 行，字段为匿名 `user`、零基 `word`、实际录制 `stem`、`prompt`、`imu_span_s`。不得将 `stem` 对应真实身份/目录映射公开。

实际词 ID 使用缓存词典排序，见公开 `role_split_public.json`，不是采集 W001 编号。清单需与各录制 `_meta.json` 的提示文本一致。`prepare_acc.py` 所需 `_full_imu.csv` 字段：`host_read_perf_counter_s, imu_sample_time, offscale_flags, accel_x, accel_y, accel_z`。

教师仅在训练/开发词打开 `_meta.json`、`_audio_timestamps.csv` 与元数据指定的整段 Audio。PCM 时戳字段：`frame_offset, frames, host_callback_perf_counter_s`。采集元数据 `segments` 数量给出已知转写重复次数。注册/校准/查询 Audio 不得打开，也不参与任何教师提取、定位或模板。

私有缓存输出到 `outputs/vr261119_10x40_raw_acc/hierarchical_phonetic/`。原始信号、后验/边界、样本时间映射、向量、模型参数、模板、阈值数值和逐样本分数均不得上传公共仓库。所有公开聚合指标以匿名 U01–U10 为用户单位，不附身份映射。

## 环境与真实教师

本次 Windows CPU 执行。基础依赖为 PyTorch、NumPy、SciPy、pandas、scikit-learn、threadpoolctl、packaging、PyYAML；实际版本记录在 `software_public.json`。教师运行版本另见 `teacher_model_public.json`。如改操作系统，可使用等价官方包，但须重新记录版本和数值核验。

Windows 公共 HTTPS 下载脚本只下载官方公开模型及 PyPI wheel，校验 SHA256，不下载私有语音数据：

```powershell
$py = 'C:/Users/windows/AppData/Local/Programs/Python/Python310/python.exe'
$src = 'experiments/audio_phonetic_cross_content_20261127'
& "$src/download_teacher.ps1"
& "$src/download_runtime.ps1"
& $py "$src/install_runtime.py"
& $py "$src/runtime.py"
```

eSpeak Windows 数据路径需短于 160 字符；默认使用工作区父级的 `.phonetic-runtime`，可用 `PHONETIC_RUNTIME` 指定短路径。本次直接调用 eSpeak-ng en-us 的 C API 产生完整 IPA 音素序列，再经真实 `Wav2Vec2PhonemeCTCTokenizer(do_phonemize=False)` 验证；没有用逐字符伪音素，也没有以词典标签冒充真实 Audio 后验。

固定官方教师 `facebook/wav2vec2-lv-60-espeak-cv-ft`，revision `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`；1,263,535,127 字节权重 SHA256 `3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e`。只在严格加载全部权重、冻结且 eval 后进行推理。CTC 词表392类、blank0、16kHz、感受野400采样点、步距320采样点。自动 CTC 对齐不是人工音素标注；blank 不等于静音。

## 顺序执行

所有命令均在仓库根目录运行，目录应为本轮独立目录。完成检查点可按协议哈希恢复；不要将其他划分的检查点复制到本目录。

在已有公开结果的仓库中做一次新的复现，应先设置两个新的空输出目录，以保留已发布报告/锁文件（例如下列变量）。原始清单仍从上述私有输入目录读取。单纯重新压缩 NPZ 的字节哈希可能不同；因此新运行生成自己的协议/哈希，不能与旧运行锁混用。默认不设置变量时，仅恢复本次原有私有缓存与检查点。

```powershell
$env:PHONETIC_PUBLIC_OUT = 'outputs/vr261127_audio_phonetic_cross_content_replay'
$env:PHONETIC_PRIVATE_OUT = 'outputs/vr261119_10x40_raw_acc/hierarchical_phonetic_replay'
```

```powershell
& $py "$src/test_common.py"
& $py "$src/test_clock.py"
& $py "$src/test_metrics.py"
& $py "$src/prepare_acc.py"
& $py "$src/extract_teacher.py"
& $py "$src/train.py" --stage development --threads 4
& $py "$src/development.py"
& $py "$src/train.py" --stage final --threads 4
& $py "$src/evaluate.py" --phase calibrate
& $py "$src/evaluate.py" --phase query
& $py "$src/verify.py"
& $py "$src/report.py"
```

开发：12优化词、4未见开发词，五组各200步、seed42；只用开发音素指标和训练梯度核验，锁定默认0.1音素/0.05词对抗权重，无查询或校准评分。最终：全部16训练词从相同种子初始状态重训，五组各1000步，seeds42/43/44，共15模型。批次80，所有用户与训练词按相同规则采样。部署检查点删去全部训练辅助头。

CPU 上缓存冻结教师后验的 `sum(q log q)` 项，避免每步重复计算常量目标熵；仍然是完整392词表的同一 KL 目标。合成测试核对未缓存实现与缓存实现的数值及梯度，不改变训练损失或预算。

可提前运行 `--stage development --arms T0,T1,T2 --threads 2`，与 CPU 教师提取并行，不读取教师缓存；本次确实采用该方式减少等待。其余开发和最终训练使用4线程。训练步数/批次/损失不因线程数改变。`run_pipeline.py` 只负责在教师与前三开发组完成后顺序恢复余下步骤，不会创建周期自动化。

所有150个阈值必须先在 `--phase calibrate` 写入私有锁文件和公开哈希，再执行 `--phase query`。均值模板每人固定12窗口。全部查询按预定三个族保留，不做查询融合。阈值为本人校准Q10与全部9个身份冒充校准Q95的中点。EER由查询ROC计算，仅作描述，不取代校准阈值。

## 核验与输出

- `test_common.py`：精确 CTC 对全路径枚举、重复/空 token、GRL 梯度符号、有效帧内错配、网络推理头隔离。
- `test_metrics.py`：完美/反转/全同分的手算 ROC、100 组独立秩统计重放、无正对 SupCon。
- `test_clock.py`：44.1/48k已知速率、300ms已知时钟跳变、感受野跨界屏蔽、禁止跨缺失时间软插值。实际U09词33曾使单一全段拟合校验失败；最终统一用纯时戳分段规则，不能将该故障当作正常固定延迟。
- `verify.py`：400录制组/2179窗口角色互斥与不重叠；仅818训练/开发窗口教师；3个预定真实CTC重放；全部15模型同初始权重/同批次1000步；仅训练统计标准化；阈值先锁定；全部用户/词/种子指标由独立实现重算；删辅助头部署输出等价。
- `summary_public.json`：主要汇总、三项主对照、内容探针、协议范围和详细指标文件索引。
- `T0`–`T4_anonymous_metrics_public.json`：每一组全部逐种子、逐用户、逐词聚合 EER/AUC/FAR/FRR/TAR/HTER，单位为0–1比例；不是逐样本分数。
- `development_lock_public.json`、`threshold_lock_public.json`：顺序、设置、哈希，不含阈值数值。
- `teacher_audit_public.json`：允许角色、时基、有效帧、blank等匿名诊断；物理同步未标定，不称为精确物理对齐。
- 同K匹配同词/跨词诊断因每人每词独立窗口数不足而不可比较；不复用窗口、不事后改变K、不填写虚构 `G_content`。

主统计单位为10个用户，先聚合词族与三个种子，再进行10,000次用户簇bootstrap、1024种精确符号置换及三项Holm校正。不是将重复录制/词对当成独立用户。当前历史查询词已被评估，独立新采批次才用于最终盲测确认。
