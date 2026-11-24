# 复现入口与私有输入

本目录公开代码和匿名汇总，不含可用于独立重新训练的私有数据。`source/` 是本轮执行代码快照，源文件 SHA256 见 `execution_manifest_public.json`。

在具有原始缓存的环境中，将 `source/` 内的 Python 文件放入原有 `outputs/vr261119_10x40_raw_acc/` 缓存根目录，再依次执行：

```powershell
python check_shared_private.py
python train_shared_private.py
python evaluate_shared_private.py
python verify_shared_private.py
python report_shared_private.py
```

结果保存到缓存根目录下的 `shared_private/`。已有本轮检查点时训练入口会复用它们；要从头重复本轮 400 步，应使用新的空 `shared_private/` 目录，保留输入目录。不要删除历史模型。

必需私有输入：

- `paired_raw_acc_private.npz`：已有样本 ID、角色和配对记录。
- `phrase_auth/frontend_private.npz`：3200 条、含训练掩码的 762 维 ACC 特征；前端生成入口见快照中的 `prepare_phrase_auth.py`，需要原始采集目录与其历史缓存，公开包不含这些数据。
- `phrase_auth/fresh_warm_private.pt`：仅 16 个训练词预热 700 步的 ACC 初始化。
- `phrase_auth/fresh_fitonly_teacher_private.pt`：仅相同 1280 条训练 Audio 适配过的教师 projection 与残差分支。
- `audio_multiword_fitonly/official_pool_private.npy`、`official_projection_private.pt`、`extraction_progress_private.json`：官方 ECAPA 冻结主干缓存及其样本对齐记录；本轮仅索引 1280 条训练记录，其他行不用于监督。

输入哈希和官方 Audio 主干 SHA256 见执行清单。不能用见过查询词的其他旧编码器、教师适配参数或归一化器替代。

运行环境：Python 3.10、PyTorch 2.5.0 CPU、NumPy 1.23.4、scikit-learn 1.4.2；两个 CPU 线程。依赖另含 SciPy、pandas、threadpoolctl；本轮使用已缓存的 ECAPA 中间特征，不需要重新启动 SpeechBrain 波形推理。

`thresholds_private.json`、`views_private.npz`、权重、身份置换映射及逐样本分数不上传。公开逐用户 ID 为固定匿名序号，没有姓名映射。
