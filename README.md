# 直播录像按主播切割

按主播声纹自动切分直播录像，跨场次保持主播归属一致。详见 [直播录像按主播切割-技术设计.md](直播录像按主播切割-技术设计.md)。

## 环境

- Python 3.10+
- ffmpeg / ffprobe 已在 PATH
- 推荐 NVIDIA GPU + CUDA（CPU 可跑但慢）

```
pip install -r requirements.txt
```

首次运行会从 ModelScope 拉取 ERes2Net 声纹模型，从 Demucs 仓库拉取 htdemucs 权重。

## 用法

```
python -m pipeline.main <video.mp4> [--dry-run] [--no-demucs] \
       [--work-dir work] [--db-dir host_db] [--output-dir output]
```

- `--dry-run`：只跑识别和段预测，**不切视频**。⚠️ 声纹库 `host_db/` 在自动入库阶段**仍会被写入**（这是两遍匹配机制的必要环节）。如果想完全只读，先备份 `host_db/` 或用 `--db-dir` 指向临时目录。
- `--no-demucs`：跳过人声分离，速度快但 BGM 重时准确率下降

### 冷启动

声纹库为空时，第一场录像所有窗口都会被判为 unknown。只要某段连续 unknown 时长 ≥2h，会做聚类，聚出来的每个**等效时长 ≥1h** 的子簇都会被自动注册为新主播。所以一场 3h 内有 1 次换班的录像，可以一次性把两个主播都入库。后续场次跑时声纹库会持续扩充。

### 输出

- `output/host_001/video1_part01.mp4` 等
- `work/<stem>.segments.json` 段时间线（人工核对用）
- `host_db/` 声纹库（自动维护）

## 关键参数

集中在 [`pipeline/config.py`](pipeline/config.py)，按数据情况调整。

## 模块

| 文件 | 职责 |
|---|---|
| preprocess.py | ffmpeg 抽音 + Demucs 人声分离 |
| embedding.py  | 60s/30s 滑窗 ERes2Net 192 维声纹（流式读取）|
| db.py         | 声纹库读写 + centroid EMA 更新 |
| matching.py   | cosine 匹配 + 阈值/margin + 平滑 |
| enrollment.py | 长 unknown 段聚类 → 自动入库 |
| segmenting.py | 段合并 + unknown 桥接 + 长度过滤 + 过渡丢弃 |
| cutting.py    | I-frame 对齐 + ffmpeg 无损切割 |
| main.py       | 两遍流水线 CLI |
