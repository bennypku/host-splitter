"""Central configuration constants for the pipeline."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    sample_rate: int = 16000

    window_sec: float = 60.0
    hop_sec: float = 60.0

    sim_threshold: float = 0.65
    margin_threshold: float = 0.10

    smooth_window_sec: int = 90

    auto_enroll_min_sec: int = 2 * 3600
    enroll_cluster_distance: float = 0.25
    enroll_purity_threshold: float = 0.80
    merge_existing_threshold: float = 0.75

    unknown_bridge_max_sec: int = 30 * 60
    min_segment_sec: int = 3600
    transition_drop_sec: int = 60

    centroid_ema_old: float = 0.9
    centroid_ema_new: float = 0.10

    embedding_model: str = "iic/speech_eres2net_sv_zh-cn_16k-common"


CFG = Config()
