"""Задача 1.3 — алгоритм сегментации по характеру реакции на ценовые условия."""
import numpy as np

from .config import SegmentationConfig

SWITCHERS = "1. Ценозависимые «переключатели»"
PROBABLE = "2. Вероятно чувствительные"
FADING = "3. Угасающие (не из-за цены)"
EPISODIC = "4. Эпизодические"
CORE = "5. Лояльное ядро"
SEGMENTS = [SWITCHERS, PROBABLE, FADING, EPISODIC, CORE]


def assign_segments(*, p_sens, beta, trend, V, G, cfg: SegmentationConfig = SegmentationConfig()):
    """Правила применяются по возрастанию приоритета (последнее присваивание побеждает).

    Ценовые сегменты требуют, чтобы у клиента вообще было что оценивать:
    разрыв менялся ≥ min_gap_range и было ≥ min_active_months активных месяцев.
    """
    zero_share = (V == 0).mean(1)
    active = (V > 0).sum(1)
    gap_range = G.max(1) - G.min(1)
    ratio = V[:, -3:].mean(1) / np.maximum(V[:, :3].mean(1), 1e-9)
    informative = (gap_range >= cfg.min_gap_range) & (active >= cfg.min_active_months)

    seg = np.full(len(V), CORE, dtype=object)
    seg[(trend <= cfg.fade_trend) & (ratio < cfg.fade_ratio)] = FADING
    seg[zero_share >= cfg.episodic_zero_share] = EPISODIC
    seg[(p_sens >= cfg.p_probable) & (beta <= cfg.beta_probable) & informative] = PROBABLE
    seg[(p_sens >= cfg.p_switcher) & (beta <= cfg.beta_switcher) & informative] = SWITCHERS
    return seg
