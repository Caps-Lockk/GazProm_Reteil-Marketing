"""Задача 1.5 — проверки устойчивости результатов."""
import numpy as np
import pandas as pd

from .calibration import placebo_index
from .config import SegmentationConfig
from .metrics import fit_gap_trend
from .segmentation import PROBABLE, SWITCHERS, assign_segments


def placebo_segments(A, G, V, region, lfdr, cfg: SegmentationConfig, seeds=range(500, 505)):
    """Прогон полного алгоритма на плацебо-ценах → ожидаемое число ложных попаданий."""
    rows = []
    for s in seeds:
        Gp = G[placebo_index(region, s)]
        fit = fit_gap_trend(A, Gp)
        p = lfdr.prob_sensitive(fit.t, fit.beta, cfg.min_effect_for_prob)
        seg = assign_segments(p_sens=p, beta=fit.beta, trend=fit.trend, V=V, G=Gp, cfg=cfg)
        rows.append({"seed": s, SWITCHERS: (seg == SWITCHERS).sum(), PROBABLE: (seg == PROBABLE).sum()})
    return pd.DataFrame(rows)


def jackknife_months(A, G, segment, t_cut=-2.2, beta_cut=-0.2):
    """Исключаем по одному месяцу: доля «переключателей», сохраняющих статус в ≥7 из 9 прогонов."""
    T = A.shape[1]
    hits = []
    for k in range(T):
        keep = [i for i in range(T) if i != k]
        fit = fit_gap_trend(A[:, keep], G[:, keep])
        hits.append((fit.t <= t_cut) & (fit.beta <= beta_cut))
    share = np.mean(hits, axis=0)
    core = segment == SWITCHERS
    return float((share[core] >= 7 / 9).mean())


def split_half(A, G, segment):
    """Оценка β по первой и второй половине периода: совпадение знака эффекта."""
    h = A.shape[1] // 2 + 1
    b1 = fit_gap_trend(A[:, :h], G[:, :h]).beta
    b2 = fit_gap_trend(A[:, -h:], G[:, -h:]).beta
    both_neg = (b1 < 0) & (b2 < 0)
    return {"switchers_both_negative": float(both_neg[segment == SWITCHERS].mean()),
            "all_both_negative": float(both_neg.mean())}


def threshold_grid(p_sens, beta, trend, V, G, base: SegmentationConfig):
    rows = []
    for b in (-0.15, -0.20, -0.30):
        for p in (0.6, 0.7, 0.8):
            cfg = SegmentationConfig(**{**base.__dict__, "beta_switcher": b, "p_switcher": p})
            seg = assign_segments(p_sens=p_sens, beta=beta, trend=trend, V=V, G=G, cfg=cfg)
            rows.append({"порог_beta": b, "порог_вероятности": p, "переключателей": int((seg == SWITCHERS).sum())})
    return pd.DataFrame(rows)
