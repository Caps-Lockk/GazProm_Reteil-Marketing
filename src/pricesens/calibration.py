"""Плацебо-калибровка: из t-статистики → вероятность того, что реакция клиента реальна.

Почему это нужно: при 9 месячных точках и гладких рядах обычные p-value сильно
занижены — наивный отбор «p < 0,1» на плацебо (чужой ценовой ряд) находит ~70%
от числа «чувствительных» на реальных данных. Поэтому нулевое распределение
t-статистик берём эмпирически: подставляем каждому клиенту ценовой ряд случайной
АЗС из другого региона. Дальше — local FDR (Efron): P(реакция реальна | t) = 1 − f0(t)/f(t).
"""
from dataclasses import dataclass

import numpy as np

from .config import CalibrationConfig
from .metrics import fit_gap_trend


def placebo_index(region: np.ndarray, seed: int, max_iter: int = 50) -> np.ndarray:
    """Перестановка клиентов так, чтобы ценовой ряд брался из другого региона."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(region))
    for _ in range(max_iter):
        bad = np.flatnonzero(region == region[idx])
        if bad.size == 0:
            break
        idx[bad] = rng.integers(0, len(region), bad.size)
    return idx


@dataclass
class LocalFDR:
    edges: np.ndarray
    lfdr: np.ndarray      # по бинам, монотонно неубывающая по t
    t_real: np.ndarray
    t_null: np.ndarray

    def prob_sensitive(self, t: np.ndarray, beta: np.ndarray, min_effect: float) -> np.ndarray:
        b = np.clip(np.digitize(t, self.edges) - 1, 0, len(self.lfdr) - 1)
        p = np.where(t < 0, 1 - self.lfdr[b], 0.0)
        return np.where((beta <= min_effect) & np.isfinite(t), p, 0.0)


def calibrate(A, G, region, cfg: CalibrationConfig = CalibrationConfig()) -> LocalFDR:
    t_real = fit_gap_trend(A, G).t
    t_null = np.concatenate([
        fit_gap_trend(A, G[placebo_index(region, cfg.seed + s)]).t for s in range(cfg.n_placebo)])
    t_real_f, t_null_f = t_real[np.isfinite(t_real)], t_null[np.isfinite(t_null)]
    edges = np.r_[-np.inf, np.arange(cfg.t_min, 1e-9, cfg.bin_width)]
    f = np.histogram(t_real_f, edges)[0] / len(t_real_f)
    f0 = np.histogram(t_null_f, edges)[0] / len(t_null_f)
    lfdr = np.clip(f0 / np.maximum(f, 1e-12), 0, 1)
    lfdr = np.convolve(np.pad(lfdr, 1, mode="edge"), np.ones(3) / 3, mode="valid")  # сглаживание
    lfdr = np.maximum.accumulate(lfdr)                                              # монотонность
    return LocalFDR(edges, lfdr, t_real, t_null)
