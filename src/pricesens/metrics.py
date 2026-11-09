"""Задача 1.2 — показатели ценовой чувствительности клиента."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SHOCK


@dataclass
class ClientFit:
    beta: np.ndarray    # полуэластичность: доля объёма на +1 ₽ разрыва
    se: np.ndarray
    t: np.ndarray
    trend: np.ndarray   # изменение индекса объёма за месяц (не объяснённое ценой)


def fit_gap_trend(A: np.ndarray, G: np.ndarray) -> ClientFit:
    """Векторизованная МНК-регрессия по каждому клиенту: A_it = a_i + β_i·gap_it + γ_i·t + e_it."""
    T = A.shape[1]
    tc = np.arange(T) - (T - 1) / 2
    x = G - G.mean(axis=1, keepdims=True)
    y = A - A.mean(axis=1, keepdims=True)
    s11, s22, s12 = (x * x).sum(1), (tc * tc).sum(), (x * tc).sum(1)
    r1, r2 = (x * y).sum(1), (y * tc).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        det = s11 * s22 - s12 ** 2
        beta = (s22 * r1 - s12 * r2) / det
        trend = (s11 * r2 - s12 * r1) / det
        resid = y - beta[:, None] * x - trend[:, None] * tc
        se = np.sqrt((resid ** 2).sum(1) / (T - 3) * s22 / det)
        t = beta / se
    return ClientFit(beta, se, t, trend)


def shock_response(A: np.ndarray, G: np.ndarray, shock: float = SHOCK):
    """Средняя реакция объёма на месяцы, где разрыв вырос/упал на ≥ shock.

    «Переключатель» теряет объём при росте разрыва и возвращает его при снижении.
    """
    dG, dA = np.diff(G, axis=1), np.diff(A, axis=1)
    out = {}
    for name, mask in (("up", dG >= shock), ("down", dG <= -shock)):
        n = mask.sum(1)
        with np.errstate(invalid="ignore"):
            out[f"resp_{name}"] = np.where(n > 0, (dA * mask).sum(1) / np.maximum(n, 1), np.nan)
        out[f"n_{name}"] = n
    return out


def empirical_bayes_shrink(beta, se, group):
    """Сжатие шумных β к точностно-взвешенному среднему по группе (АЗС × топливо)."""
    w = 1 / np.square(se)
    s = pd.Series
    prior = (s(beta * w).groupby(group).transform("sum") / s(w).groupby(group).transform("sum")).to_numpy()
    tau2 = max(np.nanvar(beta) - np.nanmedian(se ** 2), 1e-4)
    k = tau2 / (tau2 + se ** 2)
    return k * beta + (1 - k) * prior
