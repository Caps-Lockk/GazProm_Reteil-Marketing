"""Задача 1.1 — агрегатная связь ценового разрыва и объёма заправок."""
import numpy as np
import pandas as pd

from .features import group_mean, volume_index


def _two_way_demean(X, key):
    X = X - X.mean(axis=1, keepdims=True)          # фикс. эффект клиента
    return X - group_mean(X, key)                  # фикс. эффект регион×топливо×месяц


def twfe(V, G, key, cluster=None, mask=None):
    """Оценка β в модели  index_it = a_i + d_{g,t} + β·gap_it + e_it.

    Возвращает (β, кластерная SE). β — изменение объёма (в долях среднего) на +1 ₽ разрыва.
    """
    if mask is not None:
        V, G, key = V[mask], G[mask], key[mask]
        cluster = None if cluster is None else cluster[mask]
    y = _two_way_demean(volume_index(V), key)
    g = _two_way_demean(G, key)
    sxx = (g * g).sum()
    beta = (y * g).sum() / sxx
    se = np.nan
    if cluster is not None:
        score = pd.Series((g * (y - beta * g)).sum(axis=1)).groupby(cluster).sum()
        se = float(np.sqrt((score ** 2).sum()) / sxx)
    return float(beta), se


def effect_by(V, G, key, panel, columns, min_clients=300):
    """TWFE-эффект в разрезах (топливо, формат, расположение, регион)."""
    y = _two_way_demean(volume_index(V), key)
    g = _two_way_demean(G, key)
    rows = []
    for col in columns:
        for value, idx in panel.groupby(col).indices.items():
            if len(idx) < min_clients:
                continue
            b = (y[idx] * g[idx]).sum() / (g[idx] ** 2).sum()
            rows.append({"разрез": col, "значение": value, "клиентов": len(idx),
                         "эффект_%_на_1_руб": round(100 * b, 2)})
    return pd.DataFrame(rows)


GAP_CHANGE_BINS = [-np.inf, -1, -0.5, -0.2, 0.2, 0.5, 1, np.inf]
GAP_CHANGE_LABELS = ["< −1", "−1…−0,5", "−0,5…−0,2", "±0,2", "0,2…0,5", "0,5…1", "> 1"]


def binned_changes(A, G):
    """Месячное изменение разрыва vs изменение рыночно-очищенного объёма."""
    df = pd.DataFrame({"dG": np.diff(G, axis=1).ravel(), "dA": np.diff(A, axis=1).ravel()})
    df["bin"] = pd.cut(df.dG, GAP_CHANGE_BINS, labels=GAP_CHANGE_LABELS)
    out = df.groupby("bin", observed=False).agg(наблюдений=("dA", "size"), dA=("dA", "mean"))
    out["изменение_объёма_%"] = (100 * out.pop("dA")).round(2)
    return out.reset_index().rename(columns={"bin": "изменение_разрыва_руб"})
