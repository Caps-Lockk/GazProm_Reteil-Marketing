"""Нормировка объёмов и очистка от рыночных (сезонных, региональных) колебаний."""
import numpy as np
import pandas as pd


def group_mean(X: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Среднее по группе клиентов для каждого месяца, размноженное обратно на клиентов."""
    return pd.DataFrame(X).groupby(key).transform("mean").to_numpy()


def volume_index(V: np.ndarray) -> np.ndarray:
    """Индекс объёма: литры месяца / средние литры клиента (среднее = 1)."""
    base = V.mean(axis=1, keepdims=True)
    return np.divide(V, base, out=np.zeros_like(V), where=base > 0)


def market_adjusted_index(V: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Индекс объёма минус средний индекс рыночной группы (регион × топливо) в том же месяце.

    Убирает сезонность и общий тренд рынка: остаётся то, что отличает клиента
    от «соседей» по рынку — в том числе реакция на ценовой разрыв своей АЗС.
    """
    Y = volume_index(V)
    return Y - group_mean(Y, key) + 1.0
