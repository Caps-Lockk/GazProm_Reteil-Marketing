"""Задача 3 — экономический эффект механик."""
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import EconomicsConfig


def unit_margin(panel: pd.DataFrame, margin: pd.DataFrame, density: pd.Series) -> pd.DataFrame:
    """Маржа на литр: (маржа − OPEX) ₽/т × плотность т/м³ / 1000 → ₽/л."""
    m = margin.copy()
    m["density"] = m["fuel"].map(density)
    m["margin_l"] = m["margin_t"] * m["density"] / 1000
    m["net_margin_l"] = (m["margin_t"] - m["opex_t"]) * m["density"] / 1000
    return panel[["fuel", "region"]].merge(m, on=["fuel", "region"], how="left")[["margin_l", "net_margin_l"]]


def lost_volume(V, G, beta, p_sens):
    """Ожидаемый недобор литров из-за того, что мы дороже соседей (относительно паритета)."""
    base = V.mean(1, keepdims=True)
    b = np.minimum(beta, 0)[:, None]
    return p_sens[:, None] * np.minimum(1, -b * np.clip(G, 0, None)) * base


def simulate_offer(V, G, beta, p_sens, net_margin_l, target, cfg: EconomicsConfig, name=""):
    """Персональная компенсация разрыва бонусами для целевых клиентов.

    Затраты = компенсация × (весь объём клиента в месяц срабатывания + возвращённый объём):
    скидку получает и тот объём, который клиент купил бы и так.
    Выгода = возвращённый объём × маржа на литр.
    """
    base = V.mean(1, keepdims=True)
    b = np.minimum(beta, 0)[:, None]
    trig = (G > cfg.trigger_gap) & target[:, None]
    disc = np.where(trig, np.minimum(G * cfg.compensation_share, cfg.compensation_cap), 0.0)
    rec = np.where(trig, cfg.recovery_rate * p_sens[:, None] * np.minimum(1, -b * disc) * base, 0.0)
    k = cfg.months_in_year / V.shape[1]
    cost = float((disc * (V + rec)).sum()) * k
    gain = float((rec * net_margin_l[:, None]).sum()) * k
    return {"сценарий": name, "клиентов": int(target.sum()), "возвращено_л_год": rec.sum() * k,
            "доп_маржа_руб_год": gain, "затраты_руб_год": cost, "эффект_руб_год": gain - cost,
            "ROI": (gain - cost) / cost if cost else np.nan,
            "затраты_на_1_возвращённый_л": cost / max(rec.sum() * k, 1)}


def scenarios(V, G, beta, p_sens, net_margin_l, segment, reachable, cfg: EconomicsConfig, seg_names):
    sw, pr = seg_names
    target12 = np.isin(segment, [sw, pr])
    everyone = np.ones(len(V), bool)
    full = replace(cfg, compensation_share=1.0, compensation_cap=np.inf, trigger_gap=0.3)
    rows = [
        simulate_offer(V, G, beta, p_sens, net_margin_l, everyone, full, "Массовая компенсация разрыва (всем)"),
        simulate_offer(V, G, beta, p_sens, net_margin_l, target12, full, "Сегменты 1+2, компенсация всего разрыва"),
        simulate_offer(V, G, beta, p_sens, net_margin_l, segment == sw, full, "Сегмент 1, компенсация всего разрыва"),
        simulate_offer(V, G, beta, p_sens, net_margin_l, target12, cfg,
                       f"Сегменты 1+2, порог {cfg.trigger_gap:g} ₽, {cfg.compensation_share:.0%} разрыва, ≤{cfg.compensation_cap:g} ₽".replace(".", ",")),
        simulate_offer(V, G, beta, p_sens, net_margin_l, target12 & reachable, cfg, "То же, только приложение + push"),
    ]
    return pd.DataFrame(rows)


def sensitivity(V, G, beta, p_sens, net_margin_l, target, cfg: EconomicsConfig):
    rows = []
    for rec in (0.4, 0.6, 0.8):
        for share in (0.5, 0.6, 1.0):
            r = simulate_offer(V, G, beta, p_sens, net_margin_l, target,
                               replace(cfg, recovery_rate=rec, compensation_share=share))
            rows.append({"доля_возврата": rec, "доля_компенсации": share,
                         "эффект_руб_год": r["эффект_руб_год"], "ROI": r["ROI"]})
    return pd.DataFrame(rows)
