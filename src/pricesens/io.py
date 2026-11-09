"""Загрузка и разбор файла с данными кейса."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SHEET_MAIN = "Данные"
SHEET_STATIONS = "Информация об АЗС"
SHEET_CLIENTS = "Информация о клиентах"
SHEET_GOODS = "Покупки сопутствующих товаров"
SHEET_MARGIN = "Маржа"
SHEET_DENSITY = "Плотность"


@dataclass
class Dataset:
    panel: pd.DataFrame        # клиент × атрибуты (АЗС, топливо, регион, формат, ...)
    months: list[str]
    volume: np.ndarray         # V[i, t] — литры
    price_own: np.ndarray      # P[i, t] — цена основной АЗС, ₽/л
    price_comp: np.ndarray     # C[i, t] — средняя цена конкурентов, ₽/л
    clients: pd.DataFrame
    goods: pd.DataFrame
    margin: pd.DataFrame
    density: pd.Series

    @property
    def gap(self) -> np.ndarray:
        """Ценовой разрыв: наша цена − конкуренты (>0 — мы дороже)."""
        return self.price_own - self.price_comp

    @property
    def group_key(self) -> np.ndarray:
        """Ключ рыночной группы «регион | топливо»."""
        return (self.panel["region"] + "|" + self.panel["fuel"]).to_numpy()


def _read_all_sheets(path: Path) -> dict[str, pd.DataFrame]:
    cache = path.with_suffix(path.suffix + ".pkl")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pickle.loads(cache.read_bytes())
    engine = "pyxlsb" if path.suffix.lower() == ".xlsb" else None
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine=engine)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Для чтения .xlsb установите pyxlsb (`pip install pyxlsb`) "
            "или пересохраните файл в .xlsx"
        ) from exc
    cache.write_bytes(pickle.dumps(sheets))
    return sheets


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    x = _read_all_sheets(path)

    raw = x[SHEET_MAIN]
    months = [str(m)[:7] for m in raw.iloc[1, 3:12]]
    body = raw.iloc[2:].reset_index(drop=True)
    body.columns = (["client", "station", "fuel"]
                    + [f"v_{m}" for m in months]
                    + [f"p_{m}" for m in months]
                    + [f"c_{m}" for m in months])
    num = body.columns[3:]
    body[num] = body[num].apply(pd.to_numeric, errors="coerce")
    body = body.dropna(subset=["client"]).reset_index(drop=True)

    stations = x[SHEET_STATIONS].rename(columns={
        "АЗС": "station", "Подразделение": "region", "Формат": "format",
        "Сегмент расположения АЗС": "location", "Тип управления": "management",
        "Тип АЗС": "station_type", "Время работы": "hours"})
    panel = body[["client", "station", "fuel"]].merge(stations, on="station", how="left")

    margin = x[SHEET_MARGIN].rename(columns={
        "НГ": "fuel", "Регион": "region",
        "Средняя маржа, руб/т": "margin_t", "Средний OPEX, руб/т": "opex_t"})
    margin["fuel"] = margin["fuel"].str.upper()
    dens = x[SHEET_DENSITY]
    density = pd.Series(dens["Плотность"].to_numpy(), index=dens["НГ"].str.upper())

    clients = x[SHEET_CLIENTS].rename(columns={
        "ID_KLIENTA": "client", "NALICHIE_Mob prilozheniya": "has_app",
        "SOGLASIE_NA_MARKETINGOVYJ_PUSH": "push_consent",
        "SOGLASIE_NA_SMS_RASSYLKU": "sms_consent", "SOGLASIE_NA_EMAILRASSYLKU": "email_consent",
        "STATUS_BONUSNOJ_KARTY": "card_status", "DATA_PERVOIJ_POKUPKI": "first_purchase",
        "DATA_ROZHDENIYA": "birth_date"})
    goods = x[SHEET_GOODS].rename(columns={
        "ID_KLIENTA": "client",
        "Число месяцев с покупками за период анализа": "goods_months",
        "Число чеков за период анализа": "goods_checks",
        "Общая выручка по СТиУ Брутто с НДС, руб": "goods_gross",
        "Общая выручка по СТиУ Нетто с НДС, руб": "goods_net"})

    return Dataset(
        panel=panel, months=months,
        volume=body[[f"v_{m}" for m in months]].to_numpy(float),
        price_own=body[[f"p_{m}" for m in months]].to_numpy(float),
        price_comp=body[[f"c_{m}" for m in months]].to_numpy(float),
        clients=clients, goods=goods, margin=margin, density=density,
    )
