"""Выгрузка результатов в Excel и JSON."""
import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

CLIENT_COLUMNS = {
    "client": "Клиент", "station": "АЗС", "fuel": "Топливо", "region": "Регион",
    "location": "Сегмент расположения", "format": "Формат", "segment": "Сегмент",
    "p_sensitive": "Вероятность ценозависимости",
    "beta_eb": "Эффект +1 ₽ разрыва, доля объёма (EB)", "beta": "Эффект +1 ₽, сырая оценка",
    "t_stat": "t-статистика", "resp_up": "Реакция на рост разрыва ≥0,5",
    "resp_down": "Реакция на снижение разрыва ≥0,5", "trend": "Тренд индекса объёма в мес.",
    "avg_l_month": "Средний объём, л/мес", "zero_share": "Доля месяцев без заправок",
    "gap_mean": "Средний разрыв, ₽/л", "gap_last": "Разрыв в последнем месяце, ₽/л",
    "lost_l_period": "Недобор из-за разрыва за период, л", "has_app": "Есть приложение",
    "push_consent": "Согласие на push", "card_status": "Статус карты", "goods_net": "Выручка СТиУ нетто, ₽",
}


def _style(path: Path):
    wb = load_workbook(path)
    head = PatternFill("solid", fgColor="0070C0")
    for ws in wb:
        for c in ws[1]:
            c.font = Font(name="Arial", bold=True, color="FFFFFF")
            c.fill = head
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col[:50]) + 2
            ws.column_dimensions[col[0].column_letter].width = max(12, min(45, width))
        ws.freeze_panes = "A2"
    return wb


def _scaling_block(ws, scen_row: int, n_clients: int):
    """Блок масштабирования на всю программу — на формулах, с выделенным допущением."""
    blue = Font(name="Arial", color="0000FF")
    r = ws.max_row + 3
    ws.cell(r, 1, f"Масштабирование сценария из строки {scen_row}").font = Font(name="Arial", bold=True)
    rows = [
        ("Клиентов в выборке", n_clients, "Данные кейса"),
        ("Участников программы «Нам по пути»", 13_500_000, "Условие кейса"),
        ("Доля активных («текущих») клиентов", 0.4, "ДОПУЩЕНИЕ — подставьте фактическую долю из CRM"),
        ("Эффект на выборке, ₽/год", f"=F{scen_row}", ""),
        ("Эффект на 1 000 клиентов, ₽/год", f"=B{r + 4}/B{r + 1}*1000", ""),
        ("Масштаб на активную базу, ₽/год", f"=B{r + 5}/1000*B{r + 2}*B{r + 3}", "При условии репрезентативности выборки"),
    ]
    for i, (label, value, note) in enumerate(rows, 1):
        ws.cell(r + i, 1, label)
        c = ws.cell(r + i, 2, value)
        ws.cell(r + i, 3, note)
        if not isinstance(value, str):
            c.font = blue
        c.number_format = "0%" if "Доля" in label else "#,##0"
    ws.cell(r + 3, 2).fill = PatternFill("solid", fgColor="FFFF00")
    ws.cell(r + 8, 1, "Допущения: доля возврата потерянного объёма — см. config.EconomicsConfig; "
                      "затраты = компенсация × (весь объём клиента + возвращённый); "
                      "маржа = (средняя маржа − OPEX) × плотность.")


def write_excel(res, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    clients = (res["clients"][list(CLIENT_COLUMNS)].rename(columns=CLIENT_COLUMNS)
               .sort_values(["Сегмент", "Вероятность ценозависимости"], ascending=[True, False]))
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        res["segment_summary"].to_excel(w, sheet_name="Сегменты", index=False)
        res["scenarios"].to_excel(w, sheet_name="Экономика", index=False)
        res["sensitivity"].to_excel(w, sheet_name="Чувствительность экономики", index=False)
        clients.to_excel(w, sheet_name="Клиенты", index=False)
        res["effect_by"].to_excel(w, sheet_name="Эффект по разрезам", index=False)
        res["binned"].to_excel(w, sheet_name="Разрыв vs объём", index=False)
        res["robustness_table"].to_excel(w, sheet_name="Устойчивость", index=False)
        res["threshold_grid"].to_excel(w, sheet_name="Пороги", index=False)
    wb = _style(path)
    _scaling_block(wb["Экономика"], scen_row=5, n_clients=len(res["clients"]))  # строка 5 = основной сценарий
    wb.save(path)


def write_summary(summary: dict, path: Path):
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
