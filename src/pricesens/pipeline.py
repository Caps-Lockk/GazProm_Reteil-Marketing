"""Полный конвейер: данные → метрики → сегменты → устойчивость → экономика → отчёты."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import aggregate, economics, plots, report, robustness
from .calibration import calibrate
from .config import SHOCK, Config
from .features import market_adjusted_index
from .io import load_dataset
from .metrics import empirical_bayes_shrink, fit_gap_trend, shock_response
from .segmentation import PROBABLE, SEGMENTS, SWITCHERS, assign_segments

log = logging.getLogger("pricesens")


def run(data_path: str | Path, out_dir: str | Path = "results", cfg: Config = Config(),
        make_figures: bool = True) -> dict:
    out_dir = Path(out_dir)
    ds = load_dataset(data_path)
    V, G, key, panel = ds.volume, ds.gap, ds.group_key, ds.panel
    region = panel["region"].to_numpy()
    log.info("Клиентов: %d, АЗС: %d, месяцев: %d", len(V), panel.station.nunique(), len(ds.months))

    # 1.1 Агрегатная связь
    A = market_adjusted_index(V, key)
    beta_all, se_all = aggregate.twfe(V, G, key, cluster=panel.station.to_numpy())
    effect_by = aggregate.effect_by(V, G, key, panel, ["fuel", "location", "format", "region"])
    binned = aggregate.binned_changes(A, G)
    log.info("TWFE: %.2f%% объёма на +1 ₽ (SE %.2f п.п.)", 100 * beta_all, 100 * se_all)

    # 1.2 Метрики клиента + калибровка
    fit = fit_gap_trend(A, G)
    lfdr = calibrate(A, G, region, cfg.calibration)
    p_sens = lfdr.prob_sensitive(fit.t, fit.beta, cfg.segmentation.min_effect_for_prob)
    shocks = shock_response(A, G)
    beta_eb = empirical_bayes_shrink(fit.beta, fit.se, (panel.station + "|" + panel.fuel).to_numpy())
    log.info("Ожидаемое число реально ценозависимых: %.0f (%.1f%%)", p_sens.sum(), 100 * p_sens.mean())

    # 1.3–1.4 Сегментация
    seg = assign_segments(p_sens=p_sens, beta=fit.beta, trend=fit.trend, V=V, G=G, cfg=cfg.segmentation)

    clients = panel.copy()
    clients = clients.assign(
        segment=seg, p_sensitive=p_sens, beta=fit.beta, beta_eb=beta_eb, t_stat=fit.t, trend=fit.trend,
        resp_up=shocks["resp_up"], n_up=shocks["n_up"], resp_down=shocks["resp_down"], n_down=shocks["n_down"],
        avg_l_month=V.mean(1), zero_share=(V == 0).mean(1), gap_mean=G.mean(1), gap_last=G[:, -1])
    clients = (clients.merge(ds.clients, on="client", how="left")
                      .merge(ds.goods, on="client", how="left"))
    clients["goods_net"] = clients["goods_net"].fillna(0)
    margins = economics.unit_margin(panel, ds.margin, ds.density)
    clients["net_margin_l"] = margins["net_margin_l"].to_numpy()
    lost = economics.lost_volume(V, G, beta_eb, p_sens)
    clients["lost_l_period"] = lost.sum(1)

    tot = V.mean(1).sum()
    g = clients.groupby("segment")
    summary = pd.DataFrame({
        "сегмент": SEGMENTS,
        "клиентов": [int((seg == s).sum()) for s in SEGMENTS],
    })
    summary["доля_клиентов_%"] = (100 * summary["клиентов"] / len(V)).round(1)
    summary["доля_объёма_%"] = [round(100 * g.get_group(s).avg_l_month.sum() / tot, 1) for s in SEGMENTS]
    summary["медиана_л_мес"] = [round(g.get_group(s).avg_l_month.median(), 1) for s in SEGMENTS]
    summary["эффект_+0,5₽_%"] = [round(100 * SHOCK * g.get_group(s).beta_eb.median(), 1) for s in SEGMENTS]
    summary["ср_вероятность"] = [round(g.get_group(s).p_sensitive.mean(), 2) for s in SEGMENTS]
    summary["реакция_на_рост_разрыва"] = [round(g.get_group(s).resp_up.mean(), 2) for s in SEGMENTS]
    summary["реакция_на_снижение_разрыва"] = [round(g.get_group(s).resp_down.mean(), 2) for s in SEGMENTS]
    summary["доля_с_приложением_и_push"] = [
        round(((g.get_group(s).has_app == 1) & (g.get_group(s).push_consent == 1)).mean(), 2) for s in SEGMENTS]
    summary["СТиУ_на_клиента_₽"] = [round(g.get_group(s).goods_net.mean()) for s in SEGMENTS]
    k_year = 12 / V.shape[1]
    summary["недобор_л_год"] = [round(lost[seg == s].sum() * k_year) for s in SEGMENTS]
    summary["недобор_маржи_₽_год"] = [
        round((lost[seg == s].sum(1) * clients.net_margin_l.to_numpy()[seg == s]).sum() * k_year) for s in SEGMENTS]

    # 1.5 Устойчивость
    plac = robustness.placebo_segments(A, G, V, region, lfdr, cfg.segmentation)
    jk = robustness.jackknife_months(A, G, seg)
    sh = robustness.split_half(A, G, seg)
    grid = robustness.threshold_grid(p_sens, fit.beta, fit.trend, V, G, cfg.segmentation)
    b_wo, _ = aggregate.twfe(V, G, key, mask=~np.isin(seg, [SWITCHERS, PROBABLE]))
    b_12, _ = aggregate.twfe(V, G, key, mask=np.isin(seg, [SWITCHERS, PROBABLE]))
    n1, n2 = (seg == SWITCHERS).sum(), (seg == PROBABLE).sum()
    fp1, fp2 = plac[SWITCHERS].mean(), plac[PROBABLE].mean()
    robust_tbl = pd.DataFrame({"проверка": [
        "Плацебо: ложные попадания в сегмент 1 (FDR)",
        "Плацебо: ложные попадания в сегмент 2 (FDR)",
        "Ожидаемое число реально ценозависимых (сумма вероятностей)",
        "Jackknife по месяцам: доля сегмента 1, устойчивая в ≥7 из 9 прогонов",
        "Две половины периода: эффект < 0 в обеих — сегмент 1",
        "Две половины периода: эффект < 0 в обеих — все клиенты",
        "TWFE, все клиенты, % объёма на +1 ₽",
        "TWFE без сегментов 1–2",
        "TWFE только сегменты 1–2",
    ], "значение": [
        f"{fp1:.0f} из {n1} ({fp1 / n1:.0%})", f"{fp2:.0f} из {n2} ({fp2 / n2:.0%})",
        f"{p_sens.sum():.0f} ({p_sens.mean():.1%})", f"{jk:.0%}",
        f"{sh['switchers_both_negative']:.0%}", f"{sh['all_both_negative']:.0%}",
        f"{100 * beta_all:.1f}%", f"{100 * b_wo:.1f}%", f"{100 * b_12:.1f}%"]})

    # 3 Экономика
    reachable = ((clients.has_app == 1) & (clients.push_consent == 1)).to_numpy()
    nm = clients.net_margin_l.to_numpy()
    scen = economics.scenarios(V, G, beta_eb, p_sens, nm, seg, reachable, cfg.economics, (SWITCHERS, PROBABLE))
    sens = economics.sensitivity(V, G, beta_eb, p_sens, nm, np.isin(seg, [SWITCHERS, PROBABLE]), cfg.economics)

    # Примеры для графика: явный уход и возврат
    c = clients
    ex = c[(c.segment == SWITCHERS) & (c.avg_l_month > 80) & (c.n_down > 0) & (c.resp_down > 0.4)
           & (c.n_up > 0) & (c.zero_share <= 0.23)].sort_values("t_stat").head(3).index.to_numpy()
    labels = (c.client + " (" + c.fuel + ", " + c.region + ")").to_numpy()

    res = dict(V=V, G=G, months=ds.months, lfdr=lfdr, binned=binned, effect_by=effect_by,
               clients=clients, segment_summary=summary, scenarios=scen, sensitivity=sens,
               robustness_table=robust_tbl, threshold_grid=grid, labels=labels, example_idx=ex)

    main = scen.iloc[3]
    res["summary"] = {
        "clients": int(len(V)), "stations": int(panel.station.nunique()), "months": ds.months,
        "median_gap_rub": float(np.median(G)),
        "twfe_effect_per_1rub": beta_all, "twfe_se": se_all,
        "twfe_without_segments_1_2": b_wo, "twfe_segments_1_2": b_12,
        "expected_truly_sensitive": float(p_sens.sum()),
        "segments": dict(zip(summary["сегмент"], summary["клиентов"].astype(int))),
        "placebo_fdr_segment1": float(fp1 / n1), "placebo_fdr_segment2": float(fp2 / n2),
        "jackknife_stability_segment1": jk, **sh,
        "lost_margin_rub_year_total": float(summary["недобор_маржи_₽_год"].sum()),
        "main_scenario": {k: (float(v) if not isinstance(v, str) else v) for k, v in main.items()},
        "effect_per_1000_clients_rub_year": float(main["эффект_руб_год"] / len(V) * 1000),
    }

    report.write_excel(res, out_dir / "segmentation_results.xlsx")
    report.write_summary(res["summary"], out_dir / "summary.json")
    summary.to_csv(out_dir / "segments_summary.csv", index=False)
    scen.to_csv(out_dir / "scenarios.csv", index=False)
    if make_figures:
        plots.all_figures(res, out_dir / "figures")
    return res
