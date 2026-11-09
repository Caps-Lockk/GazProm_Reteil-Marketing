"""Графики для отчёта и презентации."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREY = "#0070C0", "#E8590C", "#9AA5B1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gap_vs_volume(binned, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = binned["изменение_объёма_%"].to_numpy()
    labels = binned["изменение_разрыва_руб"].astype(str)
    ax.bar(labels, vals, color=[BLUE if v > 0 else ORANGE for v in vals])
    ax.axhline(0, color="k", lw=.8)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.5 if v > 0 else -1.3), f"{v:+.1f}%", ha="center", fontsize=10)
    ax.set_xlabel("Изменение ценового разрыва за месяц (наша цена − конкуренты), ₽/л")
    ax.set_ylabel("Изменение объёма, % от среднего\n(очищено от рынка региона)")
    ax.set_title("Подорожали относительно соседей — клиенты заправляются меньше")
    _save(fig, path)


def placebo(lfdr, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-8, 6, 0.25)
    ax.hist(lfdr.t_null[np.isfinite(lfdr.t_null)], bins, density=True, alpha=.55, color=GREY,
            label="Плацебо: чужой ценовой ряд (связи нет)")
    ax.hist(lfdr.t_real[np.isfinite(lfdr.t_real)], bins, density=True, histtype="step", lw=2,
            color=BLUE, label="Реальные данные")
    ax.axvline(-2.5, color=ORANGE, ls="--")
    ax.text(-7.8, ax.get_ylim()[1] * 0.75, "«лишняя» масса слева =\nреально ценозависимые", color=ORANGE)
    ax.set_xlabel("t-статистика реакции клиента на ценовой разрыв")
    ax.set_ylabel("Плотность")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Сравнение с плацебо отделяет реальную реакцию от шума")
    _save(fig, path)


def examples(V, G, months, labels, idx, path):
    fig, axs = plt.subplots(1, len(idx), figsize=(4.4 * len(idx), 3.8))
    for ax, i in zip(np.atleast_1d(axs), idx):
        ax.bar(range(len(months)), V[i], color=BLUE, alpha=.8)
        ax2 = ax.twinx()
        ax2.plot(range(len(months)), G[i], color=ORANGE, marker="o", lw=2)
        ax2.axhline(0, color=ORANGE, lw=.6, ls=":")
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([m[2:] for m in months], rotation=60, fontsize=8)
        ax.set_title(labels[i], fontsize=10)
        ax.set_ylabel("литры", color=BLUE)
        ax2.set_ylabel("разрыв, ₽/л", color=ORANGE)
    fig.suptitle("«Переключатели»: объём (столбцы) падает, когда мы дороже соседей (линия), "
                 "и возвращается, когда разрыв закрывается", fontsize=11)
    _save(fig, path)


def segments(summary, path):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    y = np.arange(len(summary))[::-1]
    a, b = summary["доля_клиентов_%"].to_numpy(), summary["доля_объёма_%"].to_numpy()
    ax.barh(y + .2, a, .4, color=BLUE, label="% клиентов")
    ax.barh(y - .2, b, .4, color=ORANGE, label="% объёма")
    for yy, u, v in zip(y, a, b):
        ax.text(u + .5, yy + .2, f"{u:.1f}", va="center", fontsize=9)
        ax.text(v + .5, yy - .2, f"{v:.1f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["сегмент"])
    ax.legend(frameon=False)
    ax.set_title(f"Сегменты клиентов (выборка {int(summary['клиентов'].sum()):,})".replace(",", " "))
    _save(fig, path)


def economics(scen, path):
    e = scen.iloc[[0, 1, 3, 4]]
    vals = e["эффект_руб_год"].to_numpy() / 1e6
    fig, ax = plt.subplots(figsize=(9.5, 4))
    y = np.arange(len(e))[::-1]
    ax.barh(y, vals, color=[ORANGE if v < 0 else BLUE for v in vals])
    ax.axvline(0, color="k", lw=.8)
    for yy, v, r in zip(y, vals, e["ROI"]):
        inside = v < 0
        ax.text(v + .5 if inside else v + .3, yy, f"{v:+.1f} млн, ROI {r:+.0%}", va="center", ha="left",
                fontsize=9, color="white" if inside else "black")
    ax.set_yticks(y)
    ax.set_yticklabels(e["сценарий"], fontsize=9)
    lo = min(vals.min() * 1.25, -1)
    ax.set_xlim(lo, max(vals.max() * 4, 1))
    ax.set_xlabel("Чистый эффект, млн ₽/год (на выборке)")
    ax.set_title("Таргетированная компенсация окупается, массовая скидка — нет")
    _save(fig, path)


def all_figures(res, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    gap_vs_volume(res["binned"], out / "01_gap_vs_volume.png")
    placebo(res["lfdr"], out / "02_placebo.png")
    examples(res["V"], res["G"], res["months"], res["labels"], res["example_idx"], out / "03_examples.png")
    segments(res["segment_summary"], out / "04_segments.png")
    economics(res["scenarios"], out / "05_economics.png")
