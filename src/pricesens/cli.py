"""Запуск полного решения.

Пример:
    pricesens --data data/case_data.xlsx --out results
"""
import argparse
import logging
from dataclasses import replace
from pathlib import Path

from .config import Config
from .pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/case_data.xlsx", help="файл кейса (.xlsx или .xlsb)")
    ap.add_argument("--out", default="results", help="папка для результатов")
    ap.add_argument("--recovery", type=float, help="доля возвращаемого офферами объёма (по умолчанию 0,6)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    cfg = Config()
    if args.recovery is not None:
        cfg = replace(cfg, economics=replace(cfg.economics, recovery_rate=args.recovery))
    res = run(args.data, args.out, cfg, make_figures=not args.no_figures)

    s = res["summary"]
    print("\nСегменты:")
    print(res["segment_summary"][["сегмент", "клиентов", "доля_клиентов_%", "доля_объёма_%", "эффект_+0,5₽_%"]]
          .to_string(index=False))
    print(f"\nОсновной сценарий: {s['main_scenario']['эффект_руб_год'] / 1e6:+.2f} млн ₽/год, "
          f"ROI {s['main_scenario']['ROI']:+.0%}")
    print(f"Результаты: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
