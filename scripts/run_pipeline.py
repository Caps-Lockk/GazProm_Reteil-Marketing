#!/usr/bin/env python
"""Запуск без установки пакета:  python scripts/run_pipeline.py --data data/case_data.xlsx"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricesens.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
