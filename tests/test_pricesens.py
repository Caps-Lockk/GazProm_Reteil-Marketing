"""Тесты на синтетических данных с заранее заложенной ценовой реакцией.

Запуск:  python -m unittest discover -s tests   (или pytest)
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricesens.aggregate import twfe  # noqa: E402
from pricesens.calibration import calibrate, placebo_index  # noqa: E402
from pricesens.config import CalibrationConfig, EconomicsConfig, SegmentationConfig  # noqa: E402
from pricesens.economics import simulate_offer  # noqa: E402
from pricesens.features import market_adjusted_index  # noqa: E402
from pricesens.metrics import fit_gap_trend, shock_response  # noqa: E402
from pricesens.segmentation import CORE, SWITCHERS, assign_segments  # noqa: E402


def synthetic(n=6000, T=9, share=0.10, beta_true=-0.6, n_regions=6, seed=0):
    """Синтетическая панель: у доли `share` клиентов объём падает на |beta|·gap."""
    rng = np.random.default_rng(seed)
    region = rng.integers(0, n_regions, n).astype(str)
    key = np.char.add(region, "|AI95")
    G = np.cumsum(rng.normal(0, 0.6, (n, T)), axis=1) + rng.normal(0.8, 0.5, (n, 1))
    season = 1 + 0.15 * np.sin(np.arange(T) / 2)
    sens = rng.random(n) < share
    b = np.where(sens, beta_true, 0.0)
    gc = G - G.mean(1, keepdims=True)
    idx = np.clip(1 + b[:, None] * gc + rng.normal(0, 0.12, (n, T)), 0, None)
    V = 100 * idx * season
    return V, G, key, region, sens


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.V, cls.G, cls.key, cls.region, cls.sens = synthetic()
        cls.A = market_adjusted_index(cls.V, cls.key)
        cls.fit = fit_gap_trend(cls.A, cls.G)

    def test_twfe_recovers_average_effect(self):
        beta, se = twfe(self.V, self.G, self.key, cluster=np.arange(len(self.V)))
        expected = -0.6 * self.sens.mean()
        self.assertAlmostEqual(beta, expected, delta=0.02)
        self.assertGreater(se, 0)

    def test_client_beta_separates_groups(self):
        self.assertLess(np.median(self.fit.beta[self.sens]), -0.45)
        self.assertLess(abs(np.median(self.fit.beta[~self.sens])), 0.05)

    def test_placebo_index_changes_region(self):
        idx = placebo_index(self.region, seed=1)
        self.assertEqual((self.region == self.region[idx]).sum(), 0)

    def test_calibrated_probability_and_segments(self):
        lfdr = calibrate(self.A, self.G, self.region, CalibrationConfig(n_placebo=5))
        p = lfdr.prob_sensitive(self.fit.t, self.fit.beta, -0.05)
        # сумма вероятностей ≈ числу реально чувствительных
        self.assertAlmostEqual(p.sum() / self.sens.sum(), 1.0, delta=0.25)
        seg = assign_segments(p_sens=p, beta=self.fit.beta, trend=self.fit.trend,
                              V=self.V, G=self.G, cfg=SegmentationConfig())
        picked = seg == SWITCHERS
        self.assertGreater(self.sens[picked].mean(), 0.85, "точность сегмента 1")
        self.assertGreater(picked[self.sens].mean(), 0.5, "полнота сегмента 1")
        self.assertTrue((seg[~self.sens] == CORE).mean() > 0.8)

    def test_shock_response_sign(self):
        r = shock_response(self.A, self.G)
        self.assertLess(np.nanmean(r["resp_up"][self.sens]), 0)
        self.assertGreater(np.nanmean(r["resp_down"][self.sens]), 0)

    def test_offer_only_pays_off_when_targeted(self):
        p = self.sens.astype(float)
        nm = np.full(len(self.V), 10.0)
        cfg = EconomicsConfig()
        targeted = simulate_offer(self.V, self.G, self.fit.beta, p, nm, self.sens, cfg)
        mass = simulate_offer(self.V, self.G, self.fit.beta, p, nm, np.ones(len(self.V), bool), cfg)
        self.assertGreater(targeted["ROI"], mass["ROI"])
        self.assertGreater(targeted["эффект_руб_год"], 0)


if __name__ == "__main__":
    unittest.main()
