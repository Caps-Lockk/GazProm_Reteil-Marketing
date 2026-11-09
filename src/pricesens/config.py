"""Параметры модели. Все пороги собраны в одном месте, чтобы их было легко менять."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SegmentationConfig:
    # «Переключатели»: высокая вероятность реакции и сильный эффект
    p_switcher: float = 0.70          # вероятность ценозависимости
    beta_switcher: float = -0.20      # доля объёма на +1 ₽ разрыва (−10% на 0,5 ₽)
    # «Вероятно чувствительные»
    p_probable: float = 0.35
    beta_probable: float = -0.10
    # Минимальная информативность ряда клиента
    min_gap_range: float = 0.50       # разрыв должен меняться хотя бы на 0,5 ₽
    min_active_months: int = 4
    # «Угасающие» и «Эпизодические»
    fade_trend: float = -0.06         # падение индекса объёма за месяц
    fade_ratio: float = 0.60          # последние 3 мес. / первые 3 мес.
    episodic_zero_share: float = 0.34
    # Порог значимого эффекта для вероятности
    min_effect_for_prob: float = -0.05


@dataclass(frozen=True)
class CalibrationConfig:
    n_placebo: int = 10               # число плацебо-прогонов для оценки нулевого распределения
    bin_width: float = 0.25           # ширина бина t-статистики
    t_min: float = -6.0
    seed: int = 100


@dataclass(frozen=True)
class EconomicsConfig:
    trigger_gap: float = 0.50         # оффер включается, если мы дороже соседей на 0,5 ₽/л
    compensation_share: float = 0.60  # доля разрыва, компенсируемая бонусами
    compensation_cap: float = 1.00    # ₽/л, максимум компенсации
    recovery_rate: float = 0.60       # доля потерянного объёма, которую возвращает оффер (допущение)
    months_in_year: int = 12


SHOCK = 0.5  # ₽/л — величина «шока» разрыва из условия кейса (50 копеек)


@dataclass(frozen=True)
class Config:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
