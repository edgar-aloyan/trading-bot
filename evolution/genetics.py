"""Генетические операции — селекция, скрещивание, мутация.

Работает с BotParams из core/decision.py.
Все диапазоны параметров определены здесь — единый источник.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import yaml

from core.decision import BotParams

# ---------------------------------------------------------------------------
# Допустимые диапазоны параметров (из SPEC.md section 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParamRange:
    min_val: float
    max_val: float


# Диапазоны параметров — базовые (taker)
# sensitivity = масштаб tanh, weight = ON/OFF (мультипликативный score)
PARAM_RANGES: dict[str, ParamRange] = {
    "micro_sensitivity": ParamRange(0.0000001, 0.00001),
    "micro_weight": ParamRange(0.0, 1.0),
    "delta_sensitivity": ParamRange(0.05, 1.0),
    "delta_weight": ParamRange(0.0, 1.0),
    "take_profit_usd": ParamRange(8.0, 40.0),
    "stop_loss_usd": ParamRange(5.0, 25.0),
    "max_hold_seconds": ParamRange(10.0, 300.0),
    "basis_sensitivity": ParamRange(0.0001, 0.01),
    "basis_weight": ParamRange(0.0, 1.0),
}

# Расширенные диапазоны для maker-популяций
MAKER_PARAM_RANGES: dict[str, ParamRange] = {
    **PARAM_RANGES,
    "limit_offset_usd": ParamRange(0.5, 5.0),
    "cancel_timeout_seconds": ParamRange(5.0, 60.0),
    "exit_order_mode": ParamRange(0.0, 1.0),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GeneticsConfig:
    """Параметры эволюции из params.yaml."""

    elite_ratio: float
    crossover_ratio: float
    mutation_ratio: float
    mutation_rate: float
    mutation_strength: float
    crossover_alpha: float
    tournament_size: int

    @staticmethod
    def from_yaml(path: str) -> GeneticsConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        evo = raw["evolution"]
        return GeneticsConfig(
            elite_ratio=evo["elite_ratio"],
            crossover_ratio=evo["crossover_ratio"],
            mutation_ratio=evo["mutation_ratio"],
            mutation_rate=evo["mutation_rate"],
            mutation_strength=evo["mutation_strength"],
            crossover_alpha=evo["crossover_alpha"],
            tournament_size=evo["tournament_size"],
        )


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def random_params(
    param_ranges: dict[str, ParamRange] | None = None,
) -> BotParams:
    """Генерирует случайные параметры бота в допустимых диапазонах."""
    ranges = param_ranges if param_ranges is not None else PARAM_RANGES
    values: dict[str, float] = {}
    for name, r in ranges.items():
        values[name] = random.uniform(r.min_val, r.max_val)
    return BotParams(**values)


def crossover(
    parent_a: BotParams, parent_b: BotParams, alpha: float,
    param_ranges: dict[str, ParamRange] | None = None,
) -> BotParams:
    """BLX-alpha crossover — ребёнок сэмплируется из расширенного диапазона.

    При alpha=0 — uniform в [min, max] родителей.
    При alpha=0.5 — может выйти на 50% за пределы родителей (exploration).
    """
    ranges = param_ranges if param_ranges is not None else PARAM_RANGES
    values: dict[str, float] = {}
    for name, r in ranges.items():
        val_a = getattr(parent_a, name)
        val_b = getattr(parent_b, name)
        lo = min(val_a, val_b)
        hi = max(val_a, val_b)
        span = hi - lo
        child_val = random.uniform(lo - alpha * span, hi + alpha * span)
        values[name] = _clamp(child_val, r.min_val, r.max_val)
    return BotParams(**values)


def mutate(
    params: BotParams, config: GeneticsConfig,
    param_ranges: dict[str, ParamRange] | None = None,
) -> BotParams:
    """Мутация параметров бота.

    Каждый параметр с вероятностью mutation_rate сдвигается
    на случайную величину в пределах mutation_strength от диапазона.
    """
    ranges = param_ranges if param_ranges is not None else PARAM_RANGES
    values: dict[str, float] = {}
    for name, r in ranges.items():
        val = getattr(params, name)
        if random.random() < config.mutation_rate:
            # Сдвиг пропорционален размеру диапазона
            range_size = r.max_val - r.min_val
            delta = random.uniform(-1, 1) * config.mutation_strength * range_size
            val = _clamp(val + delta, r.min_val, r.max_val)
        values[name] = val
    return BotParams(**values)


def evolve(
    population: list[BotParams],
    fitness_scores: list[float],
    config: GeneticsConfig,
    param_ranges: dict[str, ParamRange] | None = None,
) -> list[BotParams]:
    """Один цикл эволюции:

    1. Сортировать по fitness
    2. Топ elite_ratio — выживают без изменений
    3. crossover_ratio — потомки от tournament-selected родителей (BLX-alpha)
    4. Остаток — заменяются случайными (diversity injection)
    """
    n = len(population)
    # Сортируем по fitness (лучшие первые)
    ranked = sorted(
        zip(fitness_scores, population, strict=True),
        key=lambda x: x[0],
        reverse=True,
    )

    n_elite = max(1, int(n * config.elite_ratio))
    n_crossover = int(n * config.crossover_ratio)
    # Остаток — мутанты
    n_mutant = n - n_elite - n_crossover

    new_population: list[BotParams] = []

    # Элита — без изменений
    for _, params in ranked[:n_elite]:
        new_population.append(params)

    # Скрещивание — tournament selection из всей популяции
    for _ in range(n_crossover):
        parent_a = _tournament_select(ranked, config.tournament_size)
        parent_b = _tournament_select(ranked, config.tournament_size)
        child = crossover(parent_a, parent_b, config.crossover_alpha, param_ranges)
        child = mutate(child, config, param_ranges)
        new_population.append(child)

    # Мутанты — полностью случайные (diversity injection)
    for _ in range(n_mutant):
        new_population.append(random_params(param_ranges))

    return new_population


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tournament_select(
    ranked: list[tuple[float, BotParams]],
    tournament_size: int,
) -> BotParams:
    """Выбирает лучшего из tournament_size случайных участников."""
    contestants = random.sample(ranked, min(tournament_size, len(ranked)))
    # ranked уже (fitness, params) — берём лучшего по fitness
    return max(contestants, key=lambda x: x[0])[1]


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))
