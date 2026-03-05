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


# Диапазоны из SPEC.md
PARAM_RANGES: dict[str, ParamRange] = {
    "imbalance_threshold": ParamRange(0.55, 0.85),
    "flow_threshold": ParamRange(1.2, 3.0),
    "take_profit_usd": ParamRange(8.0, 40.0),
    "stop_loss_usd": ParamRange(5.0, 25.0),
    "max_hold_seconds": ParamRange(10.0, 120.0),
    "eth_move_threshold": ParamRange(0.0001, 0.0005),
    "leader_weight": ParamRange(0.0, 1.0),
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
        )


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def random_params() -> BotParams:
    """Генерирует случайные параметры бота в допустимых диапазонах."""
    values: dict[str, float] = {}
    for name, r in PARAM_RANGES.items():
        values[name] = random.uniform(r.min_val, r.max_val)
    return BotParams(**values)


def crossover(parent_a: BotParams, parent_b: BotParams) -> BotParams:
    """Скрещивание двух ботов — среднее значение каждого параметра."""
    values: dict[str, float] = {}
    for name in PARAM_RANGES:
        val_a = getattr(parent_a, name)
        val_b = getattr(parent_b, name)
        values[name] = (val_a + val_b) / 2.0
    return BotParams(**values)


def mutate(params: BotParams, config: GeneticsConfig) -> BotParams:
    """Мутация параметров бота.

    Каждый параметр с вероятностью mutation_rate сдвигается
    на случайную величину в пределах mutation_strength от диапазона.
    """
    values: dict[str, float] = {}
    for name, r in PARAM_RANGES.items():
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
) -> list[BotParams]:
    """Один цикл эволюции по SPEC.md:

    1. Сортировать по fitness
    2. Топ elite_ratio — выживают
    3. Средние crossover_ratio — скрещиваются с элитой
    4. Худшие mutation_ratio — заменяются случайными
    """
    n = len(population)
    # Сортируем по fitness (лучшие первые)
    ranked = sorted(
        zip(fitness_scores, population, strict=True), key=lambda x: x[0], reverse=True
    )

    n_elite = max(1, int(n * config.elite_ratio))
    n_crossover = int(n * config.crossover_ratio)
    # Остаток — мутанты
    n_mutant = n - n_elite - n_crossover

    new_population: list[BotParams] = []

    # Элита — без изменений
    for _, params in ranked[:n_elite]:
        new_population.append(params)

    # Скрещивание — средние наследуют от элиты
    elite_params = [params for _, params in ranked[:n_elite]]
    for i in range(n_crossover):
        parent_a = elite_params[i % len(elite_params)]
        parent_b = elite_params[(i + 1) % len(elite_params)]
        child = crossover(parent_a, parent_b)
        # Лёгкая мутация потомка
        child = mutate(child, config)
        new_population.append(child)

    # Мутанты — полностью случайные
    for _ in range(n_mutant):
        new_population.append(random_params())

    return new_population


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))
