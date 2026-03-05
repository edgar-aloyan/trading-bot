"""Тесты для evolution/genetics.py."""

from __future__ import annotations

from core.decision import BotParams
from evolution.genetics import (
    PARAM_RANGES,
    GeneticsConfig,
    crossover,
    evolve,
    mutate,
    random_params,
)


def _default_config() -> GeneticsConfig:
    return GeneticsConfig(
        elite_ratio=0.3,
        crossover_ratio=0.4,
        mutation_ratio=0.3,
        mutation_rate=0.2,
        mutation_strength=0.1,
    )


def _fixed_params() -> BotParams:
    return BotParams(
        imbalance_threshold=0.65,
        flow_threshold=1.5,
        take_profit_usd=20.0,
        stop_loss_usd=10.0,
        max_hold_seconds=60.0,
        flow_window_seconds=5.0,
        eth_window_seconds=10.0,
        eth_move_threshold=0.0003,
        leader_weight=0.5,
    )


class TestRandomParams:
    def test_within_ranges(self) -> None:
        for _ in range(50):
            params = random_params()
            for name, r in PARAM_RANGES.items():
                val = getattr(params, name)
                assert r.min_val <= val <= r.max_val, f"{name}={val} out of range"


class TestCrossover:
    def test_average_of_parents(self) -> None:
        a = BotParams(
            imbalance_threshold=0.60,
            flow_threshold=1.4,
            take_profit_usd=10.0,
            stop_loss_usd=5.0,
            max_hold_seconds=30.0,
            flow_window_seconds=4.0,
            eth_window_seconds=6.0,
            eth_move_threshold=0.0002,
            leader_weight=0.2,
        )
        b = BotParams(
            imbalance_threshold=0.80,
            flow_threshold=2.6,
            take_profit_usd=30.0,
            stop_loss_usd=15.0,
            max_hold_seconds=90.0,
            flow_window_seconds=10.0,
            eth_window_seconds=20.0,
            eth_move_threshold=0.0004,
            leader_weight=0.8,
        )
        child = crossover(a, b)
        assert child.imbalance_threshold == 0.70
        assert child.flow_threshold == 2.0
        assert child.take_profit_usd == 20.0
        assert child.leader_weight == 0.5


class TestMutate:
    def test_stays_in_range(self) -> None:
        """Мутация не выходит за допустимые диапазоны."""
        config = GeneticsConfig(
            elite_ratio=0.3,
            crossover_ratio=0.4,
            mutation_ratio=0.3,
            mutation_rate=1.0,  # мутируем всё
            mutation_strength=0.5,  # сильная мутация
        )
        params = _fixed_params()
        for _ in range(100):
            mutated = mutate(params, config)
            for name, r in PARAM_RANGES.items():
                val = getattr(mutated, name)
                assert r.min_val <= val <= r.max_val, f"{name}={val} out of range"

    def test_zero_rate_no_change(self) -> None:
        """При mutation_rate=0 параметры не меняются."""
        config = GeneticsConfig(
            elite_ratio=0.3,
            crossover_ratio=0.4,
            mutation_ratio=0.3,
            mutation_rate=0.0,
            mutation_strength=0.5,
        )
        params = _fixed_params()
        mutated = mutate(params, config)
        assert mutated == params


class TestEvolve:
    def test_population_size_preserved(self) -> None:
        """Размер популяции не меняется после эволюции."""
        config = _default_config()
        pop = [random_params() for _ in range(20)]
        scores = [float(i) for i in range(20)]
        new_pop = evolve(pop, scores, config)
        assert len(new_pop) == 20

    def test_elite_preserved(self) -> None:
        """Лучшие боты сохраняются в новой популяции."""
        config = _default_config()
        pop = [random_params() for _ in range(10)]
        # Последний бот — лучший
        scores = [float(i) for i in range(10)]
        new_pop = evolve(pop, scores, config)
        # Лучший бот (score=9, index=9) должен быть первым
        assert new_pop[0] == pop[9]

    def test_all_params_in_range(self) -> None:
        """Все параметры новой популяции в допустимых диапазонах."""
        config = _default_config()
        pop = [random_params() for _ in range(20)]
        scores = [float(i) for i in range(20)]
        new_pop = evolve(pop, scores, config)
        for params in new_pop:
            for name, r in PARAM_RANGES.items():
                val = getattr(params, name)
                assert r.min_val <= val <= r.max_val
