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
        crossover_alpha=0.5,
        tournament_size=3,
    )


def _fixed_params() -> BotParams:
    return BotParams(
        micro_price_threshold=0.0005,
        delta_threshold=0.4,
        take_profit_usd=20.0,
        stop_loss_usd=10.0,
        max_hold_seconds=60.0,
        basis_threshold=0.0003,
        basis_weight=0.5,
    )


class TestRandomParams:
    def test_within_ranges(self) -> None:
        for _ in range(50):
            params = random_params()
            for name, r in PARAM_RANGES.items():
                val = getattr(params, name)
                assert r.min_val <= val <= r.max_val, f"{name}={val} out of range"


class TestCrossover:
    def test_blx_alpha_in_range(self) -> None:
        """BLX-alpha crossover всегда в допустимых диапазонах."""
        a = BotParams(
            micro_price_threshold=0.0002,
            delta_threshold=0.1,
            take_profit_usd=10.0,
            stop_loss_usd=5.0,
            max_hold_seconds=30.0,
            basis_threshold=0.0002,
            basis_weight=0.2,
        )
        b = BotParams(
            micro_price_threshold=0.0008,
            delta_threshold=0.7,
            take_profit_usd=30.0,
            stop_loss_usd=15.0,
            max_hold_seconds=90.0,
            basis_threshold=0.0004,
            basis_weight=0.8,
        )
        for _ in range(100):
            child = crossover(a, b, alpha=0.5)
            for name, r in PARAM_RANGES.items():
                val = getattr(child, name)
                assert r.min_val <= val <= r.max_val, f"{name}={val} out of range"

    def test_blx_alpha_produces_diversity(self) -> None:
        """BLX-alpha не просто усредняет — даёт разные результаты."""
        a = _fixed_params()
        b = BotParams(
            micro_price_threshold=0.0008,
            delta_threshold=0.7,
            take_profit_usd=35.0,
            stop_loss_usd=20.0,
            max_hold_seconds=100.0,
            basis_threshold=0.0004,
            basis_weight=0.9,
        )
        children = [crossover(a, b, alpha=0.5) for _ in range(20)]
        # Все дети должны быть разными (стохастичность)
        unique = {c.micro_price_threshold for c in children}
        assert len(unique) > 5

    def test_alpha_zero_between_parents(self) -> None:
        """При alpha=0 ребёнок всегда между родителями."""
        a = _fixed_params()
        b = BotParams(
            micro_price_threshold=0.0008,
            delta_threshold=0.7,
            take_profit_usd=35.0,
            stop_loss_usd=20.0,
            max_hold_seconds=100.0,
            basis_threshold=0.0004,
            basis_weight=0.9,
        )
        for _ in range(50):
            child = crossover(a, b, alpha=0.0)
            for name in PARAM_RANGES:
                val_a = getattr(a, name)
                val_b = getattr(b, name)
                val_c = getattr(child, name)
                lo, hi = min(val_a, val_b), max(val_a, val_b)
                assert lo <= val_c <= hi, f"{name}: {val_c} not in [{lo}, {hi}]"


class TestMutate:
    def test_stays_in_range(self) -> None:
        """Мутация не выходит за допустимые диапазоны."""
        config = GeneticsConfig(
            elite_ratio=0.3,
            crossover_ratio=0.4,
            mutation_ratio=0.3,
            mutation_rate=1.0,  # мутируем всё
            mutation_strength=0.5,  # сильная мутация
            crossover_alpha=0.5,
            tournament_size=3,
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
            crossover_alpha=0.5,
            tournament_size=3,
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
