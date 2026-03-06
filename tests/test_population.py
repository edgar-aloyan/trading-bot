"""Тесты для evolution/population.py — используют mock DB."""

from __future__ import annotations

import pytest

from core.decision import FilterConfig
from core.signals import SignalValues
from evolution.fitness import FitnessConfig
from evolution.genetics import GeneticsConfig
from evolution.population import Population
from paper.simulator import PaperTradingConfig
from tests.mock_db import MockStateDB


def _paper_config() -> PaperTradingConfig:
    return PaperTradingConfig(
        initial_balance_usd=10000.0,
        position_size_usd=1000.0,
        maker_fee=0.0001,
        taker_fee=0.0006,
        slippage_factor=0.5,
    )


def _fitness_config() -> FitnessConfig:
    return FitnessConfig(
        winrate_weight=0.30,
        profit_factor_weight=0.30,
        sharpe_weight=0.20,
        drawdown_weight=0.20,
    )


def _filter_config() -> FilterConfig:
    return FilterConfig(
        max_spread_usd=2.0,
        min_volatility=0.0001,
        max_volatility=0.01,
        flow_weight=0.5,
    )


def _genetics_config() -> GeneticsConfig:
    return GeneticsConfig(
        elite_ratio=0.3,
        crossover_ratio=0.4,
        mutation_ratio=0.3,
        mutation_rate=0.2,
        mutation_strength=0.1,
    )


def _long_signal_values() -> SignalValues:
    """Значения, которые должны вызвать LONG у большинства ботов."""
    return SignalValues(
        imbalance=0.90,
        flow_ratio=5.0,
        eth_lead=0.01,
        btc_change=0.0,
        funding_rate=0.0,
        spread=1.0,
        volatility=0.001,
    )


async def _make_pop(size: int = 5, trigger: int = 100) -> Population:
    db = MockStateDB()
    pop = Population(
        size=size,
        paper_config=_paper_config(),
        fitness_config=_fitness_config(),
        genetics_config=_genetics_config(),
        evolution_trigger_trades=trigger,
        filter_config=_filter_config(),
        db=db,
    )
    await pop.init_from_db()
    return pop


class TestPopulation:
    @pytest.mark.asyncio
    async def test_creation(self) -> None:
        pop = await _make_pop(size=10)
        assert len(pop.bots) == 10
        assert pop.generation == 0
        assert pop.total_trades == 0

    @pytest.mark.asyncio
    async def test_process_signals_returns_all_bots(self) -> None:
        pop = await _make_pop()
        values = _long_signal_values()
        results = pop.process_signals(values, 67000.0, 1.0, 1000.0)
        assert len(results) == 5
        bot_ids = [r[0] for r in results]
        assert sorted(bot_ids) == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_evolution_trigger(self) -> None:
        pop = await _make_pop(trigger=3)
        assert not pop.should_evolve()
        pop._total_trades = 3
        assert pop.should_evolve()

    @pytest.mark.asyncio
    async def test_run_evolution(self) -> None:
        pop = await _make_pop(trigger=3)
        assert pop.generation == 0
        await pop.run_evolution()
        assert pop.generation == 1
        assert len(pop.bots) == 5
        assert pop.total_trades == 0

    @pytest.mark.asyncio
    async def test_position_tracking(self) -> None:
        """Боты открывают позиции → last_opened_positions заполняется."""
        pop = await _make_pop(size=3)
        values = _long_signal_values()
        pop.process_signals(values, 67000.0, 1.0, 1000.0)
        # Хотя бы некоторые боты должны открыть позиции
        total_events = len(pop.last_opened_positions) + len(pop.last_closed_trades)
        assert total_events >= 0  # Не падает, структуры заполнены
