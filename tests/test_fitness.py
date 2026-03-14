"""Тесты для evolution/fitness.py — log-growth fitness."""

from __future__ import annotations

import math

import pytest

from evolution.fitness import (
    FitnessConfig,
    TradeRecord,
    compute_fitness,
    compute_metrics,
)


def _default_config() -> FitnessConfig:
    return FitnessConfig(
        min_trades_for_full_fitness=50,
    )


POSITION_SIZE = 1000.0


# ---------------------------------------------------------------------------
# compute_metrics (мониторинг, не влияет на fitness)
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_empty_trades(self) -> None:
        metrics = compute_metrics([])
        assert metrics.total_trades == 0
        assert metrics.winrate == 0.0
        assert metrics.profit_factor == 0.0

    def test_all_wins(self) -> None:
        trades = [TradeRecord(pnl=10.0, entry_time=0, exit_time=1) for _ in range(5)]
        metrics = compute_metrics(trades)
        assert metrics.winrate == 1.0
        assert metrics.profit_factor == 50.0  # gross_profit / 0 → gross_profit
        assert metrics.total_pnl == 50.0

    def test_all_losses(self) -> None:
        trades = [TradeRecord(pnl=-5.0, entry_time=0, exit_time=1) for _ in range(4)]
        metrics = compute_metrics(trades)
        assert metrics.winrate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.max_drawdown_pct == 1.0

    def test_mixed_trades(self) -> None:
        trades = [
            TradeRecord(pnl=20.0, entry_time=0, exit_time=1),
            TradeRecord(pnl=-10.0, entry_time=1, exit_time=2),
            TradeRecord(pnl=15.0, entry_time=2, exit_time=3),
            TradeRecord(pnl=-5.0, entry_time=3, exit_time=4),
        ]
        metrics = compute_metrics(trades)
        assert metrics.winrate == 0.5
        assert metrics.profit_factor == pytest.approx(35.0 / 15.0)
        assert metrics.total_pnl == 20.0
        assert metrics.total_trades == 4


# ---------------------------------------------------------------------------
# compute_fitness — log-growth (критерий Келли)
# ---------------------------------------------------------------------------


class TestComputeFitness:
    def test_empty_trades(self) -> None:
        score = compute_fitness([], POSITION_SIZE, _default_config())
        assert score == 0.0

    def test_all_wins_positive(self) -> None:
        """Бот с прибыльными сделками имеет положительный fitness."""
        trades = [
            TradeRecord(pnl=10.0, entry_time=0, exit_time=1)
            for _ in range(50)
        ]
        score = compute_fitness(trades, POSITION_SIZE, _default_config())
        # log(1 + 10/1000) = log(1.01) ≈ 0.00995
        expected = math.log(1.0 + 10.0 / POSITION_SIZE)
        assert score == pytest.approx(expected)
        assert score > 0

    def test_all_losses_negative(self) -> None:
        """Бот с убыточными сделками имеет отрицательный fitness."""
        trades = [
            TradeRecord(pnl=-5.0, entry_time=0, exit_time=1)
            for _ in range(50)
        ]
        score = compute_fitness(trades, POSITION_SIZE, _default_config())
        expected = math.log(1.0 - 5.0 / POSITION_SIZE)
        assert score == pytest.approx(expected)
        assert score < 0

    def test_loss_weighs_more_than_equal_gain(self) -> None:
        """Потеря $X даёт больший абсолютный вклад чем прибыль $X.
        Это ключевое свойство log-growth: риск встроен."""
        gain = math.log(1.0 + 10.0 / POSITION_SIZE)   # +$10
        loss = math.log(1.0 - 10.0 / POSITION_SIZE)    # -$10
        # |loss| > |gain| — асимметрия логарифма
        assert abs(loss) > abs(gain)

    def test_mixed_trades(self) -> None:
        """Смешанные сделки дают корректный средний log-return."""
        trades = [
            TradeRecord(pnl=20.0, entry_time=0, exit_time=1),
            TradeRecord(pnl=-10.0, entry_time=1, exit_time=2),
        ]
        config = FitnessConfig(min_trades_for_full_fitness=2)
        score = compute_fitness(trades, POSITION_SIZE, config)
        expected = (
            math.log(1.0 + 20.0 / POSITION_SIZE)
            + math.log(1.0 - 10.0 / POSITION_SIZE)
        ) / 2
        assert score == pytest.approx(expected)

    def test_trade_penalty(self) -> None:
        """Бот с малым числом сделок получает пропорциональный штраф."""
        trades = [
            TradeRecord(pnl=10.0, entry_time=0, exit_time=1)
            for _ in range(25)
        ]
        config = _default_config()  # min_trades = 50
        score_25 = compute_fitness(trades, POSITION_SIZE, config)
        # 50 trades — полный fitness
        trades_50 = trades * 2
        score_50 = compute_fitness(trades_50, POSITION_SIZE, config)
        # 25 из 50 → penalty = 0.5
        assert score_25 == pytest.approx(score_50 * 0.5)

    def test_bigger_position_reduces_fitness_magnitude(self) -> None:
        """При большем position_size те же $10 дают меньший log-return."""
        trades = [TradeRecord(pnl=10.0, entry_time=0, exit_time=1)] * 50
        config = _default_config()
        score_1k = compute_fitness(trades, 1000.0, config)
        score_10k = compute_fitness(trades, 10000.0, config)
        assert score_1k > score_10k > 0

    def test_profitable_bot_beats_losing_bot(self) -> None:
        """Прибыльный бот всегда имеет выше fitness чем убыточный."""
        winners = [TradeRecord(pnl=5.0, entry_time=0, exit_time=1)] * 50
        losers = [TradeRecord(pnl=-5.0, entry_time=0, exit_time=1)] * 50
        config = _default_config()
        assert compute_fitness(winners, POSITION_SIZE, config) > \
            compute_fitness(losers, POSITION_SIZE, config)
