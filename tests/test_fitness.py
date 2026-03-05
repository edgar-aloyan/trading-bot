"""Тесты для evolution/fitness.py."""

from __future__ import annotations

import pytest

from evolution.fitness import (
    FitnessConfig,
    FitnessMetrics,
    TradeRecord,
    compute_fitness,
    compute_metrics,
)


def _default_config() -> FitnessConfig:
    return FitnessConfig(
        winrate_weight=0.30,
        profit_factor_weight=0.30,
        sharpe_weight=0.20,
        drawdown_weight=0.20,
    )


# ---------------------------------------------------------------------------
# compute_metrics
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

    def test_sharpe_ratio(self) -> None:
        trades = [
            TradeRecord(pnl=10.0, entry_time=0, exit_time=1),
            TradeRecord(pnl=12.0, entry_time=1, exit_time=2),
            TradeRecord(pnl=8.0, entry_time=2, exit_time=3),
        ]
        metrics = compute_metrics(trades)
        # mean=10, std=sqrt((0+4+4)/3)≈1.63, sharpe≈6.12
        assert metrics.sharpe_ratio > 5.0

    def test_sharpe_with_few_trades(self) -> None:
        """При < 3 сделках sharpe = 0."""
        trades = [TradeRecord(pnl=10.0, entry_time=0, exit_time=1)]
        metrics = compute_metrics(trades)
        assert metrics.sharpe_ratio == 0.0

    def test_max_drawdown(self) -> None:
        trades = [
            TradeRecord(pnl=100.0, entry_time=0, exit_time=1),  # equity=100
            TradeRecord(pnl=-50.0, entry_time=1, exit_time=2),  # equity=50, dd=50%
            TradeRecord(pnl=30.0, entry_time=2, exit_time=3),  # equity=80
        ]
        metrics = compute_metrics(trades)
        assert metrics.max_drawdown_pct == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_fitness
# ---------------------------------------------------------------------------


class TestComputeFitness:
    def test_perfect_bot(self) -> None:
        metrics = FitnessMetrics(
            winrate=1.0,
            profit_factor=3.0,
            sharpe_ratio=2.0,
            max_drawdown_pct=0.0,
            total_trades=100,
            total_pnl=500.0,
        )
        score = compute_fitness(metrics, _default_config())
        # 1.0*0.3 + 3.0*0.3 + 2.0*0.2 - 0.0*0.2 = 0.3 + 0.9 + 0.4 = 1.6
        assert score == pytest.approx(1.6)

    def test_terrible_bot(self) -> None:
        metrics = FitnessMetrics(
            winrate=0.0,
            profit_factor=0.0,
            sharpe_ratio=-1.0,
            max_drawdown_pct=1.0,
            total_trades=50,
            total_pnl=-500.0,
        )
        score = compute_fitness(metrics, _default_config())
        # 0 + 0 + (-1)*0.2 - 1.0*0.2 = -0.4
        assert score == pytest.approx(-0.4)

    def test_drawdown_penalizes(self) -> None:
        """Высокий drawdown снижает fitness."""
        config = _default_config()
        good = FitnessMetrics(
            winrate=0.6, profit_factor=1.5, sharpe_ratio=1.0,
            max_drawdown_pct=0.05, total_trades=50, total_pnl=100.0,
        )
        bad = FitnessMetrics(
            winrate=0.6, profit_factor=1.5, sharpe_ratio=1.0,
            max_drawdown_pct=0.50, total_trades=50, total_pnl=100.0,
        )
        assert compute_fitness(good, config) > compute_fitness(bad, config)
