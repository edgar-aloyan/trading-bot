"""Оценка качества бота — multi-objective fitness score.

Формула из SPEC.md:
  fitness = winrate * 0.30 + profit_factor * 0.30
          + sharpe_ratio * 0.20 - max_drawdown_pct * 0.20
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    """Веса fitness функции из params.yaml."""

    winrate_weight: float
    profit_factor_weight: float
    sharpe_weight: float
    drawdown_weight: float

    @staticmethod
    def from_yaml(path: str) -> FitnessConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        fit = raw["fitness"]
        return FitnessConfig(
            winrate_weight=fit["winrate_weight"],
            profit_factor_weight=fit["profit_factor_weight"],
            sharpe_weight=fit["sharpe_weight"],
            drawdown_weight=fit["drawdown_weight"],
        )


# ---------------------------------------------------------------------------
# Trade record — для подсчёта метрик
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Результат одной закрытой сделки."""

    pnl: float
    entry_time: float
    exit_time: float


# ---------------------------------------------------------------------------
# Fitness metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitnessMetrics:
    """Все метрики одного бота за период."""

    winrate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    total_pnl: float


def compute_metrics(trades: list[TradeRecord]) -> FitnessMetrics:
    """Вычисляет все метрики из списка закрытых сделок."""
    if not trades:
        return FitnessMetrics(
            winrate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            total_trades=0,
            total_pnl=0.0,
        )

    wins = sum(1 for t in trades if t.pnl > 0)
    winrate = wins / len(trades)

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

    pnls = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    sharpe = _compute_sharpe(pnls)
    max_dd = _compute_max_drawdown_pct(pnls)

    return FitnessMetrics(
        winrate=winrate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        total_trades=len(trades),
        total_pnl=total_pnl,
    )


def compute_fitness(metrics: FitnessMetrics, config: FitnessConfig) -> float:
    """Итоговый fitness score по формуле из SPEC.md."""
    return (
        metrics.winrate * config.winrate_weight
        + metrics.profit_factor * config.profit_factor_weight
        + metrics.sharpe_ratio * config.sharpe_weight
        - metrics.max_drawdown_pct * config.drawdown_weight
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_sharpe(pnls: list[float]) -> float:
    """Sharpe ratio: mean(pnl) / std(pnl). При малом числе сделок — 0."""
    if len(pnls) < 3:
        return 0.0
    mean = sum(pnls) / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / len(pnls)
    std = variance**0.5
    if std == 0:
        return 0.0
    return float(mean / std)


def _compute_max_drawdown_pct(pnls: list[float]) -> float:
    """Максимальная просадка от пика equity в процентах [0, 1]."""
    if not pnls:
        return 0.0

    # Строим equity curve
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_dd:
            max_dd = drawdown

    # Нормализуем к пику (если пик > 0)
    if peak > 0:
        return max_dd / peak
    # Если пика не было (все сделки убыточные), drawdown = 100%
    return 1.0 if max_dd > 0 else 0.0
