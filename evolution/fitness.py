"""Оценка качества бота — log-growth fitness (критерий Келли).

fitness = mean(log(1 + pnl_i / position_size))

Логарифм естественно встраивает риск: потери весят непропорционально
больше прибылей той же величины. Не нужны отдельные веса, caps, floors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    """Параметры fitness функции из params.yaml."""

    # Бот с < min_trades получает пропорционально сниженный fitness
    min_trades_for_full_fitness: int

    @staticmethod
    def from_yaml(path: str) -> FitnessConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        fit = raw["fitness"]
        return FitnessConfig(
            min_trades_for_full_fitness=fit["min_trades_for_full_fitness"],
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


def compute_fitness(
    trades: list[TradeRecord],
    position_size: float,
    config: FitnessConfig,
) -> float:
    """Log-growth fitness — критерий Келли.

    fitness = mean(log(1 + pnl_i / position_size))

    Логарифм автоматически:
    - поощряет рост капитала (позитивная цель)
    - наказывает потери сильнее чем поощряет прибыли (риск встроен)
    - не требует весов, caps, normalization
    """
    if not trades:
        return 0.0

    log_returns = [math.log(1.0 + t.pnl / position_size) for t in trades]
    raw_fitness = sum(log_returns) / len(log_returns)

    # Штраф за малое число сделок — плавная деградация
    trade_penalty = min(1.0, len(trades) / config.min_trades_for_full_fitness)
    return raw_fitness * trade_penalty


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
        return min(1.0, max_dd / peak)
    # Если пика не было (все сделки убыточные), drawdown = 100%
    return 1.0 if max_dd > 0 else 0.0
