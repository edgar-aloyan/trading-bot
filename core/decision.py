"""Логика принятия решений одного бота — вход/выход из позиции.

Получает сырые SignalValues от signals.py и параметры бота,
выдаёт Signal (LONG/SHORT/HOLD) и управляет открытой позицией.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from core.signals import Signal, SignalValues

# ---------------------------------------------------------------------------
# Bot parameters — индивидуальный набор для каждого бота
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BotParams:
    """Параметры одного бота — мутируют при эволюции."""

    imbalance_threshold: float  # порог дисбаланса стакана (0.55–0.85)
    flow_threshold: float  # порог давления потока (1.2–3.0)
    take_profit_usd: float  # тейк-профит ($8–$40)
    stop_loss_usd: float  # стоп-лосс ($5–$25)
    max_hold_seconds: float  # макс. время удержания (10–120)
    eth_move_threshold: float  # порог движения ETH (0.01%–0.05%)
    leader_weight: float  # вес поводырей (0.0–1.0)


# ---------------------------------------------------------------------------
# Filter config — общие пороги, не эволюционируют
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Пороги фильтров из params.yaml — одинаковые для всех ботов."""

    max_spread_usd: float
    min_volatility: float
    max_volatility: float

    @staticmethod
    def from_yaml(path: str) -> FilterConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        flt = raw["filters"]
        return FilterConfig(
            max_spread_usd=flt["max_spread_usd"],
            min_volatility=flt["min_volatility"],
            max_volatility=flt["max_volatility"],
        )


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Position:
    """Текущая открытая позиция бота."""

    side: Signal  # LONG или SHORT
    entry_price: float
    entry_time: float
    size_usd: float


# ---------------------------------------------------------------------------
# Decision engine — один экземпляр на бота
# ---------------------------------------------------------------------------


class DecisionEngine:
    """Принимает решения о входе/выходе для одного бота.

    Без позиции: смотрит сигналы → LONG/SHORT/HOLD.
    С позицией: проверяет TP/SL/timeout → нужно ли закрывать.
    """

    def __init__(self, params: BotParams, filters: FilterConfig) -> None:
        self.params = params
        self.filters = filters
        self.position: Position | None = None

    def compute_entry_signal(
        self, values: SignalValues, current_price: float, now: float
    ) -> Signal:
        """Решение о входе — вызывается только когда нет открытой позиции."""
        if not self._pass_filters(values):
            return Signal.HOLD

        score = self._compute_score(values)

        if score > 0:
            return Signal.LONG
        if score < 0:
            return Signal.SHORT
        return Signal.HOLD

    def should_exit(self, current_price: float, now: float) -> bool:
        """Проверяет нужно ли закрыть текущую позицию (TP/SL/timeout)."""
        if self.position is None:
            return False

        pnl = self._unrealized_pnl(current_price)

        # Take profit
        if pnl >= self.params.take_profit_usd:
            return True

        # Stop loss
        if pnl <= -self.params.stop_loss_usd:
            return True

        # Timeout
        hold_time = now - self.position.entry_time
        return hold_time >= self.params.max_hold_seconds

    def open_position(
        self, side: Signal, price: float, time: float, size_usd: float
    ) -> Position:
        """Открывает позицию."""
        self.position = Position(
            side=side, entry_price=price, entry_time=time, size_usd=size_usd
        )
        return self.position

    def close_position(self, exit_price: float) -> float:
        """Закрывает позицию и возвращает PnL в USD."""
        if self.position is None:
            return 0.0
        pnl = self._unrealized_pnl(exit_price)
        self.position = None
        return pnl

    # ----- internal -----

    def _pass_filters(self, values: SignalValues) -> bool:
        """Фильтры по SPEC.md — не торговать когда рынок непригоден."""
        f = self.filters
        # Спред слишком широкий — ликвидности нет
        if values.spread > f.max_spread_usd:
            return False
        # Боковик или нет данных — волатильности не хватает для скальпинга
        if values.volatility < f.min_volatility:
            return False
        # Хаос — слишком высокая волатильность
        return values.volatility <= f.max_volatility

    def _compute_score(self, values: SignalValues) -> float:
        """Вычисляет композитный score для направления сделки.

        Положительный → LONG, отрицательный → SHORT, ~0 → HOLD.
        """
        params = self.params
        score = 0.0

        # Order Book Imbalance: основной сигнал
        if values.imbalance > params.imbalance_threshold:
            score += values.imbalance - params.imbalance_threshold
        elif values.imbalance < (1 - params.imbalance_threshold):
            score -= (1 - params.imbalance_threshold) - values.imbalance

        # Trade Flow: подтверждающий сигнал
        if values.flow_ratio > params.flow_threshold:
            score += (values.flow_ratio - params.flow_threshold) * 0.5
        elif values.flow_ratio < (1 / params.flow_threshold):
            score -= ((1 / params.flow_threshold) - values.flow_ratio) * 0.5

        # ETH lead-lag: поводырь усиливает сигнал
        if abs(values.eth_lead) > params.eth_move_threshold:
            # ETH двигается а BTC ещё нет — усиливаем в направлении ETH
            eth_signal = values.eth_lead - values.btc_change
            score += eth_signal * params.leader_weight

        return score

    def _unrealized_pnl(self, current_price: float) -> float:
        """Нереализованный PnL текущей позиции в USD."""
        if self.position is None:
            return 0.0
        price_diff = current_price - self.position.entry_price
        if self.position.side == Signal.SHORT:
            price_diff = -price_diff
        # PnL пропорционален размеру позиции
        btc_amount = self.position.size_usd / self.position.entry_price
        return price_diff * btc_amount
