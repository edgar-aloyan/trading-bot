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

    micro_price_threshold: float  # порог отклонения micro-price (0.00001–0.001)
    delta_threshold: float  # порог volume delta (0.05–0.8)
    take_profit_usd: float  # тейк-профит ($8–$40)
    stop_loss_usd: float  # стоп-лосс ($5–$25)
    max_hold_seconds: float  # макс. время удержания (10–120)
    basis_threshold: float  # порог perp-spot basis (0.00001–0.001)
    basis_weight: float  # вес basis сигнала (0.0–1.0)
    # Maker order params — defaults encode taker behavior
    limit_offset_usd: float = 0.0  # отступ от цены для лимитной заявки (0 → taker)
    cancel_timeout_seconds: float = 0.0  # таймаут отмены незаполненного ордера
    exit_order_mode: float = 0.0  # >0.5 → TP exit с maker fee (без slippage)


# ---------------------------------------------------------------------------
# Filter config — общие пороги, не эволюционируют
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Пороги фильтров из params.yaml — одинаковые для всех ботов."""

    max_spread_usd: float
    min_volatility: float
    max_volatility: float
    delta_weight: float  # вес volume delta сигнала в _compute_score

    @staticmethod
    def from_yaml(path: str) -> FilterConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        flt = raw["filters"]
        sig = raw["signals"]
        return FilterConfig(
            max_spread_usd=flt["max_spread_usd"],
            min_volatility=flt["min_volatility"],
            max_volatility=flt["max_volatility"],
            delta_weight=float(sig["delta_weight"]),
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

        # Timeout — checked each tick, so resolution depends on WebSocket frequency
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
        """Фильтры — не торговать когда рынок непригоден."""
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

        # Micro-price deviation: основной сигнал
        # Положительное отклонение = покупательское давление → LONG
        if values.micro_price_deviation > params.micro_price_threshold:
            score += values.micro_price_deviation - params.micro_price_threshold
        elif values.micro_price_deviation < -params.micro_price_threshold:
            score += values.micro_price_deviation + params.micro_price_threshold

        # Volume delta: подтверждающий сигнал
        dw = self.filters.delta_weight
        if values.volume_delta > params.delta_threshold:
            score += (values.volume_delta - params.delta_threshold) * dw
        elif values.volume_delta < -params.delta_threshold:
            score += (values.volume_delta + params.delta_threshold) * dw

        # Perp-spot basis: sentiment сигнал
        # Положительный basis (перп дороже спота) = бычий sentiment
        if abs(values.basis) > params.basis_threshold:
            score += values.basis * params.basis_weight

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
