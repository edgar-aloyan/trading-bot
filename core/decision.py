"""Логика принятия решений одного бота — вход/выход из позиции.

Получает сырые SignalValues от signals.py и параметры бота,
выдаёт Signal (LONG/SHORT/HOLD) и управляет открытой позицией.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import yaml

from core.signals import Signal, SignalValues

# ---------------------------------------------------------------------------
# Bot parameters — индивидуальный набор для каждого бота
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BotParams:
    """Параметры одного бота — мутируют при эволюции.

    Scoring: мультипликативный (soft AND).
    Каждый сигнал нормализуется через tanh(signal / sensitivity) → [-1, 1],
    затем комбинируется: score = Π(1 + weight_i * confidence_i) - 1.
    weight=0 → сигнал выключен, weight=1 → полный вклад.
    """

    micro_sensitivity: float  # масштаб tanh для micro-price (0.0000001–0.00001)
    micro_weight: float  # вкл/выкл micro-price (0.0–1.0)
    delta_sensitivity: float  # масштаб tanh для volume delta (0.05–1.0)
    delta_weight: float  # вкл/выкл volume delta (0.0–1.0)
    take_profit_usd: float  # тейк-профит ($8–$40)
    stop_loss_usd: float  # стоп-лосс ($5–$25)
    max_hold_seconds: float  # макс. время удержания (10–300)
    basis_sensitivity: float  # масштаб tanh для perp-spot basis (0.0001–0.01)
    basis_weight: float  # вкл/выкл basis (0.0–1.0)
    funding_sensitivity: float  # масштаб tanh для funding rate (0.00001–0.001)
    funding_weight: float  # вкл/выкл funding rate (0.0–1.0)
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
        """Мультипликативный score — soft AND с soft ON/OFF весами.

        Каждый сигнал нормализуется через tanh → confidence ∈ [-1, 1].
        Комбинация: score = Π(1 + weight * confidence) - 1.

        Поведение:
        - weight=0 → множитель=1 → сигнал выключен
        - Все сигналы согласны → произведение растёт → сильный score
        - Один сигнал против → множитель <1 → произведение проседает (soft AND)
        """
        p = self.params

        # Нормализуем каждый сигнал в [-1, 1] через tanh
        c_micro = (
            math.tanh(values.micro_price_deviation / p.micro_sensitivity)
            if p.micro_sensitivity > 0 else 0.0
        )
        c_delta = (
            math.tanh(values.volume_delta / p.delta_sensitivity)
            if p.delta_sensitivity > 0 else 0.0
        )
        c_basis = (
            math.tanh(values.basis / p.basis_sensitivity)
            if p.basis_sensitivity > 0 else 0.0
        )
        c_funding = (
            math.tanh(values.funding_rate / p.funding_sensitivity)
            if p.funding_sensitivity > 0 else 0.0
        )

        # Мультипликативная комбинация
        return (
            (1.0 + p.micro_weight * c_micro)
            * (1.0 + p.delta_weight * c_delta)
            * (1.0 + p.basis_weight * c_basis)
            * (1.0 + p.funding_weight * c_funding)
            - 1.0
        )

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
