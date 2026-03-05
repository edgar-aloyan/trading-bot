"""Вычисление сырых торговых сигналов из рыночных данных.

Этот модуль отвечает только за числа — imbalance, flow, lead-lag и т.д.
Решения (LONG/SHORT/HOLD) принимает decision.py на основе порогов бота.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import yaml

from core.market_data import MarketSnapshot, OrderBook, Trade

# ---------------------------------------------------------------------------
# Signal enum — итоговое решение бота (используется в decision.py)
# ---------------------------------------------------------------------------


class Signal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


# ---------------------------------------------------------------------------
# Raw signal values — то, что вычисляет SignalComputer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalValues:
    """Сырые значения всех сигналов — без интерпретации."""

    imbalance: float  # [0, 1] — доля bid в общем объёме топ-N уровней
    flow_ratio: float  # buy_volume / sell_volume за окно
    eth_lead: float  # % изменение ETH за окно
    btc_change: float  # % изменение BTC за то же окно (для сравнения)
    funding_rate: float  # текущий funding rate perpetual
    spread: float  # текущий спред в USD
    volatility: float  # стандартное отклонение доходностей за окно


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalsConfig:
    """Параметры вычисления сигналов из params.yaml."""

    orderbook_levels: int
    flow_window_seconds: int
    eth_window_seconds: int
    volatility_window_seconds: int

    @staticmethod
    def from_yaml(path: str) -> SignalsConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        sig = raw["signals"]
        flt = raw["filters"]
        return SignalsConfig(
            orderbook_levels=sig["orderbook_levels"],
            flow_window_seconds=sig["flow_window_seconds"],
            eth_window_seconds=sig["eth_window_seconds"],
            volatility_window_seconds=flt["volatility_window_seconds"],
        )


# ---------------------------------------------------------------------------
# Price sample — для отслеживания истории цен
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PriceSample:
    price: float
    timestamp: float


# ---------------------------------------------------------------------------
# SignalComputer — вычисляет сырые значения сигналов
# ---------------------------------------------------------------------------


class SignalComputer:
    """Вычисляет сырые значения сигналов из MarketSnapshot.

    Хранит историю цен ETH и BTC для lead-lag и volatility.
    Один экземпляр на всю систему — результат общий для всех ботов.
    """

    def __init__(self, config: SignalsConfig) -> None:
        self._config = config
        # Максимальное окно — для очистки старых данных
        max_window = max(
            config.eth_window_seconds,
            config.volatility_window_seconds,
        )
        self._max_window = max_window

        self._btc_prices: deque[_PriceSample] = deque()
        self._eth_prices: deque[_PriceSample] = deque()

    def update(self, snapshot: MarketSnapshot) -> SignalValues:
        """Обновляет историю и вычисляет все сигналы."""
        now = snapshot.timestamp or time.time()

        self._record_prices(snapshot, now)
        self._prune_history(now)

        return SignalValues(
            imbalance=compute_imbalance(
                snapshot.btc_book, self._config.orderbook_levels
            ),
            flow_ratio=compute_trade_flow(
                snapshot.recent_trades, self._config.flow_window_seconds, now
            ),
            eth_lead=self._compute_price_change(
                self._eth_prices, self._config.eth_window_seconds, now
            ),
            btc_change=self._compute_price_change(
                self._btc_prices, self._config.eth_window_seconds, now
            ),
            funding_rate=snapshot.btc_perp.funding_rate,
            spread=snapshot.btc_book.spread,
            volatility=self._compute_volatility(now),
        )

    # ----- internal -----

    def _record_prices(self, snapshot: MarketSnapshot, now: float) -> None:
        btc_mid = snapshot.btc_book.mid_price
        if btc_mid > 0:
            self._btc_prices.append(_PriceSample(btc_mid, now))

        eth_mid = snapshot.eth_book.mid_price
        if eth_mid > 0:
            self._eth_prices.append(_PriceSample(eth_mid, now))

    def _prune_history(self, now: float) -> None:
        cutoff = now - self._max_window
        while self._btc_prices and self._btc_prices[0].timestamp < cutoff:
            self._btc_prices.popleft()
        while self._eth_prices and self._eth_prices[0].timestamp < cutoff:
            self._eth_prices.popleft()

    @staticmethod
    def _compute_price_change(
        prices: deque[_PriceSample], window_seconds: int, now: float
    ) -> float:
        """Процентное изменение цены за последние window_seconds."""
        if len(prices) < 2:
            return 0.0
        cutoff = now - window_seconds
        # Находим самую старую цену в окне
        oldest_in_window: _PriceSample | None = None
        for sample in prices:
            if sample.timestamp >= cutoff:
                oldest_in_window = sample
                break
        if oldest_in_window is None or oldest_in_window.price == 0:
            return 0.0
        latest = prices[-1]
        return (latest.price - oldest_in_window.price) / oldest_in_window.price

    def _compute_volatility(self, now: float) -> float:
        """Стандартное отклонение доходностей BTC за volatility_window."""
        cutoff = now - self._config.volatility_window_seconds
        # Собираем цены в окне
        prices_in_window = [
            s.price for s in self._btc_prices if s.timestamp >= cutoff
        ]
        if len(prices_in_window) < 3:
            return 0.0
        # Считаем log-returns
        returns: list[float] = []
        for i in range(1, len(prices_in_window)):
            prev = prices_in_window[i - 1]
            if prev == 0:
                continue
            returns.append((prices_in_window[i] - prev) / prev)
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return float(variance**0.5)


# ---------------------------------------------------------------------------
# Pure functions — вычисление отдельных сигналов
# ---------------------------------------------------------------------------


def compute_imbalance(book: OrderBook, levels: int) -> float:
    """Order Book Imbalance: bid_vol / (bid_vol + ask_vol) для топ-N уровней.

    Возвращает значение в [0, 1].
    > 0.5 — давление покупателей, < 0.5 — давление продавцов.
    """
    bid_vol = sum(level.volume for level in book.bids[:levels])
    ask_vol = sum(level.volume for level in book.asks[:levels])
    total = bid_vol + ask_vol
    if total == 0:
        return 0.5
    return bid_vol / total


def compute_trade_flow(
    trades: list[Trade], window_seconds: int, now: float
) -> float:
    """Trade Flow: buy_volume / sell_volume за последние window_seconds.

    Возвращает ratio >= 0. При отсутствии sell — возвращает buy_volume или 1.0.
    """
    cutoff = now - window_seconds
    buy_vol = 0.0
    sell_vol = 0.0
    for t in trades:
        if t.timestamp < cutoff:
            continue
        if t.side == "buy":
            buy_vol += t.volume
        else:
            sell_vol += t.volume
    if sell_vol == 0:
        # Нет продаж — если есть покупки, возвращаем их объём как ratio
        return buy_vol if buy_vol > 0 else 1.0
    return buy_vol / sell_vol
