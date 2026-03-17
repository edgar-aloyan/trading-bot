"""Вычисление сырых торговых сигналов из рыночных данных.

Три сигнала:
1. Micro-price deviation — оценка fair price по best bid/ask
2. Volume delta — нормализованный дисбаланс объёмов buy/sell
3. Perp-spot basis — разница perpetual и spot цен

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

    micro_price_deviation: float  # отклонение micro-price от mid, signed
    volume_delta: float  # (buy_vol - sell_vol) / total_vol, [-1, 1]
    basis: float  # (perp_price - spot_price) / spot_price, signed
    funding_rate: float  # текущий funding rate perpetual
    spread: float  # текущий спред в USD
    volatility: float  # стандартное отклонение доходностей за окно


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalsConfig:
    """Параметры вычисления сигналов из params.yaml."""

    flow_window_seconds: int
    volatility_window_seconds: int

    @staticmethod
    def from_yaml(path: str) -> SignalsConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        sig = raw["signals"]
        flt = raw["filters"]
        return SignalsConfig(
            flow_window_seconds=sig["flow_window_seconds"],
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

    Хранит историю цен BTC для volatility.
    Один экземпляр на всю систему — результат общий для всех ботов.
    """

    def __init__(self, config: SignalsConfig) -> None:
        self._config = config
        # Volatility рассчитывается по сэмплам раз в секунду, а не на каждый тик,
        # чтобы измерять рыночную волатильность, а не тиковый шум
        self._btc_vol_prices: deque[_PriceSample] = deque()
        self._last_vol_sample_time: float = 0.0

    def update(self, snapshot: MarketSnapshot) -> SignalValues:
        """Обновляет историю и вычисляет все сигналы."""
        now = snapshot.timestamp or time.time()

        self._record_prices(snapshot, now)
        self._prune_history(now)

        return SignalValues(
            micro_price_deviation=compute_micro_price_deviation(
                snapshot.btc_book
            ),
            volume_delta=compute_volume_delta(
                snapshot.recent_trades, self._config.flow_window_seconds, now
            ),
            basis=compute_basis(
                snapshot.btc_book.mid_price, snapshot.btc_perp.last_price
            ),
            funding_rate=snapshot.btc_perp.funding_rate,
            spread=snapshot.btc_book.spread,
            volatility=self._compute_volatility(now),
        )

    # ----- internal -----

    def _record_prices(self, snapshot: MarketSnapshot, now: float) -> None:
        btc_mid = snapshot.btc_book.mid_price
        # Сэмплируем для volatility не чаще раза в секунду
        if btc_mid > 0 and now - self._last_vol_sample_time >= 1.0:
            self._btc_vol_prices.append(_PriceSample(btc_mid, now))
            self._last_vol_sample_time = now

    def _prune_history(self, now: float) -> None:
        cutoff = now - self._config.volatility_window_seconds
        while self._btc_vol_prices and self._btc_vol_prices[0].timestamp < cutoff:
            self._btc_vol_prices.popleft()

    def _compute_volatility(self, now: float) -> float:
        """Стандартное отклонение доходностей BTC за volatility_window.

        Использует 1-секундные сэмплы вместо тиковых данных,
        чтобы измерять рыночную волатильность, а не микрошум.
        """
        cutoff = now - self._config.volatility_window_seconds
        prices_in_window = [
            s.price for s in self._btc_vol_prices if s.timestamp >= cutoff
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


def compute_micro_price_deviation(book: OrderBook) -> float:
    """Micro-price: оценка fair value по объёмам best bid/ask.

    micro_price = (ask_price * bid_vol + bid_price * ask_vol) / (bid_vol + ask_vol)

    Возвращает отклонение micro-price от mid-price, нормализованное к цене.
    Положительное — покупательское давление, отрицательное — продавцы.
    """
    if not book.bids or not book.asks:
        return 0.0
    best_bid = book.bids[0]
    best_ask = book.asks[0]
    total_vol = best_bid.volume + best_ask.volume
    if total_vol == 0:
        return 0.0
    # Micro-price взвешивает цены ПРОТИВОПОЛОЖНЫМИ объёмами:
    # большой bid_vol толкает fair price к ask (покупатели давят)
    micro = (best_ask.price * best_bid.volume + best_bid.price * best_ask.volume) / total_vol
    mid = book.mid_price
    if mid == 0:
        return 0.0
    return (micro - mid) / mid


def compute_volume_delta(
    trades: list[Trade], window_seconds: int, now: float
) -> float:
    """Normalized volume delta: (buy_vol - sell_vol) / total_vol.

    Возвращает значение в [-1, 1]. Стабильнее чем ratio (не взрывается
    при малых объёмах одной стороны).
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
    total = buy_vol + sell_vol
    if total == 0:
        return 0.0
    return (buy_vol - sell_vol) / total


def compute_basis(spot_price: float, perp_price: float) -> float:
    """Perp-spot basis: (perp - spot) / spot.

    Положительный basis = perpetual торгуется с премией = бычий sentiment.
    Отрицательный = медвежий.
    """
    if spot_price == 0 or perp_price == 0:
        return 0.0
    return (perp_price - spot_price) / spot_price
