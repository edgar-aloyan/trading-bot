"""WebSocket market data stream — единый источник данных для всех ботов.

Подписывается на стримы Bybit:
- orderbook.50 BTC/USDT (основной стакан)
- publicTrade BTC/USDT (поток сделок)
- orderbook.50 ETH/USDT (поводырь)
- tickers BTC/USDT:USDT (perpetual funding rate)

Все боты читают из одного экземпляра MarketDataStream.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """Один уровень стакана (цена + объём)."""

    price: float
    volume: float


@dataclass(slots=True)
class OrderBook:
    """Стакан: списки bid/ask уровней, отсортированные по цене."""

    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.best_ask - self.best_bid


@dataclass(frozen=True, slots=True)
class Trade:
    """Одна публичная сделка."""

    price: float
    volume: float
    side: str  # "buy" или "sell" (taker side)
    timestamp: float


@dataclass(slots=True)
class TickerInfo:
    """Данные perpetual тикера (funding rate и др.)."""

    funding_rate: float = 0.0
    last_price: float = 0.0
    timestamp: float = 0.0


@dataclass(slots=True)
class MarketSnapshot:
    """Полный снимок рыночных данных — то, что получают боты."""

    btc_book: OrderBook = field(default_factory=OrderBook)
    eth_book: OrderBook = field(default_factory=OrderBook)
    btc_perp: TickerInfo = field(default_factory=TickerInfo)
    recent_trades: list[Trade] = field(default_factory=list)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Listener protocol — кто хочет получать обновления
# ---------------------------------------------------------------------------


class MarketDataListener(Protocol):
    """Интерфейс для получателей рыночных данных."""

    async def on_market_update(self, snapshot: MarketSnapshot) -> None: ...


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketDataConfig:
    """Параметры market data из params.yaml."""

    symbol: str
    leader_symbols: list[str]
    perpetual_symbol: str
    orderbook_depth: int
    trade_buffer_seconds: int
    reconnect_delay_seconds: int
    max_reconnect_attempts: int

    @staticmethod
    def from_yaml(path: str) -> MarketDataConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        md = raw["market_data"]
        return MarketDataConfig(
            symbol=md["symbol"],
            leader_symbols=list(md["leader_symbols"]),
            perpetual_symbol=md["perpetual_symbol"],
            orderbook_depth=md["orderbook_depth"],
            trade_buffer_seconds=md["trade_buffer_seconds"],
            reconnect_delay_seconds=md["reconnect_delay_seconds"],
            max_reconnect_attempts=md["max_reconnect_attempts"],
        )


# ---------------------------------------------------------------------------
# MarketDataStream
# ---------------------------------------------------------------------------


class MarketDataStream:
    """Единый WebSocket поток рыночных данных.

    Подписывается на все нужные стримы через ccxt async,
    хранит текущее состояние и уведомляет слушателей.
    """

    def __init__(self, config: MarketDataConfig) -> None:
        self._config = config

        # Текущее состояние
        self._btc_book = OrderBook()
        self._eth_book = OrderBook()
        self._btc_perp = TickerInfo()
        # Кольцевой буфер сделок — храним только последние N секунд
        self._trades: deque[Trade] = deque()

        # Слушатели
        self._listeners: list[MarketDataListener] = []

        # Управление жизненным циклом
        self._running = False
        # ccxt не имеет type stubs — Any неизбежен для exchange instance
        self._exchange: Any | None = None

    def add_listener(self, listener: MarketDataListener) -> None:
        self._listeners.append(listener)

    def snapshot(self) -> MarketSnapshot:
        """Текущий снимок без копирования — для быстрого чтения."""
        self._prune_old_trades()
        return MarketSnapshot(
            btc_book=self._btc_book,
            eth_book=self._eth_book,
            btc_perp=self._btc_perp,
            recent_trades=list(self._trades),
            timestamp=time.time(),
        )

    # ----- lifecycle -----

    async def start(self) -> None:
        """Запуск потоков данных. Блокирует до вызова stop()."""
        # Ленивый импорт ccxt — чтобы тесты работали без установленной биржи
        import ccxt.pro as ccxtpro

        self._exchange = ccxtpro.bybit({"enableRateLimit": True})
        self._running = True

        tasks = [
            self._watch_order_book(self._config.symbol, is_leader=False),
            self._watch_trades(self._config.symbol),
            self._watch_ticker(self._config.perpetual_symbol),
        ]
        if self._config.leader_symbols:
            tasks.append(
                self._watch_order_book(self._config.leader_symbols[0], is_leader=True)
            )
        else:
            logger.warning("No leader symbols configured, ETH lead-lag disabled")

        try:
            await asyncio.gather(*tasks)
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    # ----- watch loops -----

    async def _watch_order_book(self, symbol: str, *, is_leader: bool) -> None:
        assert self._exchange is not None
        exchange = self._exchange
        failures = 0
        while self._running:
            try:
                ob = await exchange.watch_order_book(
                    symbol, limit=self._config.orderbook_depth
                )
                book = self._parse_order_book(ob)
                if is_leader:
                    self._eth_book = book
                else:
                    self._btc_book = book
                await self._notify()
                failures = 0
            except Exception as exc:
                if not self._running:
                    break
                failures += 1
                logger.warning(
                    "watch_order_book %s failed (%d/%d): %s",
                    symbol, failures, self._config.max_reconnect_attempts, exc,
                )
                if failures >= self._config.max_reconnect_attempts:
                    logger.error("Max reconnects reached for order_book %s", symbol)
                    raise
                await asyncio.sleep(self._config.reconnect_delay_seconds)

    async def _watch_trades(self, symbol: str) -> None:
        assert self._exchange is not None
        exchange = self._exchange
        failures = 0
        while self._running:
            try:
                trades_raw = await exchange.watch_trades(symbol)
                for t in trades_raw:
                    trade = Trade(
                        price=float(t["price"]),
                        volume=float(t["amount"]),
                        side=str(t["side"]),
                        timestamp=float(t["timestamp"]) / 1000.0,
                    )
                    self._trades.append(trade)
                self._prune_old_trades()
                await self._notify()
                failures = 0
            except Exception as exc:
                if not self._running:
                    break
                failures += 1
                logger.warning(
                    "watch_trades %s failed (%d/%d): %s",
                    symbol, failures, self._config.max_reconnect_attempts, exc,
                )
                if failures >= self._config.max_reconnect_attempts:
                    logger.error("Max reconnects reached for trades %s", symbol)
                    raise
                await asyncio.sleep(self._config.reconnect_delay_seconds)

    async def _watch_ticker(self, symbol: str) -> None:
        assert self._exchange is not None
        exchange = self._exchange
        failures = 0
        while self._running:
            try:
                ticker = await exchange.watch_ticker(symbol)
                funding = ticker.get("info", {})
                self._btc_perp = TickerInfo(
                    funding_rate=float(funding.get("fundingRate", 0.0)),
                    last_price=float(ticker.get("last", 0.0)),
                    timestamp=time.time(),
                )
                await self._notify()
                failures = 0
            except Exception as exc:
                if not self._running:
                    break
                failures += 1
                logger.warning(
                    "watch_ticker %s failed (%d/%d): %s",
                    symbol, failures, self._config.max_reconnect_attempts, exc,
                )
                if failures >= self._config.max_reconnect_attempts:
                    logger.error("Max reconnects reached for ticker %s", symbol)
                    raise
                await asyncio.sleep(self._config.reconnect_delay_seconds)

    # ----- helpers -----

    @staticmethod
    def _parse_order_book(raw: dict[str, Any]) -> OrderBook:
        bids = [
            OrderBookLevel(price=float(level[0]), volume=float(level[1]))
            for level in raw.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(level[0]), volume=float(level[1]))
            for level in raw.get("asks", [])
        ]
        ts = float(raw.get("timestamp", 0) or 0) / 1000.0
        return OrderBook(bids=bids, asks=asks, timestamp=ts)

    def _prune_old_trades(self) -> None:
        """Удаляет сделки старше trade_buffer_seconds."""
        cutoff = time.time() - self._config.trade_buffer_seconds
        while self._trades and self._trades[0].timestamp < cutoff:
            self._trades.popleft()

    async def _notify(self) -> None:
        """Уведомляет всех слушателей о новых данных."""
        if not self._listeners:
            return
        snap = self.snapshot()
        await asyncio.gather(
            *(listener.on_market_update(snap) for listener in self._listeners)
        )
