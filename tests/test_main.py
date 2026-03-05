"""Тесты для main.py — end-to-end без реального WebSocket."""

from __future__ import annotations

import pytest

from core.market_data import (
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
    TickerInfo,
    Trade,
)
from main import TradingBot


def _make_snapshot(
    btc_mid: float = 67000.0,
    eth_mid: float = 3500.0,
    ts: float = 1000.0,
) -> MarketSnapshot:
    return MarketSnapshot(
        btc_book=OrderBook(
            bids=[OrderBookLevel(btc_mid - 0.5, 5.0)],
            asks=[OrderBookLevel(btc_mid + 0.5, 5.0)],
        ),
        eth_book=OrderBook(
            bids=[OrderBookLevel(eth_mid - 0.5, 3.0)],
            asks=[OrderBookLevel(eth_mid + 0.5, 3.0)],
        ),
        btc_perp=TickerInfo(funding_rate=0.0001),
        recent_trades=[
            Trade(btc_mid, 0.5, "buy", ts - 1),
            Trade(btc_mid, 0.3, "sell", ts - 1),
        ],
        timestamp=ts,
    )


class TestTradingBot:
    def test_creation(self) -> None:
        bot = TradingBot("config/params.yaml")
        assert len(bot._population.bots) == 20
        assert bot._population.generation == 0

    @pytest.mark.asyncio
    async def test_on_market_update(self, tmp_path: object) -> None:
        bot = TradingBot("config/params.yaml")
        bot._running = True

        # Один тик — не должен падать
        snap = _make_snapshot()
        await bot.on_market_update(snap)

    @pytest.mark.asyncio
    async def test_multiple_updates(self, tmp_path: object) -> None:
        bot = TradingBot("config/params.yaml")
        bot._running = True

        # Несколько тиков с разными ценами
        for i in range(10):
            snap = _make_snapshot(
                btc_mid=67000.0 + i * 10,
                ts=1000.0 + i,
            )
            await bot.on_market_update(snap)

    @pytest.mark.asyncio
    async def test_not_running_skips(self) -> None:
        bot = TradingBot("config/params.yaml")
        bot._running = False
        # Не должно ничего делать
        await bot.on_market_update(_make_snapshot())

    @pytest.mark.asyncio
    async def test_zero_price_skips(self) -> None:
        bot = TradingBot("config/params.yaml")
        bot._running = True
        # Пустой стакан — mid_price = 0
        snap = MarketSnapshot(timestamp=1000.0)
        await bot.on_market_update(snap)
