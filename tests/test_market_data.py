"""Тесты для core/market_data.py.

Тестируем data structures и логику без реального WebSocket.
"""

from __future__ import annotations

import time

from core.market_data import (
    MarketDataConfig,
    MarketDataStream,
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
    TickerInfo,
    Trade,
)

# ---------------------------------------------------------------------------
# OrderBook tests
# ---------------------------------------------------------------------------


class TestOrderBook:
    def test_empty_book_defaults(self) -> None:
        book = OrderBook()
        assert book.best_bid == 0.0
        assert book.best_ask == 0.0
        assert book.mid_price == 0.0
        assert book.spread == 0.0

    def test_properties(self) -> None:
        book = OrderBook(
            bids=[OrderBookLevel(100.0, 1.0), OrderBookLevel(99.0, 2.0)],
            asks=[OrderBookLevel(101.0, 1.5), OrderBookLevel(102.0, 0.5)],
        )
        assert book.best_bid == 100.0
        assert book.best_ask == 101.0
        assert book.mid_price == 100.5
        assert book.spread == 1.0


# ---------------------------------------------------------------------------
# MarketDataConfig tests
# ---------------------------------------------------------------------------


class TestMarketDataConfig:
    def test_from_yaml(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "params.yaml"
        p.write_text(
            """
market_data:
  symbol: "BTC/USDT"
  leader_symbols:
    - "ETH/USDT"
  perpetual_symbol: "BTC/USDT:USDT"
  orderbook_depth: 50
  trade_buffer_seconds: 60
  reconnect_delay_seconds: 5
  max_reconnect_attempts: 10
"""
        )
        cfg = MarketDataConfig.from_yaml(str(p))
        assert cfg.symbol == "BTC/USDT"
        assert cfg.leader_symbols == ["ETH/USDT"]
        assert cfg.orderbook_depth == 50
        assert cfg.trade_buffer_seconds == 60


# ---------------------------------------------------------------------------
# parse_order_book tests
# ---------------------------------------------------------------------------


class TestParseOrderBook:
    def test_parse(self) -> None:
        raw = {
            "bids": [[67000.0, 1.5], [66999.0, 2.0]],
            "asks": [[67001.0, 0.8], [67002.0, 1.2]],
            "timestamp": 1709640000000,
        }
        book = MarketDataStream._parse_order_book(raw)
        assert len(book.bids) == 2
        assert len(book.asks) == 2
        assert book.bids[0].price == 67000.0
        assert book.asks[0].volume == 0.8
        assert book.timestamp == 1709640000.0

    def test_parse_empty(self) -> None:
        raw: dict[str, object] = {"bids": [], "asks": [], "timestamp": 0}
        book = MarketDataStream._parse_order_book(raw)
        assert book.bids == []
        assert book.asks == []


# ---------------------------------------------------------------------------
# Snapshot and trade pruning tests
# ---------------------------------------------------------------------------


def _make_config() -> MarketDataConfig:
    return MarketDataConfig(
        symbol="BTC/USDT",
        leader_symbols=["ETH/USDT"],
        perpetual_symbol="BTC/USDT:USDT",
        orderbook_depth=50,
        trade_buffer_seconds=10,
        reconnect_delay_seconds=5,
        max_reconnect_attempts=10,
    )


class TestSnapshot:
    def test_snapshot_returns_current_state(self) -> None:
        stream = MarketDataStream(_make_config())
        snap = stream.snapshot()
        assert isinstance(snap, MarketSnapshot)
        assert snap.btc_book.best_bid == 0.0
        assert snap.recent_trades == []

    def test_prune_old_trades(self) -> None:
        stream = MarketDataStream(_make_config())
        now = time.time()

        # Старая сделка — должна быть удалена (старше 10 секунд)
        old_trade = Trade(price=67000.0, volume=0.1, side="buy", timestamp=now - 20)
        # Свежая сделка — должна остаться
        new_trade = Trade(price=67001.0, volume=0.2, side="sell", timestamp=now - 1)

        stream._trades.append(old_trade)
        stream._trades.append(new_trade)

        snap = stream.snapshot()
        assert len(snap.recent_trades) == 1
        assert snap.recent_trades[0].price == 67001.0


class TestTickerInfo:
    def test_defaults(self) -> None:
        ticker = TickerInfo()
        assert ticker.funding_rate == 0.0
        assert ticker.last_price == 0.0
