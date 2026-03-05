"""Тесты для core/signals.py."""

from __future__ import annotations

import time

from core.market_data import (
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
    TickerInfo,
    Trade,
)
from core.signals import (
    SignalComputer,
    SignalsConfig,
    compute_imbalance,
    compute_trade_flow,
)

# ---------------------------------------------------------------------------
# compute_imbalance
# ---------------------------------------------------------------------------


class TestComputeImbalance:
    def test_balanced_book(self) -> None:
        book = OrderBook(
            bids=[OrderBookLevel(100.0, 5.0)],
            asks=[OrderBookLevel(101.0, 5.0)],
        )
        assert compute_imbalance(book, levels=10) == 0.5

    def test_bid_heavy(self) -> None:
        book = OrderBook(
            bids=[OrderBookLevel(100.0, 8.0)],
            asks=[OrderBookLevel(101.0, 2.0)],
        )
        assert compute_imbalance(book, levels=10) == 0.8

    def test_ask_heavy(self) -> None:
        book = OrderBook(
            bids=[OrderBookLevel(100.0, 2.0)],
            asks=[OrderBookLevel(101.0, 8.0)],
        )
        assert compute_imbalance(book, levels=10) == 0.2

    def test_empty_book(self) -> None:
        book = OrderBook()
        assert compute_imbalance(book, levels=10) == 0.5

    def test_respects_levels_limit(self) -> None:
        """Должен учитывать только первые N уровней."""
        book = OrderBook(
            bids=[
                OrderBookLevel(100.0, 1.0),
                OrderBookLevel(99.0, 1.0),
                OrderBookLevel(98.0, 100.0),  # не должен учитываться при levels=2
            ],
            asks=[OrderBookLevel(101.0, 2.0)],
        )
        # levels=2: bid_vol=2, ask_vol=2 → 0.5
        assert compute_imbalance(book, levels=2) == 0.5


# ---------------------------------------------------------------------------
# compute_trade_flow
# ---------------------------------------------------------------------------


class TestComputeTradeFlow:
    def test_balanced_flow(self) -> None:
        now = time.time()
        trades = [
            Trade(100.0, 1.0, "buy", now - 1),
            Trade(100.0, 1.0, "sell", now - 1),
        ]
        assert compute_trade_flow(trades, window_seconds=5, now=now) == 1.0

    def test_buy_dominant(self) -> None:
        now = time.time()
        trades = [
            Trade(100.0, 3.0, "buy", now - 1),
            Trade(100.0, 1.0, "sell", now - 1),
        ]
        assert compute_trade_flow(trades, window_seconds=5, now=now) == 3.0

    def test_no_trades(self) -> None:
        now = time.time()
        assert compute_trade_flow([], window_seconds=5, now=now) == 1.0

    def test_only_buys(self) -> None:
        now = time.time()
        trades = [Trade(100.0, 5.0, "buy", now - 1)]
        # Нет продаж — возвращаем объём покупок
        assert compute_trade_flow(trades, window_seconds=5, now=now) == 5.0

    def test_only_buys_capped(self) -> None:
        """При отсутствии продаж — cap на 10.0 чтобы не раздувать score."""
        now = time.time()
        trades = [Trade(100.0, 50.0, "buy", now - 1)]
        assert compute_trade_flow(trades, window_seconds=5, now=now) == 10.0

    def test_old_trades_excluded(self) -> None:
        """Сделки старше окна не учитываются."""
        now = time.time()
        trades = [
            Trade(100.0, 10.0, "buy", now - 100),  # старая
            Trade(100.0, 1.0, "sell", now - 1),  # свежая
        ]
        # Покупка за окном, только sell = 1.0 → buy_vol=0, sell_vol=1 → ratio=0
        assert compute_trade_flow(trades, window_seconds=5, now=now) == 0.0


# ---------------------------------------------------------------------------
# SignalComputer
# ---------------------------------------------------------------------------


def _make_config() -> SignalsConfig:
    return SignalsConfig(
        orderbook_levels=10,
        flow_window_seconds=5,
        eth_window_seconds=10,
        volatility_window_seconds=60,
    )


def _make_snapshot(
    btc_mid: float = 67000.0,
    eth_mid: float = 3500.0,
    spread: float = 1.0,
    ts: float = 0.0,
) -> MarketSnapshot:
    half_spread = spread / 2
    return MarketSnapshot(
        btc_book=OrderBook(
            bids=[OrderBookLevel(btc_mid - half_spread, 5.0)],
            asks=[OrderBookLevel(btc_mid + half_spread, 5.0)],
        ),
        eth_book=OrderBook(
            bids=[OrderBookLevel(eth_mid - 0.5, 3.0)],
            asks=[OrderBookLevel(eth_mid + 0.5, 3.0)],
        ),
        btc_perp=TickerInfo(funding_rate=0.0001),
        recent_trades=[],
        timestamp=ts,
    )


class TestSignalComputer:
    def test_initial_update(self) -> None:
        comp = SignalComputer(_make_config())
        snap = _make_snapshot(ts=1000.0)
        values = comp.update(snap)

        assert values.imbalance == 0.5  # balanced book
        assert values.flow_ratio == 1.0  # no trades
        assert values.eth_lead == 0.0  # no history yet
        assert values.funding_rate == 0.0001
        assert values.spread == 1.0

    def test_eth_lead_detection(self) -> None:
        """ETH двигается, BTC нет — должен быть ненулевой eth_lead."""
        comp = SignalComputer(_make_config())

        # Первый снимок
        comp.update(_make_snapshot(btc_mid=67000, eth_mid=3500, ts=1000.0))

        # Через 5 секунд ETH вырос на ~1%, BTC на месте
        snap2 = _make_snapshot(btc_mid=67000, eth_mid=3535, ts=1005.0)
        values = comp.update(snap2)

        assert values.eth_lead > 0.009  # ~1%
        assert abs(values.btc_change) < 0.001  # BTC не двигался

    def test_volatility_calculation(self) -> None:
        """Волатильность должна быть > 0 при изменении цен."""
        comp = SignalComputer(_make_config())

        prices = [67000, 67010, 66990, 67020, 66980, 67030]
        for i, price in enumerate(prices):
            snap = _make_snapshot(btc_mid=float(price), ts=1000.0 + i)
            values = comp.update(snap)

        assert values.volatility > 0

    def test_volatility_zero_for_flat(self) -> None:
        """Волатильность 0 при одинаковых ценах."""
        comp = SignalComputer(_make_config())

        for i in range(5):
            snap = _make_snapshot(btc_mid=67000.0, ts=1000.0 + i)
            values = comp.update(snap)

        assert values.volatility == 0.0

    def test_config_from_yaml(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "params.yaml"
        p.write_text(
            """
signals:
  orderbook_levels: 10
  imbalance_threshold: 0.65
  flow_threshold: 1.5
  flow_window_seconds: 5
  eth_window_seconds: 10
  eth_move_threshold: 0.0003
  funding_positive_threshold: 0.0001
  funding_negative_threshold: -0.0001
filters:
  max_spread_usd: 2.0
  min_volatility: 0.0001
  max_volatility: 0.01
  volatility_window_seconds: 60
"""
        )
        cfg = SignalsConfig.from_yaml(str(p))
        assert cfg.orderbook_levels == 10
        assert cfg.flow_window_seconds == 5
        assert cfg.volatility_window_seconds == 60
