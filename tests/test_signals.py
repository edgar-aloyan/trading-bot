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
    compute_basis,
    compute_micro_price_deviation,
    compute_volume_delta,
)

# ---------------------------------------------------------------------------
# compute_micro_price_deviation
# ---------------------------------------------------------------------------


class TestComputeMicroPriceDeviation:
    def test_balanced_book(self) -> None:
        """Равные объёмы на best bid/ask → deviation ≈ 0."""
        book = OrderBook(
            bids=[OrderBookLevel(100.0, 5.0)],
            asks=[OrderBookLevel(101.0, 5.0)],
        )
        assert abs(compute_micro_price_deviation(book)) < 1e-10

    def test_bid_heavy(self) -> None:
        """Больше объёма на bid → micro-price выше mid → положительное отклонение."""
        book = OrderBook(
            bids=[OrderBookLevel(100.0, 8.0)],
            asks=[OrderBookLevel(101.0, 2.0)],
        )
        dev = compute_micro_price_deviation(book)
        assert dev > 0

    def test_ask_heavy(self) -> None:
        """Больше объёма на ask → micro-price ниже mid → отрицательное отклонение."""
        book = OrderBook(
            bids=[OrderBookLevel(100.0, 2.0)],
            asks=[OrderBookLevel(101.0, 8.0)],
        )
        dev = compute_micro_price_deviation(book)
        assert dev < 0

    def test_empty_book(self) -> None:
        book = OrderBook()
        assert compute_micro_price_deviation(book) == 0.0

    def test_uses_only_best_level(self) -> None:
        """Micro-price использует только best bid/ask, глубокие уровни игнорируются."""
        book = OrderBook(
            bids=[
                OrderBookLevel(100.0, 5.0),
                OrderBookLevel(99.0, 100.0),  # не влияет
            ],
            asks=[OrderBookLevel(101.0, 5.0)],
        )
        # С равными объёмами на best level — deviation ≈ 0
        assert abs(compute_micro_price_deviation(book)) < 1e-10


# ---------------------------------------------------------------------------
# compute_volume_delta
# ---------------------------------------------------------------------------


class TestComputeVolumeDelta:
    def test_balanced_flow(self) -> None:
        now = time.time()
        trades = [
            Trade(100.0, 1.0, "buy", now - 1),
            Trade(100.0, 1.0, "sell", now - 1),
        ]
        assert compute_volume_delta(trades, window_seconds=5, now=now) == 0.0

    def test_buy_dominant(self) -> None:
        now = time.time()
        trades = [
            Trade(100.0, 3.0, "buy", now - 1),
            Trade(100.0, 1.0, "sell", now - 1),
        ]
        # (3-1)/(3+1) = 0.5
        assert compute_volume_delta(trades, window_seconds=5, now=now) == 0.5

    def test_no_trades(self) -> None:
        now = time.time()
        assert compute_volume_delta([], window_seconds=5, now=now) == 0.0

    def test_only_buys(self) -> None:
        """Только покупки → delta = 1.0 (максимум)."""
        now = time.time()
        trades = [Trade(100.0, 5.0, "buy", now - 1)]
        assert compute_volume_delta(trades, window_seconds=5, now=now) == 1.0

    def test_only_sells(self) -> None:
        """Только продажи → delta = -1.0 (минимум)."""
        now = time.time()
        trades = [Trade(100.0, 5.0, "sell", now - 1)]
        assert compute_volume_delta(trades, window_seconds=5, now=now) == -1.0

    def test_old_trades_excluded(self) -> None:
        """Сделки старше окна не учитываются."""
        now = time.time()
        trades = [
            Trade(100.0, 10.0, "buy", now - 100),  # старая
            Trade(100.0, 1.0, "sell", now - 1),  # свежая
        ]
        # Покупка за окном → delta = (0-1)/1 = -1.0
        assert compute_volume_delta(trades, window_seconds=5, now=now) == -1.0


# ---------------------------------------------------------------------------
# compute_basis
# ---------------------------------------------------------------------------


class TestComputeBasis:
    def test_positive_basis(self) -> None:
        """Перп дороже спота → положительный basis."""
        assert compute_basis(100.0, 100.1) > 0

    def test_negative_basis(self) -> None:
        """Перп дешевле спота → отрицательный basis."""
        assert compute_basis(100.0, 99.9) < 0

    def test_zero_prices(self) -> None:
        assert compute_basis(0.0, 100.0) == 0.0
        assert compute_basis(100.0, 0.0) == 0.0

    def test_equal_prices(self) -> None:
        assert compute_basis(100.0, 100.0) == 0.0


# ---------------------------------------------------------------------------
# SignalComputer
# ---------------------------------------------------------------------------


def _make_config() -> SignalsConfig:
    return SignalsConfig(
        flow_window_seconds=5,
        volatility_window_seconds=60,
    )


def _make_snapshot(
    btc_mid: float = 67000.0,
    spread: float = 1.0,
    perp_price: float = 67000.0,
    ts: float = 0.0,
) -> MarketSnapshot:
    half_spread = spread / 2
    return MarketSnapshot(
        btc_book=OrderBook(
            bids=[OrderBookLevel(btc_mid - half_spread, 5.0)],
            asks=[OrderBookLevel(btc_mid + half_spread, 5.0)],
        ),
        eth_book=OrderBook(),
        btc_perp=TickerInfo(funding_rate=0.0001, last_price=perp_price),
        recent_trades=[],
        timestamp=ts,
    )


class TestSignalComputer:
    def test_initial_update(self) -> None:
        comp = SignalComputer(_make_config())
        snap = _make_snapshot(ts=1000.0)
        values = comp.update(snap)

        assert abs(values.micro_price_deviation) < 1e-10  # balanced book
        assert values.volume_delta == 0.0  # no trades
        assert values.basis == 0.0  # perp == spot
        assert values.funding_rate == 0.0001
        assert values.spread == 1.0

    def test_basis_detection(self) -> None:
        """Perp торгуется с премией → положительный basis."""
        comp = SignalComputer(_make_config())
        snap = _make_snapshot(btc_mid=67000, perp_price=67067, ts=1000.0)
        values = comp.update(snap)

        assert values.basis > 0.0009  # ~0.1%

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
  flow_window_seconds: 5
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
        assert cfg.flow_window_seconds == 5
        assert cfg.volatility_window_seconds == 60
