"""Тесты для core/decision.py."""

from __future__ import annotations

from core.decision import BotParams, DecisionEngine
from core.signals import Signal, SignalValues


def _default_params() -> BotParams:
    return BotParams(
        imbalance_threshold=0.65,
        flow_threshold=1.5,
        take_profit_usd=20.0,
        stop_loss_usd=10.0,
        max_hold_seconds=60.0,
        flow_window_seconds=5.0,
        eth_window_seconds=10.0,
        eth_move_threshold=0.0003,
        leader_weight=0.5,
    )


def _neutral_values() -> SignalValues:
    """Нейтральные значения — не должны вызывать сигнал."""
    return SignalValues(
        imbalance=0.5,
        flow_ratio=1.0,
        eth_lead=0.0,
        btc_change=0.0,
        funding_rate=0.0,
        spread=1.0,
        volatility=0.001,
    )


# ---------------------------------------------------------------------------
# Entry signals
# ---------------------------------------------------------------------------


class TestEntrySignal:
    def test_neutral_gives_hold(self) -> None:
        engine = DecisionEngine(_default_params())
        signal = engine.compute_entry_signal(_neutral_values(), 67000.0, 1000.0)
        assert signal == Signal.HOLD

    def test_strong_bid_imbalance_gives_long(self) -> None:
        engine = DecisionEngine(_default_params())
        values = SignalValues(
            imbalance=0.80,  # выше порога 0.65
            flow_ratio=2.0,  # выше порога 1.5
            eth_lead=0.0,
            btc_change=0.0,
            funding_rate=0.0,
            spread=1.0,
            volatility=0.001,
        )
        signal = engine.compute_entry_signal(values, 67000.0, 1000.0)
        assert signal == Signal.LONG

    def test_strong_ask_imbalance_gives_short(self) -> None:
        engine = DecisionEngine(_default_params())
        values = SignalValues(
            imbalance=0.20,  # ниже (1 - 0.65) = 0.35
            flow_ratio=0.5,  # ниже 1/1.5 ≈ 0.67
            eth_lead=0.0,
            btc_change=0.0,
            funding_rate=0.0,
            spread=1.0,
            volatility=0.001,
        )
        signal = engine.compute_entry_signal(values, 67000.0, 1000.0)
        assert signal == Signal.SHORT


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------


class TestPositionManagement:
    def test_open_and_close_long(self) -> None:
        engine = DecisionEngine(_default_params())
        engine.open_position(Signal.LONG, 67000.0, 1000.0, 1000.0)

        assert engine.position is not None
        assert engine.position.side == Signal.LONG

        # Цена выросла на $67 → PnL = 67 * (1000/67000) ≈ $1
        pnl = engine.close_position(67067.0)
        assert pnl > 0
        assert engine.position is None

    def test_open_and_close_short(self) -> None:
        engine = DecisionEngine(_default_params())
        engine.open_position(Signal.SHORT, 67000.0, 1000.0, 1000.0)

        # Цена упала → профит для шорта
        pnl = engine.close_position(66900.0)
        assert pnl > 0

    def test_close_without_position(self) -> None:
        engine = DecisionEngine(_default_params())
        pnl = engine.close_position(67000.0)
        assert pnl == 0.0


# ---------------------------------------------------------------------------
# Exit conditions
# ---------------------------------------------------------------------------


class TestExitConditions:
    def test_take_profit_long(self) -> None:
        engine = DecisionEngine(_default_params())
        engine.open_position(Signal.LONG, 67000.0, 1000.0, 10000.0)

        # TP = $20, size = $10000 → нужен рост ~$134 (20 / (10000/67000))
        assert not engine.should_exit(67100.0, 1010.0)  # +$14.9, не достигнут
        assert engine.should_exit(67200.0, 1010.0)  # +$29.8, достигнут

    def test_stop_loss_long(self) -> None:
        engine = DecisionEngine(_default_params())
        engine.open_position(Signal.LONG, 67000.0, 1000.0, 10000.0)

        # SL = $10, цена упала
        assert engine.should_exit(66930.0, 1010.0)  # -$10.4

    def test_timeout(self) -> None:
        engine = DecisionEngine(_default_params())
        engine.open_position(Signal.LONG, 67000.0, 1000.0, 10000.0)

        assert not engine.should_exit(67000.0, 1050.0)  # 50s < 60s
        assert engine.should_exit(67000.0, 1060.0)  # 60s >= 60s

    def test_no_exit_without_position(self) -> None:
        engine = DecisionEngine(_default_params())
        assert not engine.should_exit(67000.0, 1000.0)
