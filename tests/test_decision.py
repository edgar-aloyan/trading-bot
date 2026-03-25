"""Тесты для core/decision.py."""

from __future__ import annotations

from core.decision import BotParams, DecisionEngine, FilterConfig
from core.signals import Signal, SignalValues


def _default_filters() -> FilterConfig:
    return FilterConfig(
        max_spread_usd=2.0,
        min_volatility=0.0001,
        max_volatility=0.01,
    )


def _default_params() -> BotParams:
    return BotParams(
        micro_sensitivity=0.0001,
        micro_weight=0.5,
        delta_sensitivity=0.3,
        delta_weight=0.5,
        take_profit_usd=20.0,
        stop_loss_usd=10.0,
        max_hold_seconds=60.0,
        basis_sensitivity=0.001,
        basis_weight=0.0,
        funding_sensitivity=0.0001,
        funding_weight=0.0,
        micro_mode=1.0,  # AND по умолчанию (как было)
        delta_mode=1.0,
        basis_mode=1.0,
        funding_mode=1.0,
    )


def _neutral_values() -> SignalValues:
    """Нейтральные значения — не должны вызывать сигнал."""
    return SignalValues(
        micro_price_deviation=0.0,
        volume_delta=0.0,
        basis=0.0,
        funding_rate=0.0,
        spread=1.0,
        volatility=0.001,
    )


# ---------------------------------------------------------------------------
# Entry signals
# ---------------------------------------------------------------------------


class TestEntrySignal:
    def test_neutral_gives_hold(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        signal = engine.compute_entry_signal(_neutral_values(), 67000.0, 1000.0)
        assert signal == Signal.HOLD

    def test_strong_buy_pressure_gives_long(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        values = SignalValues(
            micro_price_deviation=0.0005,  # tanh(0.0005/0.0001) ≈ 1.0
            volume_delta=0.6,  # tanh(0.6/0.3) ≈ 0.96
            basis=0.0,
            funding_rate=0.0,
            spread=1.0,
            volatility=0.001,
        )
        signal = engine.compute_entry_signal(values, 67000.0, 1000.0)
        assert signal == Signal.LONG

    def test_strong_sell_pressure_gives_short(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        values = SignalValues(
            micro_price_deviation=-0.0005,  # tanh(-5) ≈ -1.0
            volume_delta=-0.6,  # tanh(-2) ≈ -0.96
            basis=0.0,
            funding_rate=0.0,
            spread=1.0,
            volatility=0.001,
        )
        signal = engine.compute_entry_signal(values, 67000.0, 1000.0)
        assert signal == Signal.SHORT

    def test_conflicting_signals_penalized(self) -> None:
        """Soft AND: micro LONG + delta SHORT → не LONG (сигналы конфликтуют)."""
        engine = DecisionEngine(_default_params(), _default_filters())
        values = SignalValues(
            micro_price_deviation=0.0005,  # сильный LONG
            volume_delta=-0.6,  # сильный SHORT — конфликт
            basis=0.0,
            funding_rate=0.0,
            spread=1.0,
            volatility=0.001,
        )
        signal = engine.compute_entry_signal(values, 67000.0, 1000.0)
        # Конфликтующие сигналы дают SHORT (delta против), не LONG
        assert signal == Signal.SHORT

    def test_weight_zero_disables_signal(self) -> None:
        """Weight=0 полностью выключает сигнал."""
        params = BotParams(
            micro_sensitivity=0.0001,
            micro_weight=0.0,  # micro выключен
            delta_sensitivity=0.3,
            delta_weight=0.5,
            take_profit_usd=20.0,
            stop_loss_usd=10.0,
            max_hold_seconds=60.0,
            basis_sensitivity=0.001,
            basis_weight=0.0,
            funding_sensitivity=0.0001,
            funding_weight=0.0,
            micro_mode=1.0, delta_mode=1.0,
            basis_mode=1.0, funding_mode=1.0,
        )
        engine = DecisionEngine(params, _default_filters())
        # Micro LONG, delta SHORT — но micro выключен, только delta решает
        values = SignalValues(
            micro_price_deviation=0.0005,
            volume_delta=-0.6,
            basis=0.0,
            funding_rate=0.0,
            spread=1.0,
            volatility=0.001,
        )
        signal = engine.compute_entry_signal(values, 67000.0, 1000.0)
        assert signal == Signal.SHORT

    def test_all_signals_agree_strong_score(self) -> None:
        """Все три сигнала согласны → сильнее чем два."""
        params = BotParams(
            micro_sensitivity=0.0001,
            micro_weight=0.5,
            delta_sensitivity=0.3,
            delta_weight=0.5,
            take_profit_usd=20.0,
            stop_loss_usd=10.0,
            max_hold_seconds=60.0,
            basis_sensitivity=0.001,
            basis_weight=0.5,
            funding_sensitivity=0.0001,
            funding_weight=0.0,
            micro_mode=1.0, delta_mode=1.0,
            basis_mode=1.0, funding_mode=1.0,
        )
        engine = DecisionEngine(params, _default_filters())
        # Все три LONG
        values_3 = SignalValues(
            micro_price_deviation=0.0005,
            volume_delta=0.6,
            basis=0.005,  # positive basis = бычий
            funding_rate=0.0,
            spread=1.0,
            volatility=0.001,
        )
        score_3 = engine._compute_score(values_3)
        # Два сигнала LONG, basis нейтральный
        values_2 = SignalValues(
            micro_price_deviation=0.0005,
            volume_delta=0.6,
            basis=0.0,
            funding_rate=0.0,
            spread=1.0,
            volatility=0.001,
        )
        score_2 = engine._compute_score(values_2)
        assert score_3 > score_2 > 0
        # Количественно: три согласных сигнала дают заметно больший score
        assert score_3 > score_2 * 1.2

    def test_or_mode_no_veto(self) -> None:
        """OR mode: конфликтующий сигнал не ветирует, а вычитает."""
        # micro=LONG (AND), delta=SHORT (OR) — delta не ветирует
        params_or = BotParams(
            micro_sensitivity=0.0001,
            micro_weight=0.5,
            delta_sensitivity=0.3,
            delta_weight=0.5,
            take_profit_usd=20.0,
            stop_loss_usd=10.0,
            max_hold_seconds=60.0,
            basis_sensitivity=0.001,
            basis_weight=0.0,
            funding_sensitivity=0.0001,
            funding_weight=0.0,
            micro_mode=1.0,  # AND
            delta_mode=0.0,  # OR — не ветирует
            basis_mode=0.0,
            funding_mode=0.0,
        )
        # micro=LONG (AND), delta=SHORT (AND) — delta ветирует
        params_and = BotParams(
            micro_sensitivity=0.0001,
            micro_weight=0.5,
            delta_sensitivity=0.3,
            delta_weight=0.5,
            take_profit_usd=20.0,
            stop_loss_usd=10.0,
            max_hold_seconds=60.0,
            basis_sensitivity=0.001,
            basis_weight=0.0,
            funding_sensitivity=0.0001,
            funding_weight=0.0,
            micro_mode=1.0,  # AND
            delta_mode=1.0,  # AND — ветирует
            basis_mode=0.0,
            funding_mode=0.0,
        )
        values = SignalValues(
            micro_price_deviation=0.0005,  # LONG
            volume_delta=-0.6,  # SHORT — конфликт
            basis=0.0,
            funding_rate=0.0,
            spread=1.0,
            volatility=0.001,
        )
        engine_or = DecisionEngine(params_or, _default_filters())
        engine_and = DecisionEngine(params_and, _default_filters())
        score_or = engine_or._compute_score(values)
        score_and = engine_and._compute_score(values)
        # OR mode: delta вычитает, но micro LONG доминирует → score > 0
        assert score_or > 0
        # AND mode: delta ветирует → score < 0
        assert score_and < 0


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------


class TestPositionManagement:
    def test_open_and_close_long(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        engine.open_position(Signal.LONG, 67000.0, 1000.0, 1000.0)

        assert engine.position is not None
        assert engine.position.side == Signal.LONG

        # Цена выросла на $67 → PnL = 67 * (1000/67000) ≈ $1
        pnl = engine.close_position(67067.0)
        assert pnl > 0
        assert engine.position is None

    def test_open_and_close_short(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        engine.open_position(Signal.SHORT, 67000.0, 1000.0, 1000.0)

        # Цена упала → профит для шорта
        pnl = engine.close_position(66900.0)
        assert pnl > 0

    def test_close_without_position(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        pnl = engine.close_position(67000.0)
        assert pnl == 0.0


# ---------------------------------------------------------------------------
# Exit conditions
# ---------------------------------------------------------------------------


class TestExitConditions:
    def test_take_profit_long(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        engine.open_position(Signal.LONG, 67000.0, 1000.0, 10000.0)

        # TP = $20, size = $10000 → нужен рост ~$134 (20 / (10000/67000))
        assert not engine.should_exit(67100.0, 1010.0)  # +$14.9, не достигнут
        assert engine.should_exit(67200.0, 1010.0)  # +$29.8, достигнут

    def test_stop_loss_long(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        engine.open_position(Signal.LONG, 67000.0, 1000.0, 10000.0)

        # SL = $10, цена упала
        assert engine.should_exit(66930.0, 1010.0)  # -$10.4

    def test_timeout(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        engine.open_position(Signal.LONG, 67000.0, 1000.0, 10000.0)

        assert not engine.should_exit(67000.0, 1050.0)  # 50s < 60s
        assert engine.should_exit(67000.0, 1060.0)  # 60s >= 60s

    def test_no_exit_without_position(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        assert not engine.should_exit(67000.0, 1000.0)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestFilters:
    def _long_values(self, **kwargs: float) -> SignalValues:
        """Сильные значения для LONG, можно переопределить поля."""
        defaults: dict[str, float] = dict(
            micro_price_deviation=0.0005,
            volume_delta=0.6,
            basis=0.0,
            funding_rate=0.0,
            spread=1.0,
            volatility=0.001,
        )
        defaults.update(kwargs)
        return SignalValues(**defaults)

    def test_wide_spread_blocks_entry(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        # Спред > max_spread_usd (2.0) — фильтр отсекает
        values = self._long_values(spread=3.0)
        assert engine.compute_entry_signal(values, 67000.0, 1.0) == Signal.HOLD

    def test_low_volatility_blocks_entry(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        # volatility < min_volatility (0.0001) — боковик
        values = self._long_values(volatility=0.00005)
        assert engine.compute_entry_signal(values, 67000.0, 1.0) == Signal.HOLD

    def test_high_volatility_blocks_entry(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        # volatility > max_volatility (0.01) — хаос
        values = self._long_values(volatility=0.02)
        assert engine.compute_entry_signal(values, 67000.0, 1.0) == Signal.HOLD

    def test_zero_volatility_blocks_entry(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        # volatility = 0 — нет данных на старте, не торговать
        values = self._long_values(volatility=0.0)
        assert engine.compute_entry_signal(values, 67000.0, 1.0) == Signal.HOLD

    def test_normal_conditions_pass(self) -> None:
        engine = DecisionEngine(_default_params(), _default_filters())
        # Всё в пределах нормы — сигнал должен пройти
        values = self._long_values()
        assert engine.compute_entry_signal(values, 67000.0, 1.0) == Signal.LONG
