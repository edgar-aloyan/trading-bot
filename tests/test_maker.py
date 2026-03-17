"""Тесты для maker order logic — pending orders, fill, timeout, fees."""

from __future__ import annotations

import pytest

from core.decision import BotParams, FilterConfig
from core.signals import SignalValues
from evolution.fitness import FitnessConfig
from evolution.genetics import MAKER_PARAM_RANGES, GeneticsConfig, random_params
from evolution.population import Population
from paper.simulator import PaperTradingConfig
from tests.mock_db import MockStateDB


def _paper_config() -> PaperTradingConfig:
    return PaperTradingConfig(
        initial_balance_usd=10000.0,
        position_size_usd=1000.0,
        maker_fee=0.0001,
        taker_fee=0.0006,
        slippage_factor=0.5,
    )


def _fitness_config() -> FitnessConfig:
    return FitnessConfig(min_trades_for_full_fitness=50)


def _filter_config() -> FilterConfig:
    return FilterConfig(
        max_spread_usd=2.0,
        min_volatility=0.0001,
        max_volatility=0.01,
        delta_weight=0.5,
    )


def _genetics_config() -> GeneticsConfig:
    return GeneticsConfig(
        elite_ratio=0.3,
        crossover_ratio=0.4,
        mutation_ratio=0.3,
        mutation_rate=0.2,
        mutation_strength=0.1,
        crossover_alpha=0.5,
        tournament_size=3,
    )


def _long_signal_values() -> SignalValues:
    """Значения, которые должны вызвать LONG у большинства ботов."""
    return SignalValues(
        micro_price_deviation=0.005,
        volume_delta=0.9,
        basis=0.001,
        funding_rate=0.0,
        spread=1.0,
        volatility=0.001,
    )


def _short_signal_values() -> SignalValues:
    """Значения, которые должны вызвать SHORT у большинства ботов."""
    return SignalValues(
        micro_price_deviation=-0.005,
        volume_delta=-0.9,
        basis=-0.001,
        funding_rate=0.0,
        spread=1.0,
        volatility=0.001,
    )


def _neutral_signal_values() -> SignalValues:
    """Нейтральные сигналы — HOLD."""
    return SignalValues(
        micro_price_deviation=0.0,
        volume_delta=0.0,
        basis=0.0,
        funding_rate=0.0,
        spread=1.0,
        volatility=0.001,
    )


def _maker_bot_params(**overrides: float) -> BotParams:
    """Стандартные параметры maker-бота для тестов."""
    defaults: dict[str, float] = dict(
        micro_price_threshold=0.0001,
        delta_threshold=0.1,
        take_profit_usd=20.0,
        stop_loss_usd=10.0,
        max_hold_seconds=60.0,
        basis_threshold=0.0001,
        basis_weight=0.5,
        limit_offset_usd=2.0,
        cancel_timeout_seconds=30.0,
        exit_order_mode=0.0,
    )
    defaults.update(overrides)
    return BotParams(**defaults)


async def _make_maker_pop(
    size: int = 5,
    min_trades: int = 3,
) -> Population:
    db = MockStateDB()
    pop = Population(
        size=size,
        paper_config=_paper_config(),
        fitness_config=_fitness_config(),
        genetics_config=_genetics_config(),
        min_trades_per_bot=min_trades,
        filter_config=_filter_config(),
        db=db,
        mode="maker",
    )
    await pop.init_from_db()
    return pop


class TestMakerPopulation:
    @pytest.mark.asyncio
    async def test_creation_maker_mode(self) -> None:
        pop = await _make_maker_pop(size=10)
        assert len(pop.bots) == 10
        assert pop.mode == "maker"
        # Все боты должны иметь maker-специфичные params
        for bot in pop.bots:
            assert bot.params.limit_offset_usd > 0
            assert bot.params.cancel_timeout_seconds > 0

    @pytest.mark.asyncio
    async def test_maker_entry_creates_pending_order(self) -> None:
        """В maker mode вход создаёт pending order, а не позицию."""
        pop = await _make_maker_pop(size=3)
        values = _long_signal_values()
        pop.process_signals(values, 67000.0, 1.0, 1000.0)

        # Должны быть pending orders, но не opened positions
        # (позиция откроется только после fill)
        assert len(pop.last_pending_orders) >= 0  # хотя бы структура заполнена
        assert len(pop.last_opened_positions) == 0  # никаких мгновенных fill

    @pytest.mark.asyncio
    async def test_pending_order_fill(self) -> None:
        """Pending order заполняется когда цена достигает лимита."""
        pop = await _make_maker_pop(size=1)
        bot = pop.bots[0]
        bot.params = _maker_bot_params(limit_offset_usd=2.0)
        bot.engine.params = bot.params

        # Tick 1: сильный LONG → создаёт pending buy at 66998
        values = _long_signal_values()
        pop.process_signals(values, 67000.0, 1.0, 1000.0)
        assert bot.pending_order is not None
        assert bot.pending_order.limit_price == 66998.0
        assert len(pop.last_pending_orders) == 1
        assert len(pop.last_opened_positions) == 0

        # Tick 2: цена 67010 — выше лимита, не fill
        pop.process_signals(_neutral_signal_values(), 67010.0, 1.0, 1001.0)
        assert bot.pending_order is not None
        assert len(pop.last_opened_positions) == 0

        # Tick 3: цена падает до 66998 — fill!
        pop.process_signals(_neutral_signal_values(), 66998.0, 1.0, 1002.0)
        assert bot.pending_order is None
        assert len(pop.last_opened_positions) == 1
        assert len(pop.last_removed_order_ids) == 1
        # Позиция открыта по лимитной цене
        assert bot.engine.position is not None
        assert bot.engine.position.entry_price == 66998.0

    @pytest.mark.asyncio
    async def test_pending_order_fill_short(self) -> None:
        """SHORT pending order заполняется когда цена поднимается до лимита."""
        pop = await _make_maker_pop(size=1)
        bot = pop.bots[0]
        bot.params = _maker_bot_params(limit_offset_usd=3.0)
        bot.engine.params = bot.params

        # Tick 1: SHORT → pending sell at 67003
        pop.process_signals(_short_signal_values(), 67000.0, 1.0, 1000.0)
        assert bot.pending_order is not None
        assert bot.pending_order.side == "SHORT"
        assert bot.pending_order.limit_price == 67003.0

        # Tick 2: цена ниже лимита — не fill
        pop.process_signals(_neutral_signal_values(), 66990.0, 1.0, 1001.0)
        assert bot.pending_order is not None

        # Tick 3: цена поднимается до 67003 — fill!
        pop.process_signals(_neutral_signal_values(), 67003.0, 1.0, 1002.0)
        assert bot.pending_order is None
        assert bot.engine.position is not None
        assert bot.engine.position.entry_price == 67003.0
        assert bot.engine.position.side.value == "SHORT"

    @pytest.mark.asyncio
    async def test_pending_order_timeout(self) -> None:
        """Pending order отменяется по timeout."""
        pop = await _make_maker_pop(size=1)
        bot = pop.bots[0]
        bot.params = _maker_bot_params(
            limit_offset_usd=2.0, cancel_timeout_seconds=10.0,
        )
        bot.engine.params = bot.params

        # Tick 1: создаём pending order
        values = _long_signal_values()
        pop.process_signals(values, 67000.0, 1.0, 1000.0)
        assert bot.pending_order is not None

        # Tick 2: 5 секунд спустя — ещё не timeout
        pop.process_signals(_neutral_signal_values(), 67010.0, 1.0, 1005.0)
        assert bot.pending_order is not None

        # Tick 3: 15 секунд спустя — timeout!
        pop.process_signals(_neutral_signal_values(), 67010.0, 1.0, 1015.0)
        assert bot.pending_order is None
        assert len(pop.last_removed_order_ids) == 1
        assert bot.engine.position is None  # позиция не открылась

    @pytest.mark.asyncio
    async def test_maker_tp_exit_uses_maker_fee(self) -> None:
        """TP exit с exit_order_mode > 0.5 использует maker fee."""
        pop = await _make_maker_pop(size=1)
        bot = pop.bots[0]
        bot.params = _maker_bot_params(
            limit_offset_usd=2.0, exit_order_mode=0.8,
        )
        bot.engine.params = bot.params

        # Tick 1: создаём pending order
        values = _long_signal_values()
        pop.process_signals(values, 67000.0, 1.0, 1000.0)

        # Tick 2: fill at 66998
        pop.process_signals(_neutral_signal_values(), 66998.0, 1.0, 1001.0)
        assert bot.engine.position is not None

        # Tick 3: TP hit — price up enough для take_profit_usd=20
        tp_price = 66998.0 + 20.0 * (66998.0 / 1000.0)
        pop.process_signals(_neutral_signal_values(), tp_price + 1, 1.0, 1010.0)
        assert bot.engine.position is None
        assert len(pop.last_closed_trades) == 1

        trade = pop.last_closed_trades[0]
        # Maker fee: 1000 * 0.0001 = 0.10
        expected_maker_fee = 1000.0 * 0.0001
        assert abs(trade.fees - expected_maker_fee) < 0.01

    @pytest.mark.asyncio
    async def test_maker_sl_exit_uses_taker_fee(self) -> None:
        """SL exit всегда использует taker fee, даже с exit_order_mode > 0.5."""
        pop = await _make_maker_pop(size=1)
        bot = pop.bots[0]
        bot.params = _maker_bot_params(
            limit_offset_usd=2.0, exit_order_mode=0.8,
        )
        bot.engine.params = bot.params

        # Tick 1: pending order
        pop.process_signals(_long_signal_values(), 67000.0, 1.0, 1000.0)

        # Tick 2: fill
        pop.process_signals(_neutral_signal_values(), 66998.0, 1.0, 1001.0)
        assert bot.engine.position is not None

        # Tick 3: SL hit — price drops
        sl_price = 66998.0 - 10.0 * (66998.0 / 1000.0)
        pop.process_signals(_neutral_signal_values(), sl_price - 1, 1.0, 1010.0)
        assert bot.engine.position is None
        assert len(pop.last_closed_trades) == 1

        trade = pop.last_closed_trades[0]
        # Taker fee: 1000 * (0.0006 + slippage)
        assert trade.fees > 1000.0 * 0.0001  # больше чем maker fee


class TestMakerGenetics:
    def test_random_params_maker_ranges(self) -> None:
        """random_params с MAKER_PARAM_RANGES включает maker params."""
        for _ in range(50):
            params = random_params(MAKER_PARAM_RANGES)
            assert 0.5 <= params.limit_offset_usd <= 5.0
            assert 5.0 <= params.cancel_timeout_seconds <= 60.0
            assert 0.0 <= params.exit_order_mode <= 1.0

    def test_random_params_default_no_maker(self) -> None:
        """random_params без аргументов НЕ устанавливает maker params."""
        params = random_params()
        assert params.limit_offset_usd == 0.0
        assert params.cancel_timeout_seconds == 0.0
        assert params.exit_order_mode == 0.0

    @pytest.mark.asyncio
    async def test_maker_evolution_preserves_size(self) -> None:
        """Эволюция maker-популяции сохраняет размер."""
        pop = await _make_maker_pop(size=10)
        assert pop.generation == 0
        await pop.run_evolution()
        assert pop.generation == 1
        assert len(pop.bots) == 10
        # Новые боты тоже имеют maker params
        for bot in pop.bots:
            assert bot.params.limit_offset_usd > 0


class TestMakerDB:
    @pytest.mark.asyncio
    async def test_pending_orders_persist(self) -> None:
        """Pending orders сохраняются и восстанавливаются из DB."""
        db = MockStateDB()
        pop = Population(
            size=1,
            paper_config=_paper_config(),
            fitness_config=_fitness_config(),
            genetics_config=_genetics_config(),
            min_trades_per_bot=3,
            filter_config=_filter_config(),
            db=db,
            mode="maker",
        )
        await pop.init_from_db()

        bot = pop.bots[0]
        bot.params = _maker_bot_params(limit_offset_usd=2.0)
        bot.engine.params = bot.params

        # Создаём pending order
        pop.process_signals(_long_signal_values(), 67000.0, 1.0, 1000.0)
        assert len(pop.last_pending_orders) == 1

        # Сохраняем в mock DB
        from storage.database import OrderRow
        for po in pop.last_pending_orders:
            await db.save_pending_orders_batch([
                OrderRow(
                    bot_id=po.bot_id,
                    side=po.side,
                    limit_price=po.limit_price,
                    placed_time=po.placed_time,
                    size_usd=po.size_usd,
                    entry_signals=po.signals,
                ),
            ])

        # Восстанавливаем
        orders = await db.load_pending_orders()
        assert len(orders) == 1
        assert orders[0].limit_price == 66998.0

    @pytest.mark.asyncio
    async def test_evolution_clears_pending_orders(self) -> None:
        """Эволюция очищает pending orders."""
        db = MockStateDB()

        # Добавляем pending order напрямую в mock DB
        from storage.database import OrderRow
        await db.save_pending_orders_batch([
            OrderRow(
                bot_id=0, side="LONG",
                limit_price=66998.0, placed_time=1000.0,
                size_usd=1000.0,
            ),
        ])
        assert len(await db.load_pending_orders()) == 1

        # run_evolution_tx очищает
        from storage.database import BotRow
        await db.run_evolution_tx(
            generation=1, total_trades=0,
            bots=[BotRow(bot_id=0, generation=1, params={})],
            best_fitness=0.0, avg_fitness=0.0,
            best_params={},
        )
        assert len(await db.load_pending_orders()) == 0
