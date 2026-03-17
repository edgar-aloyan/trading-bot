"""Управление популяцией ботов — создание, эволюция, отслеживание.

Stateless: всё состояние в PostgreSQL (StateDB).
В памяти только кэш для быстрого tick processing (позиции, params, balance).
При каждом изменении состояния — запись в DB из main.py.

Поддерживает два режима:
- taker (default): мгновенное исполнение, taker fees
- maker: лимитные ордера на вход, maker fees при fill
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.decision import BotParams, DecisionEngine, FilterConfig, Position
from core.signals import Signal, SignalValues
from evolution.fitness import FitnessConfig, TradeRecord, compute_fitness
from evolution.genetics import (
    MAKER_PARAM_RANGES,
    PARAM_RANGES,
    GeneticsConfig,
    ParamRange,
    evolve,
    random_params,
)
from paper.simulator import PaperTradingConfig
from storage.database import BotRow, StateDBProtocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bot — in-memory кэш для быстрого tick processing
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Bot:
    """Один бот: параметры + движок (кэш позиции) + кэш баланса."""

    bot_id: int
    params: BotParams
    engine: DecisionEngine
    generation: int = 0
    balance: float = 0.0
    entry_signals: dict[str, object] | None = None
    pending_order: PendingOrder | None = None


# ---------------------------------------------------------------------------
# Events — результаты tick processing для записи в DB
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """Закрытая сделка — для записи в DB."""

    bot_id: int
    generation: int
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    fees: float
    entry_time: float
    exit_time: float
    entry_signals: dict[str, object] | None
    exit_signals: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class OpenedPosition:
    """Открытая позиция — для записи в DB."""

    bot_id: int
    side: str
    entry_price: float
    entry_time: float
    size_usd: float
    signals: dict[str, object]


@dataclass(frozen=True, slots=True)
class PendingOrder:
    """Лимитный ордер ожидающий fill — in-memory state."""

    bot_id: int
    side: str
    limit_price: float
    placed_time: float
    size_usd: float
    signals: dict[str, object]


# ---------------------------------------------------------------------------
# Param name sets per mode
# ---------------------------------------------------------------------------

TAKER_PARAM_NAMES = (
    "micro_price_threshold", "delta_threshold", "take_profit_usd",
    "stop_loss_usd", "max_hold_seconds", "basis_threshold", "basis_weight",
)

MAKER_PARAM_NAMES = TAKER_PARAM_NAMES + (
    "limit_offset_usd", "cancel_timeout_seconds", "exit_order_mode",
)

# Обратная совместимость
PARAM_NAMES = TAKER_PARAM_NAMES


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


class Population:
    """Популяция ботов — stateless, состояние в DB."""

    def __init__(
        self,
        size: int,
        paper_config: PaperTradingConfig,
        fitness_config: FitnessConfig,
        genetics_config: GeneticsConfig,
        min_trades_per_bot: int,
        filter_config: FilterConfig,
        db: StateDBProtocol,
        *,
        population_id: int = 1,
        mode: str = "taker",
    ) -> None:
        self._size = size
        self._population_id = population_id
        self._mode = mode
        self._paper_config = paper_config
        self._fitness_config = fitness_config
        self._genetics_config = genetics_config
        self._filter_config = filter_config
        self._min_trades_per_bot = min_trades_per_bot
        self._db = db

        # Параметры зависят от режима
        self._param_ranges: dict[str, ParamRange] = (
            MAKER_PARAM_RANGES if mode == "maker" else PARAM_RANGES
        )
        self._param_names: tuple[str, ...] = (
            MAKER_PARAM_NAMES if mode == "maker" else TAKER_PARAM_NAMES
        )

        # In-memory кэш — заполняется в init_from_db()
        self.bots: list[Bot] = []
        self._generation = 0
        self._total_trades = 0
        # Счётчик сделок на бота в текущем поколении (in-memory кэш)
        self._bot_trade_counts: dict[int, int] = {}

        # Результаты последнего tick — main.py читает и пишет в DB
        self.last_closed_trades: list[ClosedTrade] = []
        self.last_opened_positions: list[OpenedPosition] = []
        self.last_pending_orders: list[PendingOrder] = []
        self.last_removed_order_ids: list[int] = []

    async def init_from_db(self) -> None:
        """Загружает состояние из DB или создаёт новую популяцию."""
        pid = self._population_id
        gen, total = await self._db.get_generation(population_id=pid)
        self._generation = gen
        self._total_trades = total

        bot_rows = await self._db.load_bots(population_id=pid)
        if bot_rows:
            logger.info(
                "Pop %d: loaded %d bots from DB, generation=%d, total_trades=%d",
                pid, len(bot_rows), gen, total,
            )
            self.bots = self._bots_from_rows(bot_rows)
        else:
            logger.info(
                "Pop %d: no bots in DB, creating new population of %d",
                pid, self._size,
            )
            self.bots = self._create_new_bots(self._size)
            await self._save_bots_to_db()

        # Восстанавливаем балансы и счётчики сделок из DB
        self._bot_trade_counts = await self._db.get_trade_counts(
            self._generation, population_id=pid,
        )
        for bot in self.bots:
            bot.balance = await self._db.get_bot_balance(
                bot.bot_id, self._paper_config.initial_balance_usd,
                self._generation, population_id=pid,
            )

        # Восстанавливаем открытые позиции из DB
        positions = await self._db.load_positions(population_id=pid)
        for pos in positions:
            found = self._find_bot(pos.bot_id)
            if found is not None:
                found.engine.position = Position(
                    side=Signal(pos.side),
                    entry_price=pos.entry_price,
                    entry_time=pos.entry_time,
                    size_usd=pos.size_usd,
                )
                found.entry_signals = pos.entry_signals
                logger.info(
                    "Restored position for bot %d: %s @ %.2f",
                    pos.bot_id, pos.side, pos.entry_price,
                )

        # Восстанавливаем pending orders из DB (maker mode)
        if self._mode == "maker":
            orders = await self._db.load_pending_orders(population_id=pid)
            for order in orders:
                found = self._find_bot(order.bot_id)
                if found is not None:
                    found.pending_order = PendingOrder(
                        bot_id=order.bot_id,
                        side=order.side,
                        limit_price=order.limit_price,
                        placed_time=order.placed_time,
                        size_usd=order.size_usd,
                        signals=order.entry_signals or {},
                    )
                    logger.info(
                        "Restored pending order for bot %d: %s @ %.2f",
                        order.bot_id, order.side, order.limit_price,
                    )

    def process_signals(
        self, values: SignalValues, current_price: float, spread: float, now: float,
    ) -> list[tuple[int, Signal]]:
        """Обрабатывает сигналы для всех ботов. Возвращает (bot_id, signal).

        Побочные эффекты доступны через:
        - self.last_closed_trades (закрытые сделки)
        - self.last_opened_positions (открытые позиции)
        - self.last_pending_orders (новые ордера, maker mode)
        - self.last_removed_order_ids (заполненные/отменённые ордера)
        """
        self.last_closed_trades = []
        self.last_opened_positions = []
        self.last_pending_orders = []
        self.last_removed_order_ids = []
        results: list[tuple[int, Signal]] = []

        for bot in self.bots:
            signal = self._process_bot(bot, values, current_price, spread, now)
            results.append((bot.bot_id, signal))

        return results

    def should_evolve(self) -> bool:
        """Пора ли запускать эволюцию.

        Триггер: суммарное число сделок популяции >= size * min_trades_per_bot.
        Не ждём пока каждый бот наторгует — fitness penalty (min_trades_for_full_fitness)
        уже штрафует ботов с малым числом сделок пропорционально.
        """
        if not self.bots:
            return False
        total = sum(self._bot_trade_counts.values())
        return total >= len(self.bots) * self._min_trades_per_bot

    async def run_evolution(self) -> None:
        """Запуск цикла эволюции — читает trades из DB, эволюционирует, сохраняет."""
        pid = self._population_id
        scores: list[float] = []
        for bot in self.bots:
            trade_rows = await self._db.get_trades_for_bot(
                bot.bot_id, self._generation, population_id=pid,
            )
            records = [
                TradeRecord(pnl=t.pnl, entry_time=t.entry_time, exit_time=t.exit_time)
                for t in trade_rows
            ]
            scores.append(compute_fitness(
                records, self._paper_config.position_size_usd, self._fitness_config,
            ))

        params_list = [bot.params for bot in self.bots]
        new_params = evolve(
            params_list, scores, self._genetics_config, self._param_ranges,
        )

        best_idx = scores.index(max(scores)) if scores else 0
        avg_fitness = sum(scores) / len(scores) if scores else 0.0
        best_bot = self.bots[best_idx]
        best_params = {
            k: getattr(best_bot.params, k) for k in self._param_names
        }

        new_generation = self._generation + 1

        # Новые боты
        new_bots: list[Bot] = []
        new_bot_rows: list[BotRow] = []
        for i, params in enumerate(new_params):
            new_bots.append(Bot(
                bot_id=i,
                params=params,
                engine=DecisionEngine(params, self._filter_config),
                generation=new_generation,
                balance=self._paper_config.initial_balance_usd,
            ))
            new_bot_rows.append(BotRow(
                bot_id=i,
                generation=new_generation,
                params={k: getattr(params, k) for k in self._param_names},
            ))

        # Атомарная запись в DB (включая очистку pending_orders)
        await self._db.run_evolution_tx(
            generation=new_generation,
            total_trades=0,
            bots=new_bot_rows,
            best_fitness=scores[best_idx] if scores else 0.0,
            avg_fitness=avg_fitness,
            best_params=best_params,
            population_id=pid,
        )

        self._generation = new_generation
        self._total_trades = 0
        self._bot_trade_counts = {}
        self.bots = new_bots

        logger.info(
            "Pop %d: evolution complete, generation=%d",
            self._population_id, self._generation,
        )

    def on_trade_closed(self, ct: ClosedTrade) -> None:
        """Обновляет кэш после записи сделки в DB."""
        bot = self._find_bot(ct.bot_id)
        if bot is not None:
            # ct.pnl уже net (gross - fees), fees вычитать не нужно
            bot.balance += ct.pnl
        self._total_trades += 1
        self._bot_trade_counts[ct.bot_id] = (
            self._bot_trade_counts.get(ct.bot_id, 0) + 1
        )

    @property
    def population_id(self) -> int:
        return self._population_id

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def total_trades(self) -> int:
        return self._total_trades

    @property
    def mode(self) -> str:
        return self._mode

    # ----- internal -----

    def _process_bot(
        self,
        bot: Bot,
        values: SignalValues,
        current_price: float,
        spread: float,
        now: float,
    ) -> Signal:
        """Обрабатывает один тик для одного бота."""
        engine = bot.engine

        # Maker mode: проверяем pending order (fill / timeout)
        if bot.pending_order is not None:
            return self._process_pending_order(bot, current_price, now)

        # Проверяем выход из позиции
        if engine.position is not None and engine.should_exit(current_price, now):
            return self._close_position(
                bot, engine, values, current_price, spread, now,
            )

        # Проверяем вход
        if engine.position is None:
            return self._try_entry(
                bot, engine, values, current_price, spread, now,
            )

        return Signal.HOLD

    def _process_pending_order(
        self, bot: Bot, current_price: float, now: float,
    ) -> Signal:
        """Проверяет fill или timeout для pending order."""
        order = bot.pending_order
        assert order is not None

        # Проверяем fill: цена дошла до лимита
        filled = (
            (order.side == "LONG" and current_price <= order.limit_price)
            or (order.side == "SHORT" and current_price >= order.limit_price)
        )

        if filled:
            # Ордер исполнен — открываем позицию по лимитной цене
            signal = Signal(order.side)
            bot.engine.open_position(
                signal, order.limit_price, now,
                size_usd=order.size_usd,
            )
            bot.entry_signals = order.signals
            self.last_opened_positions.append(OpenedPosition(
                bot_id=bot.bot_id,
                side=order.side,
                entry_price=order.limit_price,
                entry_time=now,
                size_usd=order.size_usd,
                signals=order.signals,
            ))
            bot.pending_order = None
            self.last_removed_order_ids.append(bot.bot_id)
            return Signal.HOLD

        # Проверяем timeout
        if now - order.placed_time >= bot.params.cancel_timeout_seconds:
            bot.pending_order = None
            self.last_removed_order_ids.append(bot.bot_id)

        return Signal.HOLD

    def _close_position(
        self,
        bot: Bot,
        engine: DecisionEngine,
        values: SignalValues,
        current_price: float,
        spread: float,
        now: float,
    ) -> Signal:
        """Закрывает позицию бота с правильным расчётом fees."""
        pos = engine.position
        assert pos is not None

        # Определяем тип выхода для расчёта fees
        is_maker_tp = False
        if self._mode == "maker" and bot.params.exit_order_mode > 0.5:
            # TP exit с maker fee — если PnL достигает take_profit
            unrealized = engine._unrealized_pnl(current_price)
            if unrealized >= bot.params.take_profit_usd:
                is_maker_tp = True

        pnl = engine.close_position(current_price)
        fees = self._compute_fees(spread, current_price, maker=is_maker_tp)
        net_pnl = pnl - fees
        self.last_closed_trades.append(ClosedTrade(
            bot_id=bot.bot_id,
            generation=bot.generation,
            side=pos.side.value,
            entry_price=pos.entry_price,
            exit_price=current_price,
            pnl=net_pnl,
            fees=fees,
            entry_time=pos.entry_time,
            exit_time=now,
            entry_signals=bot.entry_signals,
            exit_signals=_values_to_dict(values),
        ))
        bot.entry_signals = None
        return Signal.HOLD

    def _try_entry(
        self,
        bot: Bot,
        engine: DecisionEngine,
        values: SignalValues,
        current_price: float,
        spread: float,
        now: float,
    ) -> Signal:
        """Пробует открыть позицию или поставить лимитный ордер."""
        if bot.balance < self._paper_config.position_size_usd:
            return Signal.HOLD

        signal = engine.compute_entry_signal(values, current_price, now)
        if signal == Signal.HOLD:
            return Signal.HOLD

        signals_dict = _values_to_dict(values)
        size_usd = self._paper_config.position_size_usd

        if self._mode == "maker":
            # Maker mode: ставим лимитный ордер с отступом от цены
            offset = bot.params.limit_offset_usd
            if signal == Signal.LONG:
                limit_price = current_price - offset
            else:
                limit_price = current_price + offset

            pending = PendingOrder(
                bot_id=bot.bot_id,
                side=signal.value,
                limit_price=limit_price,
                placed_time=now,
                size_usd=size_usd,
                signals=signals_dict,
            )
            bot.pending_order = pending
            self.last_pending_orders.append(pending)
        else:
            # Taker mode: мгновенное исполнение
            engine.open_position(signal, current_price, now, size_usd=size_usd)
            bot.entry_signals = signals_dict
            self.last_opened_positions.append(OpenedPosition(
                bot_id=bot.bot_id,
                side=signal.value,
                entry_price=current_price,
                entry_time=now,
                size_usd=size_usd,
                signals=signals_dict,
            ))

        return signal

    def _compute_fees(
        self, spread: float, price: float, *, maker: bool = False,
    ) -> float:
        """Комиссия: maker fee (без slippage) или taker fee + slippage."""
        cfg = self._paper_config
        if maker:
            return cfg.position_size_usd * cfg.maker_fee
        slippage_usd = cfg.slippage_factor * spread
        slippage_pct = slippage_usd / price if price > 0 else 0.0
        return cfg.position_size_usd * (cfg.taker_fee + slippage_pct)

    def _create_new_bots(self, size: int) -> list[Bot]:
        bots: list[Bot] = []
        for i in range(size):
            params = random_params(self._param_ranges)
            bots.append(Bot(
                bot_id=i,
                params=params,
                engine=DecisionEngine(params, self._filter_config),
                generation=self._generation,
                balance=self._paper_config.initial_balance_usd,
            ))
        return bots

    def _bots_from_rows(self, rows: list[BotRow]) -> list[Bot]:
        bots: list[Bot] = []
        for row in rows:
            # params хранятся как JSONB — значения всегда float
            # .get() для совместимости со старыми записями без maker params
            raw = row.params
            params = BotParams(**{
                k: float(str(raw[k])) if k in raw else 0.0
                for k in self._param_names
            })
            bots.append(Bot(
                bot_id=row.bot_id,
                params=params,
                engine=DecisionEngine(params, self._filter_config),
                generation=row.generation,
            ))
        return bots

    async def _save_bots_to_db(self) -> None:
        await self._db.save_bots([
            BotRow(
                bot_id=b.bot_id,
                generation=b.generation,
                params={k: getattr(b.params, k) for k in self._param_names},
            )
            for b in self.bots
        ], population_id=self._population_id)

    def _find_bot(self, bot_id: int) -> Bot | None:
        for bot in self.bots:
            if bot.bot_id == bot_id:
                return bot
        return None


def _values_to_dict(values: SignalValues) -> dict[str, object]:
    """Конвертирует SignalValues в dict для JSONB."""
    return {
        "micro_price_deviation": values.micro_price_deviation,
        "volume_delta": values.volume_delta,
        "basis": values.basis,
        "volatility": values.volatility,
        "spread": values.spread,
    }
