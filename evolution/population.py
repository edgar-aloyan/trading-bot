"""Управление популяцией ботов — создание, эволюция, отслеживание.

Stateless: всё состояние в PostgreSQL (StateDB).
В памяти только кэш для быстрого tick processing (позиции, params, balance).
При каждом изменении состояния — запись в DB из main.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.decision import BotParams, DecisionEngine, FilterConfig, Position
from core.signals import Signal, SignalValues
from evolution.fitness import FitnessConfig, TradeRecord, compute_fitness
from evolution.genetics import GeneticsConfig, evolve, random_params
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


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

# Имена параметров BotParams — для сериализации
PARAM_NAMES = (
    "imbalance_threshold", "flow_threshold", "take_profit_usd",
    "stop_loss_usd", "max_hold_seconds", "eth_move_threshold", "leader_weight",
)


class Population:
    """Популяция ботов — stateless, состояние в DB."""

    def __init__(
        self,
        size: int,
        paper_config: PaperTradingConfig,
        fitness_config: FitnessConfig,
        genetics_config: GeneticsConfig,
        min_trades_per_bot: int,
        evolution_ready_ratio: float,
        filter_config: FilterConfig,
        db: StateDBProtocol,
    ) -> None:
        self._size = size
        self._paper_config = paper_config
        self._fitness_config = fitness_config
        self._genetics_config = genetics_config
        self._filter_config = filter_config
        self._min_trades_per_bot = min_trades_per_bot
        self._evolution_ready_ratio = evolution_ready_ratio
        self._db = db

        # In-memory кэш — заполняется в init_from_db()
        self.bots: list[Bot] = []
        self._generation = 0
        self._total_trades = 0
        # Счётчик сделок на бота в текущем поколении (in-memory кэш)
        self._bot_trade_counts: dict[int, int] = {}

        # Результаты последнего tick — main.py читает и пишет в DB
        self.last_closed_trades: list[ClosedTrade] = []
        self.last_opened_positions: list[OpenedPosition] = []

    async def init_from_db(self) -> None:
        """Загружает состояние из DB или создаёт новую популяцию."""
        gen, total = await self._db.get_generation()
        self._generation = gen
        self._total_trades = total

        bot_rows = await self._db.load_bots()
        if bot_rows:
            logger.info(
                "Loaded %d bots from DB, generation=%d, total_trades=%d",
                len(bot_rows), gen, total,
            )
            self.bots = self._bots_from_rows(bot_rows)
        else:
            logger.info("No bots in DB, creating new population of %d", self._size)
            self.bots = self._create_new_bots(self._size)
            await self._save_bots_to_db()

        # Восстанавливаем балансы и счётчики сделок из DB
        self._bot_trade_counts = await self._db.get_trade_counts(
            self._generation,
        )
        for bot in self.bots:
            bot.balance = await self._db.get_bot_balance(
                bot.bot_id, self._paper_config.initial_balance_usd,
                self._generation,
            )

        # Восстанавливаем открытые позиции из DB
        positions = await self._db.load_positions()
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

    def process_signals(
        self, values: SignalValues, current_price: float, spread: float, now: float,
    ) -> list[tuple[int, Signal]]:
        """Обрабатывает сигналы для всех ботов. Возвращает (bot_id, signal).

        Побочные эффекты доступны через:
        - self.last_closed_trades (закрытые сделки)
        - self.last_opened_positions (открытые позиции)
        """
        self.last_closed_trades = []
        self.last_opened_positions = []
        results: list[tuple[int, Signal]] = []

        for bot in self.bots:
            signal = self._process_bot(bot, values, current_price, spread, now)
            results.append((bot.bot_id, signal))

        return results

    def should_evolve(self) -> bool:
        """Пора ли запускать эволюцию.

        Эволюция когда evolution_ready_ratio ботов набрали min_trades_per_bot.
        """
        if not self.bots:
            return False
        ready = sum(
            1 for c in self._bot_trade_counts.values()
            if c >= self._min_trades_per_bot
        )
        return ready >= len(self.bots) * self._evolution_ready_ratio

    async def run_evolution(self) -> None:
        """Запуск цикла эволюции — читает trades из DB, эволюционирует, сохраняет."""
        scores: list[float] = []
        for bot in self.bots:
            trade_rows = await self._db.get_trades_for_bot(
                bot.bot_id, self._generation,
            )
            records = [
                TradeRecord(pnl=t.pnl, entry_time=t.entry_time, exit_time=t.exit_time)
                for t in trade_rows
            ]
            scores.append(compute_fitness(
                records, self._paper_config.position_size_usd, self._fitness_config,
            ))

        params_list = [bot.params for bot in self.bots]
        new_params = evolve(params_list, scores, self._genetics_config)

        best_idx = scores.index(max(scores)) if scores else 0
        avg_fitness = sum(scores) / len(scores) if scores else 0.0
        best_bot = self.bots[best_idx]
        best_params = {k: getattr(best_bot.params, k) for k in PARAM_NAMES}

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
                params={k: getattr(params, k) for k in PARAM_NAMES},
            ))

        # Атомарная запись в DB
        await self._db.run_evolution_tx(
            generation=new_generation,
            total_trades=0,
            bots=new_bot_rows,
            best_fitness=scores[best_idx] if scores else 0.0,
            avg_fitness=avg_fitness,
            best_params=best_params,
        )

        self._generation = new_generation
        self._total_trades = 0
        self._bot_trade_counts = {}
        self.bots = new_bots

        logger.info("Evolution complete: generation=%d", self._generation)

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
    def generation(self) -> int:
        return self._generation

    @property
    def total_trades(self) -> int:
        return self._total_trades

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

        # Проверяем выход из позиции
        if engine.position is not None and engine.should_exit(current_price, now):
            pos = engine.position
            pnl = engine.close_position(current_price)
            fees = self._compute_fees(spread, current_price)
            # PnL уже не включает fees — вычитаем отдельно
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

        # Проверяем вход
        if engine.position is None:
            # Проверяем что баланс позволяет открыть позицию
            if bot.balance < self._paper_config.position_size_usd:
                return Signal.HOLD

            signal = engine.compute_entry_signal(values, current_price, now)
            if signal != Signal.HOLD:
                engine.open_position(
                    signal, current_price, now,
                    size_usd=self._paper_config.position_size_usd,
                )
                signals_dict = _values_to_dict(values)
                bot.entry_signals = signals_dict
                self.last_opened_positions.append(OpenedPosition(
                    bot_id=bot.bot_id,
                    side=signal.value,
                    entry_price=current_price,
                    entry_time=now,
                    size_usd=self._paper_config.position_size_usd,
                    signals=signals_dict,
                ))
            return signal

        return Signal.HOLD

    def _compute_fees(self, spread: float, price: float) -> float:
        """Комиссия: taker fee + slippage."""
        cfg = self._paper_config
        slippage_usd = cfg.slippage_factor * spread
        slippage_pct = slippage_usd / price if price > 0 else 0.0
        return cfg.position_size_usd * (cfg.taker_fee + slippage_pct)

    def _create_new_bots(self, size: int) -> list[Bot]:
        bots: list[Bot] = []
        for i in range(size):
            params = random_params()
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
            raw = row.params
            params = BotParams(**{k: float(str(raw[k])) for k in PARAM_NAMES})
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
                params={k: getattr(b.params, k) for k in PARAM_NAMES},
            )
            for b in self.bots
        ])

    def _find_bot(self, bot_id: int) -> Bot | None:
        for bot in self.bots:
            if bot.bot_id == bot_id:
                return bot
        return None


def _values_to_dict(values: SignalValues) -> dict[str, object]:
    """Конвертирует SignalValues в dict для JSONB."""
    return {
        "imbalance": values.imbalance,
        "flow_ratio": values.flow_ratio,
        "eth_lead": values.eth_lead,
        "btc_change": values.btc_change,
        "volatility": values.volatility,
        "spread": values.spread,
    }
