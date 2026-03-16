"""Main entrypoint — связывает все модули и запускает бота.

Цикл работы:
1. MarketDataStream получает данные с биржи
2. SignalComputer вычисляет сырые сигналы
3. Каждая популяция обрабатывает сигналы для своих ботов
4. Voting определяет итоговый сигнал (per-population)
5. DB записывает всё (единственный stateful компонент)
6. При достижении trigger — эволюция (per-population)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time

from core.decision import FilterConfig
from core.market_data import MarketDataConfig, MarketDataStream, MarketSnapshot
from core.signals import SignalComputer, SignalsConfig
from ensemble.voting import VotingConfig, compute_vote
from evolution.fitness import FitnessConfig
from evolution.genetics import GeneticsConfig
from evolution.population import Population
from paper.simulator import PaperTradingConfig
from storage.database import PositionRow, StateDB, StateDBProtocol, TradeRow

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/params.yaml"


class TradingBot:
    """Оркестратор — связывает все модули. Stateless — состояние в DB."""

    def __init__(self, db: StateDBProtocol, config_path: str = CONFIG_PATH) -> None:
        self._config_path = config_path
        self._db = db

        # Загружаем все конфиги
        self._market_config = MarketDataConfig.from_yaml(config_path)
        self._signals_config = SignalsConfig.from_yaml(config_path)
        self._voting_config = VotingConfig.from_yaml(config_path)
        self._fitness_config = FitnessConfig.from_yaml(config_path)
        self._genetics_config = GeneticsConfig.from_yaml(config_path)
        self._paper_config = PaperTradingConfig.from_yaml(config_path)
        self._filter_config = FilterConfig.from_yaml(config_path)

        # Читаем evolution params напрямую
        import yaml

        with open(config_path) as f:
            raw = yaml.safe_load(f)
        evo = raw["evolution"]
        self._pop_size: int = evo["population_size"]
        self._min_trades_per_bot: int = evo["min_trades_per_bot"]
        self._evolution_ready_ratio: float = evo["evolution_ready_ratio"]

        # Эксперименты: список популяций для параллельного тестирования
        self._experiments: list[dict[str, object]] = raw.get(
            "experiments",
            [{"population_id": 1, "name": "default"}],
        )

        # Создаём stateless компоненты
        self._stream = MarketDataStream(self._market_config)
        self._signal_computer = SignalComputer(self._signals_config)
        self._populations: list[Population] = []
        self._running = False
        # Защита от одновременных on_market_update (4 watch-цикла в gather)
        self._lock = asyncio.Lock()

    async def _init_populations(self) -> None:
        """Создаёт и загружает все популяции из DB."""
        for exp in self._experiments:
            pop_id = int(str(exp.get("population_id", 1)))
            pop_name = str(exp.get("name", f"pop-{pop_id}"))
            pop = Population(
                size=self._pop_size,
                paper_config=self._paper_config,
                fitness_config=self._fitness_config,
                genetics_config=self._genetics_config,
                min_trades_per_bot=self._min_trades_per_bot,
                evolution_ready_ratio=self._evolution_ready_ratio,
                filter_config=self._filter_config,
                db=self._db,
                population_id=pop_id,
            )
            await pop.init_from_db()
            self._populations.append(pop)
            logger.info(
                "Population '%s' (id=%d) ready: %d bots, generation=%d",
                pop_name, pop_id, len(pop.bots), pop.generation,
            )

    # Обратная совместимость: тесты используют _init_population / _population
    async def _init_population(self) -> None:
        """Обратная совместимость — инициализирует одну популяцию."""
        await self._init_populations()

    @property
    def _population(self) -> Population | None:
        """Обратная совместимость — первая популяция."""
        return self._populations[0] if self._populations else None

    async def start(self) -> None:
        """Запуск бота."""
        await self._init_populations()

        self._running = True
        total_bots = sum(len(p.bots) for p in self._populations)
        logger.info(
            "Trading bot started: %d populations, %d total bots, symbol=%s",
            len(self._populations), total_bots,
            self._market_config.symbol,
        )

        # Регистрируемся как слушатель market data
        self._stream.add_listener(self)

        # Запускаем WebSocket потоки (блокирует до stop)
        await self._stream.start()

    async def stop(self) -> None:
        """Остановка бота."""
        self._running = False
        await self._stream.stop()
        gens = [p.generation for p in self._populations]
        logger.info("Trading bot stopped: generations=%s", gens)

    async def on_market_update(self, snapshot: MarketSnapshot) -> None:
        """Callback от MarketDataStream — основной цикл обработки.

        Lock нужен: 4 watch-цикла шлют обновления параллельно через
        asyncio.gather. Без lock'а process_signals() одного вызова может
        сбросить last_closed_trades пока другой ещё не дописал их в DB.
        """
        if not self._running or not self._populations:
            return

        async with self._lock:
            try:
                await self._process_snapshot(snapshot)
            except Exception:
                logger.exception("Error in on_market_update")

    async def _process_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Обработка одного снапшота для всех популяций.

        NOTE: process_signals() mutates in-memory state before DB writes.
        If a DB write fails, memory diverges from DB until restart (DB wins
        on reload). Acceptable for paper trading — self-heals on restart.
        """
        now = snapshot.timestamp or time.time()

        # 1. Вычисляем сырые сигналы (один раз для всех популяций)
        values = self._signal_computer.update(snapshot)
        current_price = snapshot.btc_book.mid_price
        spread = snapshot.btc_book.spread

        if current_price == 0:
            return

        # 2. Каждая популяция обрабатывает сигналы независимо
        for pop in self._populations:
            await self._process_population(pop, values, current_price, spread, now)

    async def _process_population(
        self, pop: Population, values: object, current_price: float,
        spread: float, now: float,
    ) -> None:
        """Обработка одного снапшота для одной популяции."""
        from core.signals import SignalValues
        assert isinstance(values, SignalValues)

        pid = pop.population_id

        # Каждый бот обрабатывает сигналы
        bot_signals = pop.process_signals(values, current_price, spread, now)

        # Записываем открытые позиции в DB (батчом)
        if pop.last_opened_positions:
            positions = [
                PositionRow(
                    bot_id=op.bot_id,
                    side=op.side,
                    entry_price=op.entry_price,
                    entry_time=op.entry_time,
                    size_usd=op.size_usd,
                    entry_signals=op.signals,
                )
                for op in pop.last_opened_positions
            ]
            await self._db.open_positions_batch(positions, population_id=pid)

        # Записываем закрытые сделки в DB (батчом в одной транзакции)
        if pop.last_closed_trades:
            trades = [
                TradeRow(
                    bot_id=ct.bot_id,
                    generation=ct.generation,
                    side=ct.side,
                    entry_price=ct.entry_price,
                    exit_price=ct.exit_price,
                    pnl=ct.pnl,
                    fees=ct.fees,
                    entry_time=ct.entry_time,
                    exit_time=ct.exit_time,
                    entry_signals=ct.entry_signals,
                    exit_signals=ct.exit_signals,
                )
                for ct in pop.last_closed_trades
            ]
            await self._db.close_trades_batch(trades, population_id=pid)
            for ct in pop.last_closed_trades:
                pop.on_trade_closed(ct)
            logger.info(
                "Pop %d: %d trades closed this tick",
                pid, len(pop.last_closed_trades),
            )

        # Голосование
        vote = compute_vote(bot_signals, self._voting_config)

        if vote.signal.value != "HOLD":
            logger.info(
                "Pop %d ensemble: %s confidence=%.2f price=%.2f",
                pid, vote.signal.value, vote.confidence, current_price,
            )

        # Эволюция если пора
        if pop.should_evolve():
            await pop.run_evolution()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    dsn = os.environ.get(
        "DATABASE_URL",
        "postgresql://trading:trading_dev@localhost:5432/trading",
    )
    db = StateDB(dsn)
    await db.connect()

    bot = TradingBot(db)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))

    try:
        await bot.start()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Bot crashed")
        raise
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
