"""Управление популяцией ботов — создание, эволюция, отслеживание.

Каждый бот — это DecisionEngine + PaperExecutor + история сделок.
Population связывает все части и управляет жизненным циклом.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.decision import BotParams, DecisionEngine
from core.signals import Signal, SignalValues
from evolution.fitness import (
    FitnessConfig,
    FitnessMetrics,
    TradeRecord,
    compute_fitness,
    compute_metrics,
)
from evolution.genetics import GeneticsConfig, evolve, random_params
from paper.simulator import PaperExecutor, PaperTradingConfig

# ---------------------------------------------------------------------------
# Bot — один участник популяции
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Bot:
    """Один бот в популяции: параметры + движок + баланс + история."""

    bot_id: int
    params: BotParams
    engine: DecisionEngine
    executor: PaperExecutor
    trades: list[TradeRecord] = field(default_factory=list)
    generation: int = 0

    @property
    def fitness_metrics(self) -> FitnessMetrics:
        return compute_metrics(self.trades)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


class Population:
    """Популяция ботов — создание, торговля, эволюция."""

    def __init__(
        self,
        size: int,
        paper_config: PaperTradingConfig,
        fitness_config: FitnessConfig,
        genetics_config: GeneticsConfig,
        evolution_trigger_trades: int,
    ) -> None:
        self._paper_config = paper_config
        self._fitness_config = fitness_config
        self._genetics_config = genetics_config
        self._evolution_trigger = evolution_trigger_trades
        self._generation = 0
        self._total_trades = 0

        self.bots = self._create_bots(size)

    def _create_bots(self, size: int) -> list[Bot]:
        """Создаёт начальную популяцию со случайными параметрами."""
        bots: list[Bot] = []
        for i in range(size):
            params = random_params()
            bots.append(
                Bot(
                    bot_id=i,
                    params=params,
                    engine=DecisionEngine(params),
                    executor=PaperExecutor(self._paper_config),
                    generation=self._generation,
                )
            )
        return bots

    def process_signals(
        self, values: SignalValues, current_price: float, spread: float, now: float
    ) -> list[tuple[int, Signal]]:
        """Обрабатывает сигналы для всех ботов. Возвращает (bot_id, signal)."""
        results: list[tuple[int, Signal]] = []

        for bot in self.bots:
            signal = self._process_bot(bot, values, current_price, spread, now)
            results.append((bot.bot_id, signal))

        return results

    def should_evolve(self) -> bool:
        """Пора ли запускать эволюцию."""
        return self._total_trades >= self._evolution_trigger

    def run_evolution(self) -> None:
        """Запуск цикла эволюции."""
        scores = [
            compute_fitness(bot.fitness_metrics, self._fitness_config)
            for bot in self.bots
        ]
        params_list = [bot.params for bot in self.bots]
        new_params = evolve(params_list, scores, self._genetics_config)

        self._generation += 1
        self._total_trades = 0

        new_bots: list[Bot] = []
        for i, params in enumerate(new_params):
            new_bots.append(
                Bot(
                    bot_id=i,
                    params=params,
                    engine=DecisionEngine(params),
                    executor=PaperExecutor(self._paper_config),
                    generation=self._generation,
                )
            )
        self.bots = new_bots

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
            entry_time = engine.position.entry_time
            pnl = engine.close_position(current_price)
            bot.executor.apply_pnl(pnl)
            bot.trades.append(TradeRecord(
                pnl=pnl,
                entry_time=entry_time,
                exit_time=now,
            ))
            self._total_trades += 1
            return Signal.HOLD

        # Проверяем вход
        if engine.position is None:
            signal = engine.compute_entry_signal(values, current_price, now)
            if signal != Signal.HOLD:
                engine.open_position(signal, current_price, now, size_usd=1000.0)
            return signal

        return Signal.HOLD
