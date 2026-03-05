"""Main entrypoint — связывает все модули и запускает бота.

Цикл работы:
1. MarketDataStream получает данные с биржи
2. SignalComputer вычисляет сырые сигналы
3. Population обрабатывает сигналы для каждого бота
4. Voting определяет итоговый сигнал
5. Logger записывает всё
6. При достижении trigger — эволюция
"""

from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import asdict

from core.decision import FilterConfig
from core.market_data import MarketDataConfig, MarketDataStream, MarketSnapshot
from core.signals import SignalComputer, SignalsConfig
from ensemble.voting import VotingConfig, compute_vote
from evolution.fitness import FitnessConfig, compute_fitness
from evolution.genetics import GeneticsConfig
from evolution.population import Population
from monitoring.logger import (
    EvolutionEvent,
    StructuredLogger,
    SystemEvent,
    TradeEvent,
)
from paper.simulator import PaperTradingConfig

CONFIG_PATH = "config/params.yaml"


class TradingBot:
    """Оркестратор — связывает все модули."""

    def __init__(self, config_path: str = CONFIG_PATH) -> None:
        self._config_path = config_path

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
        pop_size: int = evo["population_size"]
        trigger: int = evo["evolution_trigger_trades"]

        # Создаём компоненты
        self._stream = MarketDataStream(self._market_config)
        self._signal_computer = SignalComputer(self._signals_config)
        self._population = Population(
            size=pop_size,
            paper_config=self._paper_config,
            fitness_config=self._fitness_config,
            genetics_config=self._genetics_config,
            evolution_trigger_trades=trigger,
            filter_config=self._filter_config,
        )
        self._logger = StructuredLogger()
        self._running = False

    async def start(self) -> None:
        """Запуск бота."""
        self._running = True
        self._logger.log_system(SystemEvent(
            event="system_start",
            timestamp=time.time(),
            message="Trading bot started",
            details={
                "population_size": len(self._population.bots),
                "symbol": self._market_config.symbol,
            },
        ))

        # Регистрируемся как слушатель market data
        self._stream.add_listener(self)

        # Запускаем WebSocket потоки (блокирует до stop)
        await self._stream.start()

    async def stop(self) -> None:
        """Остановка бота."""
        self._running = False
        await self._stream.stop()
        self._logger.log_system(SystemEvent(
            event="system_stop",
            timestamp=time.time(),
            message="Trading bot stopped",
            details={"generation": self._population.generation},
        ))

    async def on_market_update(self, snapshot: MarketSnapshot) -> None:
        """Callback от MarketDataStream — основной цикл обработки."""
        if not self._running:
            return

        try:
            self._process_snapshot(snapshot)
        except Exception as exc:
            self._logger.log_system(SystemEvent(
                event="processing_error",
                timestamp=time.time(),
                message=f"Error in on_market_update: {exc}",
                details={"type": type(exc).__name__},
            ))

    def _process_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Обработка одного снапшота — вынесена для try/except."""
        now = snapshot.timestamp or time.time()

        # 1. Вычисляем сырые сигналы
        values = self._signal_computer.update(snapshot)
        current_price = snapshot.btc_book.mid_price
        spread = snapshot.btc_book.spread

        if current_price == 0:
            return

        # 2. Каждый бот обрабатывает сигналы
        bot_signals = self._population.process_signals(
            values, current_price, spread, now
        )

        # 3. Логируем закрытые сделки
        for ct in self._population.last_closed_trades:
            bot = self._population.bots[ct.bot_id]
            self._logger.log_trade(TradeEvent(
                event="trade_closed",
                timestamp=now,
                bot_id=ct.bot_id,
                generation=ct.generation,
                params={
                    k: getattr(bot.params, k)
                    for k in [
                        "imbalance_threshold", "flow_threshold",
                        "take_profit_usd", "stop_loss_usd",
                    ]
                },
                signals=asdict(values),
                trade={
                    "side": ct.side,
                    "entry_price": ct.entry_price,
                    "exit_price": ct.exit_price,
                    "pnl": ct.pnl,
                    "hold_seconds": ct.exit_time - ct.entry_time,
                },
            ))

        # 4. Голосование
        vote = compute_vote(bot_signals, self._voting_config)

        # 5. Логируем если есть сигнал
        if vote.signal.value != "HOLD":
            self._logger.log_raw({
                "event": "ensemble_signal",
                "timestamp": now,
                "signal": vote.signal.value,
                "long_ratio": vote.long_ratio,
                "short_ratio": vote.short_ratio,
                "confidence": vote.confidence,
                "generation": self._population.generation,
                "price": current_price,
                "signals": asdict(values),
            })

        # 6. Эволюция если пора
        if self._population.should_evolve():
            self._run_evolution(now)

    def _run_evolution(self, now: float) -> None:
        """Запуск эволюции с логированием."""
        # Собираем метрики до эволюции
        scores = [
            compute_fitness(bot.fitness_metrics, self._fitness_config)
            for bot in self._population.bots
        ]
        best_idx = scores.index(max(scores))
        best_bot = self._population.bots[best_idx]

        avg_fitness = sum(scores) / len(scores) if scores else 0.0

        self._logger.log_evolution(EvolutionEvent(
            event="evolution_completed",
            timestamp=now,
            generation=self._population.generation,
            population_size=len(self._population.bots),
            best_fitness=scores[best_idx],
            avg_fitness=avg_fitness,
            best_params={
                k: getattr(best_bot.params, k)
                for k in [
                    "imbalance_threshold", "flow_threshold",
                    "take_profit_usd", "stop_loss_usd",
                    "max_hold_seconds", "leader_weight",
                ]
            },
        ))

        self._population.run_evolution()


async def main() -> None:
    bot = TradingBot()

    # Graceful shutdown по Ctrl+C
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))

    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
