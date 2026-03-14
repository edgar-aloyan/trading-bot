"""In-memory mock для StateDB — используется в тестах без PostgreSQL."""

from __future__ import annotations

from storage.database import BotRow, PositionRow, TradeRow


class MockStateDB:
    """StateDB-совместимый мок — хранит данные в памяти."""

    def __init__(self) -> None:
        self._generation = 0
        self._total_trades = 0
        self._bots: list[BotRow] = []
        self._positions: dict[int, PositionRow] = {}
        self._trades: list[TradeRow] = []
        self._evolutions: list[dict[str, object]] = []

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def get_generation(self) -> tuple[int, int]:
        return (self._generation, self._total_trades)

    async def set_generation(self, generation: int, total_trades: int) -> None:
        self._generation = generation
        self._total_trades = total_trades

    async def save_bots(self, bots: list[BotRow]) -> None:
        self._bots = list(bots)

    async def load_bots(self) -> list[BotRow]:
        return list(self._bots)

    async def open_position(self, pos: PositionRow) -> None:
        self._positions[pos.bot_id] = pos

    async def open_positions_batch(self, positions: list[PositionRow]) -> None:
        for pos in positions:
            self._positions[pos.bot_id] = pos

    async def close_position(self, bot_id: int) -> None:
        self._positions.pop(bot_id, None)

    async def load_positions(self) -> list[PositionRow]:
        return list(self._positions.values())

    async def close_trade(self, trade: TradeRow) -> int:
        """Атомарно: удалить позицию + вставить trade + инкремент."""
        self._positions.pop(trade.bot_id, None)
        self._trades.append(trade)
        self._total_trades += 1
        return self._total_trades

    async def close_trades_batch(self, trades: list[TradeRow]) -> None:
        """Батч-закрытие сделок."""
        for trade in trades:
            self._positions.pop(trade.bot_id, None)
            self._trades.append(trade)
            self._total_trades += 1

    async def get_trades_for_generation(self, generation: int) -> list[TradeRow]:
        return [t for t in self._trades if t.generation == generation]

    async def get_trades_for_bot(self, bot_id: int, generation: int) -> list[TradeRow]:
        return [
            t for t in self._trades
            if t.bot_id == bot_id and t.generation == generation
        ]

    async def get_trade_counts(self, generation: int) -> dict[int, int]:
        counts: dict[int, int] = {}
        for t in self._trades:
            if t.generation == generation:
                counts[t.bot_id] = counts.get(t.bot_id, 0) + 1
        return counts

    async def get_bot_balance(
        self, bot_id: int, initial_balance: float, generation: int,
    ) -> float:
        matching = [
            t for t in self._trades
            if t.bot_id == bot_id and t.generation == generation
        ]
        # pnl уже net (gross - fees)
        total_pnl = sum(t.pnl for t in matching)
        return initial_balance + total_pnl

    async def insert_evolution(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        best_params: dict[str, float],
    ) -> None:
        self._evolutions.append({
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "best_params": best_params,
        })

    async def run_evolution_tx(
        self,
        generation: int,
        total_trades: int,
        bots: list[BotRow],
        best_fitness: float,
        avg_fitness: float,
        best_params: dict[str, float],
    ) -> None:
        """Атомарная эволюция в памяти."""
        await self.insert_evolution(generation, best_fitness, avg_fitness, best_params)
        self._bots = list(bots)
        self._positions.clear()
        self._generation = generation
        self._total_trades = total_trades
