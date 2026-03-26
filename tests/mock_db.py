"""In-memory mock для StateDB — используется в тестах без PostgreSQL.

Поддерживает мультипопуляцию: данные хранятся per-population_id.
Default population_id=1 для обратной совместимости с существующими тестами.
"""

from __future__ import annotations

from storage.database import BotRow, OrderRow, PositionRow, TradeRow


class MockStateDB:
    """StateDB-совместимый мок — хранит данные в памяти."""

    def __init__(self) -> None:
        # Per-population storage
        self._generations: dict[int, tuple[int, int]] = {}  # pop_id -> (gen, total)
        self._bots: dict[int, list[BotRow]] = {}  # pop_id -> bots
        self._positions: dict[int, dict[int, PositionRow]] = {}  # pop_id -> {bot_id: pos}
        self._pending_orders: dict[int, dict[int, OrderRow]] = {}  # pop_id -> {bot_id: order}
        self._trades: dict[int, list[TradeRow]] = {}  # pop_id -> trades
        self._evolutions: dict[int, list[dict[str, object]]] = {}  # pop_id -> records

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def _ensure_pop(self, population_id: int) -> None:
        """Инициализирует хранилище для популяции если ещё нет."""
        if population_id not in self._generations:
            self._generations[population_id] = (0, 0)
        if population_id not in self._bots:
            self._bots[population_id] = []
        if population_id not in self._positions:
            self._positions[population_id] = {}
        if population_id not in self._pending_orders:
            self._pending_orders[population_id] = {}
        if population_id not in self._trades:
            self._trades[population_id] = []
        if population_id not in self._evolutions:
            self._evolutions[population_id] = []

    async def get_generation(
        self, *, population_id: int = 1,
    ) -> tuple[int, int]:
        self._ensure_pop(population_id)
        return self._generations[population_id]

    async def set_generation(
        self, generation: int, total_trades: int, *, population_id: int = 1,
    ) -> None:
        self._ensure_pop(population_id)
        self._generations[population_id] = (generation, total_trades)

    async def save_bots(
        self, bots: list[BotRow], *, population_id: int = 1,
    ) -> None:
        self._ensure_pop(population_id)
        self._bots[population_id] = list(bots)

    async def load_bots(
        self, *, population_id: int = 1,
    ) -> list[BotRow]:
        self._ensure_pop(population_id)
        return list(self._bots[population_id])

    async def open_position(
        self, pos: PositionRow, *, population_id: int = 1,
    ) -> None:
        self._ensure_pop(population_id)
        self._positions[population_id][pos.bot_id] = pos

    async def open_positions_batch(
        self, positions: list[PositionRow], *, population_id: int = 1,
    ) -> None:
        self._ensure_pop(population_id)
        for pos in positions:
            self._positions[population_id][pos.bot_id] = pos

    async def close_position(
        self, bot_id: int, *, population_id: int = 1,
    ) -> None:
        self._ensure_pop(population_id)
        self._positions[population_id].pop(bot_id, None)

    async def load_positions(
        self, *, population_id: int = 1,
    ) -> list[PositionRow]:
        self._ensure_pop(population_id)
        return list(self._positions[population_id].values())

    async def save_pending_orders_batch(
        self, orders: list[OrderRow], *, population_id: int = 1,
    ) -> None:
        self._ensure_pop(population_id)
        for order in orders:
            self._pending_orders[population_id][order.bot_id] = order

    async def load_pending_orders(
        self, *, population_id: int = 1,
    ) -> list[OrderRow]:
        self._ensure_pop(population_id)
        return list(self._pending_orders[population_id].values())

    async def delete_pending_orders_batch(
        self, bot_ids: list[int], *, population_id: int = 1,
    ) -> None:
        self._ensure_pop(population_id)
        for bot_id in bot_ids:
            self._pending_orders[population_id].pop(bot_id, None)

    async def close_trade(
        self, trade: TradeRow, *, population_id: int = 1,
    ) -> int:
        """Атомарно: удалить позицию + вставить trade + инкремент."""
        self._ensure_pop(population_id)
        self._positions[population_id].pop(trade.bot_id, None)
        self._trades[population_id].append(trade)
        gen, total = self._generations[population_id]
        total += 1
        self._generations[population_id] = (gen, total)
        return total

    async def close_trades_batch(
        self, trades: list[TradeRow], *, population_id: int = 1,
    ) -> None:
        """Батч-закрытие сделок."""
        self._ensure_pop(population_id)
        for trade in trades:
            self._positions[population_id].pop(trade.bot_id, None)
            self._trades[population_id].append(trade)
        gen, total = self._generations[population_id]
        total += len(trades)
        self._generations[population_id] = (gen, total)

    async def get_trades_for_generation(
        self, generation: int, *, population_id: int = 1,
    ) -> list[TradeRow]:
        self._ensure_pop(population_id)
        return [t for t in self._trades[population_id] if t.generation == generation]

    async def get_trades_for_bot(
        self, bot_id: int, generation: int, *, population_id: int = 1,
    ) -> list[TradeRow]:
        self._ensure_pop(population_id)
        return [
            t for t in self._trades[population_id]
            if t.bot_id == bot_id and t.generation == generation
        ]

    async def get_trade_counts(
        self, generation: int, *, population_id: int = 1,
    ) -> dict[int, int]:
        self._ensure_pop(population_id)
        counts: dict[int, int] = {}
        for t in self._trades[population_id]:
            if t.generation == generation:
                counts[t.bot_id] = counts.get(t.bot_id, 0) + 1
        return counts

    async def get_bot_balance(
        self, bot_id: int, initial_balance: float, generation: int,
        *, population_id: int = 1,
    ) -> float:
        self._ensure_pop(population_id)
        matching = [
            t for t in self._trades[population_id]
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
        *,
        population_id: int = 1,
    ) -> None:
        self._ensure_pop(population_id)
        self._evolutions[population_id].append({
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
        *,
        population_id: int = 1,
    ) -> None:
        """Атомарная эволюция в памяти."""
        self._ensure_pop(population_id)
        await self.insert_evolution(
            generation, best_fitness, avg_fitness, best_params,
            population_id=population_id,
        )
        self._bots[population_id] = list(bots)
        self._positions[population_id].clear()
        self._pending_orders[population_id].clear()
        self._generations[population_id] = (generation, total_trades)

    async def load_hall_of_fame(
        self, limit: int, *, population_id: int = 1,
        min_generation: int = 0,
    ) -> list[tuple[float, dict[str, object]]]:
        """Top-N лучших ботов из истории эволюции по fitness."""
        self._ensure_pop(population_id)
        entries: list[tuple[float, dict[str, object]]] = []
        for e in self._evolutions[population_id]:
            bp = e.get("best_params")
            gen = int(str(e.get("generation", 0)))
            if bp is not None and isinstance(bp, dict) and gen >= min_generation:
                fit = e["best_fitness"]
                entries.append((float(str(fit)), dict(bp)))
        entries.sort(key=lambda x: x[0], reverse=True)
        return entries[:limit]
