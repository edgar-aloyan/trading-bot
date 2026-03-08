"""PostgreSQL (TimescaleDB) — единственный stateful компонент системы.

Все остальные модули stateless. Состояние ботов, позиции, сделки,
эволюция — всё здесь. Полный audit trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Единственная строка — текущее состояние популяции
CREATE TABLE IF NOT EXISTS population (
    id           INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    generation   INT NOT NULL DEFAULT 0,
    total_trades INT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bots (
    id          SERIAL PRIMARY KEY,
    bot_id      INT NOT NULL,
    generation  INT NOT NULL,
    params      JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS positions (
    bot_id       INT PRIMARY KEY,
    side         TEXT NOT NULL,
    entry_price  DOUBLE PRECISION NOT NULL,
    entry_time   DOUBLE PRECISION NOT NULL,
    size_usd     DOUBLE PRECISION NOT NULL,
    entry_signals JSONB
);

CREATE TABLE IF NOT EXISTS trades (
    id            SERIAL,
    bot_id        INT NOT NULL,
    generation    INT NOT NULL,
    side          TEXT NOT NULL,
    entry_price   DOUBLE PRECISION NOT NULL,
    exit_price    DOUBLE PRECISION NOT NULL,
    pnl           DOUBLE PRECISION NOT NULL,
    fees          DOUBLE PRECISION NOT NULL,
    entry_time    DOUBLE PRECISION NOT NULL,
    exit_time     DOUBLE PRECISION NOT NULL,
    entry_signals JSONB,
    exit_signals  JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evolutions (
    id           SERIAL PRIMARY KEY,
    generation   INT NOT NULL UNIQUE,
    best_fitness DOUBLE PRECISION NOT NULL,
    avg_fitness  DOUBLE PRECISION NOT NULL,
    best_params  JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Индекс для fitness-запросов: trades по (bot_id, generation)
CREATE INDEX IF NOT EXISTS idx_trades_bot_gen ON trades (bot_id, generation);

-- TimescaleDB: партиционируем trades по времени создания записи
SELECT create_hypertable('trades', 'created_at', if_not_exists => TRUE);
"""


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BotRow:
    """Бот из БД."""

    bot_id: int
    generation: int
    params: dict[str, object]


@dataclass(frozen=True, slots=True)
class PositionRow:
    """Открытая позиция из БД."""

    bot_id: int
    side: str
    entry_price: float
    entry_time: float
    size_usd: float
    entry_signals: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class TradeRow:
    """Закрытая сделка из БД."""

    bot_id: int
    generation: int
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    fees: float
    entry_time: float
    exit_time: float
    entry_signals: dict[str, object] | None = None
    exit_signals: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Database Protocol — для тестов с MockStateDB
# ---------------------------------------------------------------------------


@runtime_checkable
class StateDBProtocol(Protocol):
    """Интерфейс для работы с состоянием — реализуют StateDB и MockStateDB."""

    async def get_generation(self) -> tuple[int, int]: ...
    async def set_generation(self, generation: int, total_trades: int) -> None: ...
    async def save_bots(self, bots: list[BotRow]) -> None: ...
    async def load_bots(self) -> list[BotRow]: ...
    async def open_position(self, pos: PositionRow) -> None: ...
    async def open_positions_batch(self, positions: list[PositionRow]) -> None: ...
    async def close_position(self, bot_id: int) -> None: ...
    async def load_positions(self) -> list[PositionRow]: ...
    async def close_trade(self, trade: TradeRow) -> int: ...
    async def close_trades_batch(self, trades: list[TradeRow]) -> None: ...
    async def get_trades_for_bot(
        self, bot_id: int, generation: int,
    ) -> list[TradeRow]: ...
    async def get_trade_counts(
        self, generation: int,
    ) -> dict[int, int]: ...
    async def get_bot_balance(
        self, bot_id: int, initial_balance: float, generation: int,
    ) -> float: ...
    async def insert_evolution(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        best_params: dict[str, float],
    ) -> None: ...
    async def run_evolution_tx(
        self,
        generation: int,
        total_trades: int,
        bots: list[BotRow],
        best_fitness: float,
        avg_fitness: float,
        best_params: dict[str, float],
    ) -> None: ...


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class StateDB:
    """Async PostgreSQL клиент — единственный источник состояния."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        # asyncpg не импортирован на верхнем уровне — чтобы тесты работали без него
        self._pool: Any = None

    async def connect(self, *, max_retries: int = 10, retry_delay: float = 2.0) -> None:
        """Подключение с retry и создание схемы."""
        import asyncio

        import asyncpg as apg

        for attempt in range(1, max_retries + 1):
            try:
                self._pool = await apg.create_pool(self._dsn, min_size=2, max_size=10)
                break
            except (OSError, apg.PostgresError) as exc:
                if attempt == max_retries:
                    logger.error("Failed to connect after %d attempts", max_retries)
                    raise
                logger.warning(
                    "DB connect attempt %d/%d failed: %s, retrying in %.0fs",
                    attempt, max_retries, exc, retry_delay,
                )
                await asyncio.sleep(retry_delay)

        async with self._pool.acquire() as conn:
            for statement in _SCHEMA.split(";"):
                stmt = statement.strip()
                if stmt:
                    await conn.execute(stmt + ";")
        logger.info("Database connected and schema ready")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> Any:  # noqa: ANN401 — asyncpg.Pool не имеет type stubs
        assert self._pool is not None, "Call connect() first"
        return self._pool

    # ---- population ----

    async def get_generation(self) -> tuple[int, int]:
        """Возвращает (generation, total_trades). Создаёт запись если нет."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT generation, total_trades FROM population LIMIT 1")
            if row is None:
                await conn.execute(
                    "INSERT INTO population (generation, total_trades) VALUES (0, 0)"
                )
                return (0, 0)
            return (int(row["generation"]), int(row["total_trades"]))

    async def set_generation(self, generation: int, total_trades: int) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE population SET generation = $1, total_trades = $2",
                generation, total_trades,
            )


    # ---- bots ----

    async def save_bots(self, bots: list[BotRow]) -> None:
        """Сохраняет всех ботов текущего поколения (после эволюции или при старте)."""
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM bots")
            await conn.executemany(
                "INSERT INTO bots (bot_id, generation, params) VALUES ($1, $2, $3::jsonb)",
                [(b.bot_id, b.generation, _to_json(b.params)) for b in bots],
            )

    async def load_bots(self) -> list[BotRow]:
        """Загружает всех ботов."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT bot_id, generation, params FROM bots ORDER BY bot_id")
            return [
                BotRow(
                    bot_id=int(r["bot_id"]),
                    generation=int(r["generation"]),
                    params=_from_json(r["params"]),
                )
                for r in rows
            ]

    # ---- positions ----

    async def open_position(self, pos: PositionRow) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO positions "
                "(bot_id, side, entry_price, entry_time, size_usd, entry_signals) "
                "VALUES ($1, $2, $3, $4, $5, $6::jsonb) "
                "ON CONFLICT (bot_id) DO UPDATE SET "
                "side=$2, entry_price=$3, entry_time=$4, size_usd=$5, entry_signals=$6::jsonb",
                pos.bot_id, pos.side, pos.entry_price, pos.entry_time, pos.size_usd,
                _to_json(pos.entry_signals),
            )

    async def open_positions_batch(self, positions: list[PositionRow]) -> None:
        """Батч-запись открытых позиций в одной транзакции."""
        async with self.pool.acquire() as conn, conn.transaction():
            for pos in positions:
                await conn.execute(
                    "INSERT INTO positions "
                    "(bot_id, side, entry_price, entry_time, size_usd, "
                    "entry_signals) VALUES ($1, $2, $3, $4, $5, $6::jsonb) "
                    "ON CONFLICT (bot_id) DO UPDATE SET "
                    "side=$2, entry_price=$3, entry_time=$4, "
                    "size_usd=$5, entry_signals=$6::jsonb",
                    pos.bot_id, pos.side, pos.entry_price,
                    pos.entry_time, pos.size_usd,
                    _to_json(pos.entry_signals),
                )

    async def close_position(self, bot_id: int) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM positions WHERE bot_id = $1", bot_id)

    async def load_positions(self) -> list[PositionRow]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bot_id, side, entry_price, entry_time, size_usd, "
                "entry_signals FROM positions"
            )
            return [
                PositionRow(
                    bot_id=int(r["bot_id"]),
                    side=str(r["side"]),
                    entry_price=float(r["entry_price"]),
                    entry_time=float(r["entry_time"]),
                    size_usd=float(r["size_usd"]),
                    entry_signals=_parse_signals(r["entry_signals"]),
                )
                for r in rows
            ]

    # ---- trades ----

    async def close_trade(self, trade: TradeRow) -> int:
        """Атомарно: удаляет позицию + вставляет trade + инкрементирует total_trades.

        Возвращает новое значение total_trades.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM positions WHERE bot_id = $1", trade.bot_id,
            )
            await conn.execute(
                "INSERT INTO trades "
                "(bot_id, generation, side, entry_price, exit_price, pnl, fees, "
                "entry_time, exit_time, entry_signals, exit_signals) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::jsonb)",
                trade.bot_id, trade.generation, trade.side,
                trade.entry_price, trade.exit_price, trade.pnl, trade.fees,
                trade.entry_time, trade.exit_time,
                _to_json(trade.entry_signals),
                _to_json(trade.exit_signals),
            )
            row = await conn.fetchrow(
                "UPDATE population SET total_trades = total_trades + 1 "
                "RETURNING total_trades",
            )
            assert row is not None
            return int(row["total_trades"])

    async def close_trades_batch(self, trades: list[TradeRow]) -> None:
        """Батч-закрытие сделок: все в одной транзакции."""
        async with self.pool.acquire() as conn, conn.transaction():
            for trade in trades:
                await conn.execute(
                    "DELETE FROM positions WHERE bot_id = $1",
                    trade.bot_id,
                )
                await conn.execute(
                    "INSERT INTO trades "
                    "(bot_id, generation, side, entry_price, exit_price, "
                    "pnl, fees, entry_time, exit_time, "
                    "entry_signals, exit_signals) "
                    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11::jsonb)",
                    trade.bot_id, trade.generation, trade.side,
                    trade.entry_price, trade.exit_price,
                    trade.pnl, trade.fees,
                    trade.entry_time, trade.exit_time,
                    _to_json(trade.entry_signals),
                    _to_json(trade.exit_signals),
                )
            await conn.execute(
                "UPDATE population SET total_trades = total_trades + $1",
                len(trades),
            )

    async def get_trade_counts(self, generation: int) -> dict[int, int]:
        """Количество сделок на бота в поколении — для триггера эволюции."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bot_id, COUNT(*) as cnt "
                "FROM trades WHERE generation = $1 GROUP BY bot_id",
                generation,
            )
            return {int(r["bot_id"]): int(r["cnt"]) for r in rows}

    async def get_trades_for_generation(self, generation: int) -> list[TradeRow]:
        """Все сделки одного поколения — для fitness расчёта."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bot_id, generation, side, entry_price, exit_price, "
                "pnl, fees, entry_time, exit_time, entry_signals, exit_signals "
                "FROM trades WHERE generation = $1 ORDER BY exit_time",
                generation,
            )
            return [_row_to_trade(r) for r in rows]

    async def get_trades_for_bot(self, bot_id: int, generation: int) -> list[TradeRow]:
        """Сделки конкретного бота в поколении — для fitness."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bot_id, generation, side, entry_price, exit_price, "
                "pnl, fees, entry_time, exit_time, entry_signals, exit_signals "
                "FROM trades WHERE bot_id = $1 AND generation = $2 ORDER BY exit_time",
                bot_id, generation,
            )
            return [_row_to_trade(r) for r in rows]

    # ---- evolutions ----

    async def insert_evolution(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        best_params: dict[str, float],
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO evolutions (generation, best_fitness, avg_fitness, best_params) "
                "VALUES ($1, $2, $3, $4::jsonb)",
                generation, best_fitness, avg_fitness, _to_json(best_params),
            )

    async def run_evolution_tx(
        self,
        generation: int,
        total_trades: int,
        bots: list[BotRow],
        best_fitness: float,
        avg_fitness: float,
        best_params: dict[str, float],
    ) -> None:
        """Атомарная эволюция: сохраняет новое поколение в одной транзакции.

        Внутри: insert evolution + delete bots + insert new bots +
        delete positions + update population.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO evolutions (generation, best_fitness, avg_fitness, best_params) "
                "VALUES ($1, $2, $3, $4::jsonb)",
                generation, best_fitness, avg_fitness, _to_json(best_params),
            )
            await conn.execute("DELETE FROM bots")
            await conn.executemany(
                "INSERT INTO bots (bot_id, generation, params) "
                "VALUES ($1, $2, $3::jsonb)",
                [(b.bot_id, b.generation, _to_json(b.params)) for b in bots],
            )
            await conn.execute("DELETE FROM positions")
            await conn.execute(
                "UPDATE population SET generation = $1, total_trades = $2",
                generation, total_trades,
            )

    # ---- balance (computed) ----

    async def get_bot_balance(
        self, bot_id: int, initial_balance: float, generation: int,
    ) -> float:
        """Balance = initial + SUM(pnl) - SUM(fees) for current generation only."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COALESCE(SUM(pnl), 0) as total_pnl, "
                "COALESCE(SUM(fees), 0) as total_fees "
                "FROM trades WHERE bot_id = $1 AND generation = $2",
                bot_id, generation,
            )
            if row is None:
                return initial_balance
            return initial_balance + float(row["total_pnl"]) - float(row["total_fees"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_json(data: dict[str, float] | dict[str, object] | None) -> str | None:
    if data is None:
        return None
    import json
    return json.dumps(data, ensure_ascii=False)


def _from_json(data: str | dict[str, object]) -> dict[str, object]:
    import json
    if isinstance(data, str):
        return json.loads(data)  # type: ignore[no-any-return]
    # asyncpg автоматически парсит JSONB в dict
    return dict(data)


def _parse_signals(raw: object) -> dict[str, object] | None:
    """Парсит JSONB signals — asyncpg может вернуть dict или str."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return _from_json(raw)
    if isinstance(raw, dict):
        return raw
    return _from_json(str(raw))


def _row_to_trade(r: Any) -> TradeRow:  # noqa: ANN401 — asyncpg.Record
    return TradeRow(
        bot_id=int(r["bot_id"]),
        generation=int(r["generation"]),
        side=str(r["side"]),
        entry_price=float(r["entry_price"]),
        exit_price=float(r["exit_price"]),
        pnl=float(r["pnl"]),
        fees=float(r["fees"]),
        entry_time=float(r["entry_time"]),
        exit_time=float(r["exit_time"]),
        entry_signals=_parse_signals(r["entry_signals"]),
        exit_signals=_parse_signals(r["exit_signals"]),
    )
