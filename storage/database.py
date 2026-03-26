"""PostgreSQL (TimescaleDB) — единственный stateful компонент системы.

Все остальные модули stateless. Состояние ботов, позиции, сделки,
эволюция — всё здесь. Полный audit trail.

Мультипопуляция: population_id разделяет данные между независимыми
популяциями. Default=1 для обратной совместимости.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema (fresh installs)
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS population (
    id           INT PRIMARY KEY,
    generation   INT NOT NULL DEFAULT 0,
    total_trades INT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bots (
    id             SERIAL PRIMARY KEY,
    population_id  INT NOT NULL DEFAULT 1,
    bot_id         INT NOT NULL,
    generation     INT NOT NULL,
    params         JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS positions (
    population_id  INT NOT NULL DEFAULT 1,
    bot_id         INT NOT NULL,
    side           TEXT NOT NULL,
    entry_price    DOUBLE PRECISION NOT NULL,
    entry_time     DOUBLE PRECISION NOT NULL,
    size_usd       DOUBLE PRECISION NOT NULL,
    entry_signals  JSONB,
    PRIMARY KEY (population_id, bot_id)
);

CREATE TABLE IF NOT EXISTS pending_orders (
    population_id  INT NOT NULL DEFAULT 1,
    bot_id         INT NOT NULL,
    side           TEXT NOT NULL,
    limit_price    DOUBLE PRECISION NOT NULL,
    placed_time    DOUBLE PRECISION NOT NULL,
    size_usd       DOUBLE PRECISION NOT NULL,
    entry_signals  JSONB,
    PRIMARY KEY (population_id, bot_id)
);

CREATE TABLE IF NOT EXISTS trades (
    id             SERIAL,
    population_id  INT NOT NULL DEFAULT 1,
    bot_id         INT NOT NULL,
    generation     INT NOT NULL,
    side           TEXT NOT NULL,
    entry_price    DOUBLE PRECISION NOT NULL,
    exit_price     DOUBLE PRECISION NOT NULL,
    pnl            DOUBLE PRECISION NOT NULL,
    fees           DOUBLE PRECISION NOT NULL,
    entry_time     DOUBLE PRECISION NOT NULL,
    exit_time      DOUBLE PRECISION NOT NULL,
    entry_signals  JSONB,
    exit_signals   JSONB,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evolutions (
    id             SERIAL PRIMARY KEY,
    population_id  INT NOT NULL DEFAULT 1,
    generation     INT NOT NULL,
    best_fitness   DOUBLE PRECISION NOT NULL,
    avg_fitness    DOUBLE PRECISION NOT NULL,
    best_params    JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (population_id, generation)
);

CREATE INDEX IF NOT EXISTS idx_trades_bot_gen ON trades (bot_id, generation);

SELECT create_hypertable('trades', 'created_at', if_not_exists => TRUE);
"""

# ---------------------------------------------------------------------------
# Migration v2: multi-population support for existing DBs
# ---------------------------------------------------------------------------

_MIGRATION_V2_COLUMNS = [
    "ALTER TABLE bots ADD COLUMN IF NOT EXISTS population_id INT NOT NULL DEFAULT 1",
    "ALTER TABLE positions ADD COLUMN IF NOT EXISTS population_id INT NOT NULL DEFAULT 1",
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS population_id INT NOT NULL DEFAULT 1",
    "ALTER TABLE evolutions ADD COLUMN IF NOT EXISTS population_id INT NOT NULL DEFAULT 1",
    ("CREATE INDEX IF NOT EXISTS idx_trades_pop_bot_gen "
     "ON trades (population_id, bot_id, generation)"),
    "CREATE INDEX IF NOT EXISTS idx_bots_pop ON bots (population_id)",
]

# PL/pgSQL: убирает singleton check, обновляет PK и unique constraints
_MIGRATION_V2_PLPGSQL = """\
DO $$
DECLARE
    r RECORD;
BEGIN
    -- Убрать CHECK (id = 1) на population если есть
    FOR r IN (SELECT conname FROM pg_constraint
              WHERE conrelid = 'population'::regclass AND contype = 'c') LOOP
        EXECUTE 'ALTER TABLE population DROP CONSTRAINT ' || quote_ident(r.conname);
    END LOOP;

    -- positions PK: (bot_id) → (population_id, bot_id)
    IF EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
        WHERE c.conname = 'positions_pkey' AND c.conrelid = 'positions'::regclass
        GROUP BY c.conname HAVING COUNT(*) = 1
    ) THEN
        ALTER TABLE positions DROP CONSTRAINT positions_pkey;
        ALTER TABLE positions ADD PRIMARY KEY (population_id, bot_id);
    END IF;

    -- evolutions unique: (generation) → (population_id, generation)
    IF EXISTS (SELECT 1 FROM pg_constraint
               WHERE conname = 'evolutions_generation_key'
               AND conrelid = 'evolutions'::regclass) THEN
        ALTER TABLE evolutions DROP CONSTRAINT evolutions_generation_key;
        ALTER TABLE evolutions ADD CONSTRAINT evolutions_pop_gen_unique
            UNIQUE (population_id, generation);
    END IF;
END $$;
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
class OrderRow:
    """Pending (лимитный) ордер из БД."""

    bot_id: int
    side: str
    limit_price: float
    placed_time: float
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
#
# population_id — keyword-only с default=1 для обратной совместимости.
# Существующие вызовы без population_id продолжают работать.
# ---------------------------------------------------------------------------


@runtime_checkable
class StateDBProtocol(Protocol):
    """Интерфейс для работы с состоянием — реализуют StateDB и MockStateDB."""

    async def get_generation(
        self, *, population_id: int = 1,
    ) -> tuple[int, int]: ...
    async def set_generation(
        self, generation: int, total_trades: int, *, population_id: int = 1,
    ) -> None: ...
    async def save_bots(
        self, bots: list[BotRow], *, population_id: int = 1,
    ) -> None: ...
    async def load_bots(
        self, *, population_id: int = 1,
    ) -> list[BotRow]: ...
    async def open_position(
        self, pos: PositionRow, *, population_id: int = 1,
    ) -> None: ...
    async def open_positions_batch(
        self, positions: list[PositionRow], *, population_id: int = 1,
    ) -> None: ...
    async def close_position(
        self, bot_id: int, *, population_id: int = 1,
    ) -> None: ...
    async def load_positions(
        self, *, population_id: int = 1,
    ) -> list[PositionRow]: ...
    async def save_pending_orders_batch(
        self, orders: list[OrderRow], *, population_id: int = 1,
    ) -> None: ...
    async def load_pending_orders(
        self, *, population_id: int = 1,
    ) -> list[OrderRow]: ...
    async def delete_pending_orders_batch(
        self, bot_ids: list[int], *, population_id: int = 1,
    ) -> None: ...
    async def close_trade(
        self, trade: TradeRow, *, population_id: int = 1,
    ) -> int: ...
    async def close_trades_batch(
        self, trades: list[TradeRow], *, population_id: int = 1,
    ) -> None: ...
    async def get_trades_for_bot(
        self, bot_id: int, generation: int, *, population_id: int = 1,
    ) -> list[TradeRow]: ...
    async def get_trade_counts(
        self, generation: int, *, population_id: int = 1,
    ) -> dict[int, int]: ...
    async def get_bot_balance(
        self, bot_id: int, initial_balance: float, generation: int,
        *, population_id: int = 1,
    ) -> float: ...
    async def insert_evolution(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        best_params: dict[str, float],
        *,
        population_id: int = 1,
    ) -> None: ...
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
    ) -> None: ...
    async def load_hall_of_fame(
        self, limit: int, *, population_id: int = 1,
        min_generation: int = 0,
    ) -> list[tuple[float, dict[str, object]]]: ...


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
            # Миграция v2: мультипопуляция
            for stmt in _MIGRATION_V2_COLUMNS:
                await conn.execute(stmt)
            await conn.execute(_MIGRATION_V2_PLPGSQL)
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

    async def get_generation(
        self, *, population_id: int = 1,
    ) -> tuple[int, int]:
        """Возвращает (generation, total_trades). Создаёт запись если нет."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT generation, total_trades FROM population WHERE id = $1",
                population_id,
            )
            if row is None:
                await conn.execute(
                    "INSERT INTO population (id, generation, total_trades) "
                    "VALUES ($1, 0, 0)",
                    population_id,
                )
                return (0, 0)
            return (int(row["generation"]), int(row["total_trades"]))

    async def set_generation(
        self, generation: int, total_trades: int, *, population_id: int = 1,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE population SET generation = $1, total_trades = $2 "
                "WHERE id = $3",
                generation, total_trades, population_id,
            )

    # ---- bots ----

    async def save_bots(
        self, bots: list[BotRow], *, population_id: int = 1,
    ) -> None:
        """Сохраняет всех ботов текущего поколения (после эволюции или при старте)."""
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM bots WHERE population_id = $1", population_id,
            )
            await conn.executemany(
                "INSERT INTO bots (population_id, bot_id, generation, params) "
                "VALUES ($1, $2, $3, $4::jsonb)",
                [(population_id, b.bot_id, b.generation, _to_json(b.params))
                 for b in bots],
            )

    async def load_bots(
        self, *, population_id: int = 1,
    ) -> list[BotRow]:
        """Загружает всех ботов."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bot_id, generation, params FROM bots "
                "WHERE population_id = $1 ORDER BY bot_id",
                population_id,
            )
            return [
                BotRow(
                    bot_id=int(r["bot_id"]),
                    generation=int(r["generation"]),
                    params=_from_json(r["params"]),
                )
                for r in rows
            ]

    # ---- positions ----

    async def open_position(
        self, pos: PositionRow, *, population_id: int = 1,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO positions "
                "(population_id, bot_id, side, entry_price, entry_time, "
                "size_usd, entry_signals) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb) "
                "ON CONFLICT (population_id, bot_id) DO UPDATE SET "
                "side=$3, entry_price=$4, entry_time=$5, "
                "size_usd=$6, entry_signals=$7::jsonb",
                population_id, pos.bot_id, pos.side, pos.entry_price,
                pos.entry_time, pos.size_usd,
                _to_json(pos.entry_signals),
            )

    async def open_positions_batch(
        self, positions: list[PositionRow], *, population_id: int = 1,
    ) -> None:
        """Батч-запись открытых позиций в одной транзакции."""
        async with self.pool.acquire() as conn, conn.transaction():
            for pos in positions:
                await conn.execute(
                    "INSERT INTO positions "
                    "(population_id, bot_id, side, entry_price, entry_time, "
                    "size_usd, entry_signals) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb) "
                    "ON CONFLICT (population_id, bot_id) DO UPDATE SET "
                    "side=$3, entry_price=$4, entry_time=$5, "
                    "size_usd=$6, entry_signals=$7::jsonb",
                    population_id, pos.bot_id, pos.side, pos.entry_price,
                    pos.entry_time, pos.size_usd,
                    _to_json(pos.entry_signals),
                )

    async def close_position(
        self, bot_id: int, *, population_id: int = 1,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM positions WHERE population_id = $1 AND bot_id = $2",
                population_id, bot_id,
            )

    async def load_positions(
        self, *, population_id: int = 1,
    ) -> list[PositionRow]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bot_id, side, entry_price, entry_time, size_usd, "
                "entry_signals FROM positions WHERE population_id = $1",
                population_id,
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

    # ---- pending orders ----

    async def save_pending_orders_batch(
        self, orders: list[OrderRow], *, population_id: int = 1,
    ) -> None:
        """Батч-запись pending orders в одной транзакции."""
        async with self.pool.acquire() as conn, conn.transaction():
            for order in orders:
                await conn.execute(
                    "INSERT INTO pending_orders "
                    "(population_id, bot_id, side, limit_price, placed_time, "
                    "size_usd, entry_signals) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb) "
                    "ON CONFLICT (population_id, bot_id) DO UPDATE SET "
                    "side=$3, limit_price=$4, placed_time=$5, "
                    "size_usd=$6, entry_signals=$7::jsonb",
                    population_id, order.bot_id, order.side,
                    order.limit_price, order.placed_time,
                    order.size_usd, _to_json(order.entry_signals),
                )

    async def load_pending_orders(
        self, *, population_id: int = 1,
    ) -> list[OrderRow]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bot_id, side, limit_price, placed_time, size_usd, "
                "entry_signals FROM pending_orders WHERE population_id = $1",
                population_id,
            )
            return [
                OrderRow(
                    bot_id=int(r["bot_id"]),
                    side=str(r["side"]),
                    limit_price=float(r["limit_price"]),
                    placed_time=float(r["placed_time"]),
                    size_usd=float(r["size_usd"]),
                    entry_signals=_parse_signals(r["entry_signals"]),
                )
                for r in rows
            ]

    async def delete_pending_orders_batch(
        self, bot_ids: list[int], *, population_id: int = 1,
    ) -> None:
        if not bot_ids:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM pending_orders "
                "WHERE population_id = $1 AND bot_id = ANY($2::int[])",
                population_id, bot_ids,
            )

    # ---- trades ----

    async def close_trade(
        self, trade: TradeRow, *, population_id: int = 1,
    ) -> int:
        """Атомарно: удаляет позицию + вставляет trade + инкрементирует total_trades.

        Возвращает новое значение total_trades.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM positions WHERE population_id = $1 AND bot_id = $2",
                population_id, trade.bot_id,
            )
            await conn.execute(
                "INSERT INTO trades "
                "(population_id, bot_id, generation, side, entry_price, exit_price, "
                "pnl, fees, entry_time, exit_time, entry_signals, exit_signals) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12::jsonb)",
                population_id, trade.bot_id, trade.generation, trade.side,
                trade.entry_price, trade.exit_price, trade.pnl, trade.fees,
                trade.entry_time, trade.exit_time,
                _to_json(trade.entry_signals),
                _to_json(trade.exit_signals),
            )
            row = await conn.fetchrow(
                "UPDATE population SET total_trades = total_trades + 1 "
                "WHERE id = $1 RETURNING total_trades",
                population_id,
            )
            assert row is not None
            return int(row["total_trades"])

    async def close_trades_batch(
        self, trades: list[TradeRow], *, population_id: int = 1,
    ) -> None:
        """Батч-закрытие сделок: все в одной транзакции."""
        async with self.pool.acquire() as conn, conn.transaction():
            for trade in trades:
                await conn.execute(
                    "DELETE FROM positions WHERE population_id = $1 AND bot_id = $2",
                    population_id, trade.bot_id,
                )
                await conn.execute(
                    "INSERT INTO trades "
                    "(population_id, bot_id, generation, side, entry_price, "
                    "exit_price, pnl, fees, entry_time, exit_time, "
                    "entry_signals, exit_signals) "
                    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12::jsonb)",
                    population_id, trade.bot_id, trade.generation, trade.side,
                    trade.entry_price, trade.exit_price,
                    trade.pnl, trade.fees,
                    trade.entry_time, trade.exit_time,
                    _to_json(trade.entry_signals),
                    _to_json(trade.exit_signals),
                )
            await conn.execute(
                "UPDATE population SET total_trades = total_trades + $1 "
                "WHERE id = $2",
                len(trades), population_id,
            )

    async def get_trade_counts(
        self, generation: int, *, population_id: int = 1,
    ) -> dict[int, int]:
        """Количество сделок на бота в поколении — для триггера эволюции."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bot_id, COUNT(*) as cnt "
                "FROM trades WHERE population_id = $1 AND generation = $2 "
                "GROUP BY bot_id",
                population_id, generation,
            )
            return {int(r["bot_id"]): int(r["cnt"]) for r in rows}

    async def get_trades_for_generation(
        self, generation: int, *, population_id: int = 1,
    ) -> list[TradeRow]:
        """Все сделки одного поколения — для fitness расчёта."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bot_id, generation, side, entry_price, exit_price, "
                "pnl, fees, entry_time, exit_time, entry_signals, exit_signals "
                "FROM trades WHERE population_id = $1 AND generation = $2 "
                "ORDER BY exit_time",
                population_id, generation,
            )
            return [_row_to_trade(r) for r in rows]

    async def get_trades_for_bot(
        self, bot_id: int, generation: int, *, population_id: int = 1,
    ) -> list[TradeRow]:
        """Сделки конкретного бота в поколении — для fitness."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bot_id, generation, side, entry_price, exit_price, "
                "pnl, fees, entry_time, exit_time, entry_signals, exit_signals "
                "FROM trades WHERE population_id = $1 AND bot_id = $2 "
                "AND generation = $3 ORDER BY exit_time",
                population_id, bot_id, generation,
            )
            return [_row_to_trade(r) for r in rows]

    # ---- evolutions ----

    async def insert_evolution(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        best_params: dict[str, float],
        *,
        population_id: int = 1,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO evolutions "
                "(population_id, generation, best_fitness, avg_fitness, best_params) "
                "VALUES ($1, $2, $3, $4, $5::jsonb)",
                population_id, generation, best_fitness, avg_fitness,
                _to_json(best_params),
            )

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
        """Атомарная эволюция: сохраняет новое поколение в одной транзакции.

        Внутри: insert evolution + delete bots + insert new bots +
        delete positions + update population.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO evolutions "
                "(population_id, generation, best_fitness, avg_fitness, best_params) "
                "VALUES ($1, $2, $3, $4, $5::jsonb)",
                population_id, generation, best_fitness, avg_fitness,
                _to_json(best_params),
            )
            await conn.execute(
                "DELETE FROM bots WHERE population_id = $1", population_id,
            )
            await conn.executemany(
                "INSERT INTO bots (population_id, bot_id, generation, params) "
                "VALUES ($1, $2, $3, $4::jsonb)",
                [(population_id, b.bot_id, b.generation, _to_json(b.params))
                 for b in bots],
            )
            await conn.execute(
                "DELETE FROM positions WHERE population_id = $1", population_id,
            )
            await conn.execute(
                "DELETE FROM pending_orders WHERE population_id = $1",
                population_id,
            )
            await conn.execute(
                "UPDATE population SET generation = $1, total_trades = $2 "
                "WHERE id = $3",
                generation, total_trades, population_id,
            )

    # ---- hall of fame ----

    async def load_hall_of_fame(
        self, limit: int, *, population_id: int = 1,
        min_generation: int = 0,
    ) -> list[tuple[float, dict[str, object]]]:
        """Top-N лучших ботов из истории эволюции по fitness."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT best_fitness, best_params FROM evolutions "
                "WHERE population_id = $1 AND best_params IS NOT NULL "
                "AND generation >= $3 "
                "ORDER BY best_fitness DESC LIMIT $2",
                population_id, limit, min_generation,
            )
            return [
                (float(row["best_fitness"]), _from_json(row["best_params"]))
                for row in rows
            ]

    # ---- balance (computed) ----

    async def get_bot_balance(
        self, bot_id: int, initial_balance: float, generation: int,
        *, population_id: int = 1,
    ) -> float:
        """Balance = initial + SUM(pnl) for current generation only.

        pnl в DB уже net (gross - fees), повторно вычитать fees не нужно.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COALESCE(SUM(pnl), 0) as total_pnl "
                "FROM trades WHERE population_id = $1 AND bot_id = $2 "
                "AND generation = $3",
                population_id, bot_id, generation,
            )
            if row is None:
                return initial_balance
            return initial_balance + float(row["total_pnl"])


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
