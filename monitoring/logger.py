"""Структурированное JSON логирование всех событий системы.

Каждое событие пишется в JSON формате с полным контекстом:
- параметры бота
- значения сигналов
- результат сделки
- метрики fitness

Лог должен объяснять каждое решение бота.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """Событие открытия/закрытия сделки."""

    event: str  # "trade_opened" | "trade_closed"
    timestamp: float
    bot_id: int
    generation: int
    params: dict[str, float]
    signals: dict[str, float]
    trade: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvolutionEvent:
    """Событие эволюции поколения."""

    event: str  # "evolution_completed"
    timestamp: float
    generation: int
    population_size: int
    best_fitness: float
    avg_fitness: float
    best_params: dict[str, float]


@dataclass(frozen=True, slots=True)
class SystemEvent:
    """Системное событие (старт, стоп, ошибка)."""

    event: str
    timestamp: float
    message: str
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class StructuredLogger:
    """JSON логгер — пишет в файл и/или stdout."""

    def __init__(
        self,
        log_dir: str = "logs",
        log_file: str = "trading.jsonl",
        *,
        stdout: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / log_file
        self._stdout = stdout

        # Python logger для fallback
        self._logger = logging.getLogger("trading-bot")
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def log_trade(self, event: TradeEvent) -> None:
        self._write(asdict(event))

    def log_evolution(self, event: EvolutionEvent) -> None:
        self._write(asdict(event))

    def log_system(self, event: SystemEvent) -> None:
        self._write(asdict(event))

    def log_raw(self, data: dict[str, Any]) -> None:
        """Запись произвольного события."""
        self._write(data)

    def _write(self, data: dict[str, Any]) -> None:
        line = json.dumps(data, default=str, ensure_ascii=False)

        # Файл
        with open(self._log_path, "a") as f:
            f.write(line + "\n")

        # stdout
        if self._stdout:
            self._logger.info(line)
