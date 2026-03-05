"""Тесты для monitoring/logger.py."""

from __future__ import annotations

import json
from pathlib import Path

from monitoring.logger import (
    EvolutionEvent,
    StructuredLogger,
    SystemEvent,
    TradeEvent,
)


class TestStructuredLogger:
    def test_log_trade(self, tmp_path: Path) -> None:
        logger = StructuredLogger(log_dir=str(tmp_path), stdout=False)
        event = TradeEvent(
            event="trade_opened",
            timestamp=1000.0,
            bot_id=7,
            generation=3,
            params={"imbalance_threshold": 0.71},
            signals={"imbalance": 0.74, "flow_ratio": 1.9},
            trade={"side": "LONG", "entry": 67421},
        )
        logger.log_trade(event)

        log_file = tmp_path / "trading.jsonl"
        assert log_file.exists()
        data = json.loads(log_file.read_text().strip())
        assert data["event"] == "trade_opened"
        assert data["bot_id"] == 7
        assert data["signals"]["imbalance"] == 0.74

    def test_log_evolution(self, tmp_path: Path) -> None:
        logger = StructuredLogger(log_dir=str(tmp_path), stdout=False)
        event = EvolutionEvent(
            event="evolution_completed",
            timestamp=2000.0,
            generation=5,
            population_size=20,
            best_fitness=1.5,
            avg_fitness=0.8,
            best_params={"imbalance_threshold": 0.72},
        )
        logger.log_evolution(event)

        data = json.loads((tmp_path / "trading.jsonl").read_text().strip())
        assert data["event"] == "evolution_completed"
        assert data["generation"] == 5

    def test_log_system(self, tmp_path: Path) -> None:
        logger = StructuredLogger(log_dir=str(tmp_path), stdout=False)
        event = SystemEvent(
            event="system_start",
            timestamp=500.0,
            message="Bot started",
            details={"version": "0.1.0"},
        )
        logger.log_system(event)

        data = json.loads((tmp_path / "trading.jsonl").read_text().strip())
        assert data["event"] == "system_start"

    def test_multiple_events_appended(self, tmp_path: Path) -> None:
        logger = StructuredLogger(log_dir=str(tmp_path), stdout=False)
        logger.log_raw({"event": "first", "n": 1})
        logger.log_raw({"event": "second", "n": 2})

        lines = (tmp_path / "trading.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "first"
        assert json.loads(lines[1])["event"] == "second"
