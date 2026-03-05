"""Отправка ордеров — абстракция над paper и real trading.

Execution слой не знает о сигналах и решениях.
Он получает команду "купить/продать X по цене Y" и исполняет.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

# ---------------------------------------------------------------------------
# Order types
# ---------------------------------------------------------------------------


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Результат исполнения ордера."""

    order_id: str
    side: OrderSide
    price: float  # цена исполнения (с учётом slippage для paper)
    amount_usd: float
    fee_usd: float
    timestamp: float


# ---------------------------------------------------------------------------
# Executor protocol
# ---------------------------------------------------------------------------


class OrderExecutor(Protocol):
    """Интерфейс исполнителя ордеров.

    Реализации: PaperExecutor (симуляция), LiveExecutor (реальная биржа).
    """

    async def execute_order(
        self,
        side: OrderSide,
        amount_usd: float,
        current_price: float,
        spread: float,
        timestamp: float,
    ) -> OrderResult: ...
