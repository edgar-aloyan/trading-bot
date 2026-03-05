"""Paper trading simulator — виртуальное исполнение ордеров.

Симулирует исполнение по рыночной цене с учётом:
- Комиссий Bybit (taker fee)
- Slippage (configurable, по умолчанию 0.5 * spread)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import yaml

from core.execution import OrderResult, OrderSide

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PaperTradingConfig:
    """Параметры paper trading из params.yaml."""

    initial_balance_usd: float
    position_size_usd: float
    maker_fee: float
    taker_fee: float
    slippage_factor: float

    @staticmethod
    def from_yaml(path: str) -> PaperTradingConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        pt = raw["paper_trading"]
        return PaperTradingConfig(
            initial_balance_usd=pt["initial_balance_usd"],
            position_size_usd=pt["position_size_usd"],
            maker_fee=pt["maker_fee"],
            taker_fee=pt["taker_fee"],
            slippage_factor=pt["slippage_factor"],
        )


# ---------------------------------------------------------------------------
# PaperExecutor
# ---------------------------------------------------------------------------


class PaperExecutor:
    """Симулятор исполнения ордеров — без реальной биржи.

    Учитывает slippage и комиссии.
    Ведёт виртуальный баланс бота.
    """

    def __init__(self, config: PaperTradingConfig) -> None:
        self._config = config
        self.balance: float = config.initial_balance_usd
        self.total_fees: float = 0.0
        self.trade_count: int = 0

    async def execute_order(
        self,
        side: OrderSide,
        amount_usd: float,
        current_price: float,
        spread: float,
        timestamp: float,
    ) -> OrderResult:
        """Симулирует исполнение ордера.

        Цена сдвигается на slippage в невыгодную сторону.
        """
        slippage = self._config.slippage_factor * spread

        # Slippage: покупка дороже, продажа дешевле
        sign = 1.0 if side == OrderSide.BUY else -1.0
        exec_price = current_price + sign * slippage

        fee_usd = amount_usd * self._config.taker_fee
        self.total_fees += fee_usd
        self.trade_count += 1

        return OrderResult(
            order_id=uuid.uuid4().hex[:12],
            side=side,
            price=exec_price,
            amount_usd=amount_usd,
            fee_usd=fee_usd,
            timestamp=timestamp,
        )

    def apply_pnl(self, pnl: float) -> None:
        """Применяет PnL закрытой сделки к балансу."""
        self.balance += pnl

    def reset(self) -> None:
        """Сброс к начальному состоянию."""
        self.balance = self._config.initial_balance_usd
        self.total_fees = 0.0
        self.trade_count = 0
