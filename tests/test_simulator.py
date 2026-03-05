"""Тесты для paper/simulator.py."""

from __future__ import annotations

import pytest

from core.execution import OrderSide
from paper.simulator import PaperExecutor, PaperTradingConfig


def _default_config() -> PaperTradingConfig:
    return PaperTradingConfig(
        initial_balance_usd=10000.0,
        position_size_usd=1000.0,
        maker_fee=0.0001,
        taker_fee=0.0006,
        slippage_factor=0.5,
    )


class TestPaperExecutor:
    @pytest.mark.asyncio
    async def test_buy_slippage(self) -> None:
        executor = PaperExecutor(_default_config())
        result = await executor.execute_order(
            side=OrderSide.BUY,
            amount_usd=1000.0,
            current_price=67000.0,
            spread=2.0,
            timestamp=1000.0,
        )
        # Slippage = 0.5 * 2.0 = 1.0 → buy price = 67001.0
        assert result.price == 67001.0
        assert result.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_sell_slippage(self) -> None:
        executor = PaperExecutor(_default_config())
        result = await executor.execute_order(
            side=OrderSide.SELL,
            amount_usd=1000.0,
            current_price=67000.0,
            spread=2.0,
            timestamp=1000.0,
        )
        # Slippage = 0.5 * 2.0 = 1.0 → sell price = 66999.0
        assert result.price == 66999.0

    @pytest.mark.asyncio
    async def test_fee_calculation(self) -> None:
        executor = PaperExecutor(_default_config())
        result = await executor.execute_order(
            side=OrderSide.BUY,
            amount_usd=1000.0,
            current_price=67000.0,
            spread=2.0,
            timestamp=1000.0,
        )
        # Fee = 1000 * 0.0006 = 0.6
        assert result.fee_usd == pytest.approx(0.6)
        assert executor.total_fees == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_trade_count(self) -> None:
        executor = PaperExecutor(_default_config())
        await executor.execute_order(OrderSide.BUY, 1000.0, 67000.0, 2.0, 1000.0)
        await executor.execute_order(OrderSide.SELL, 1000.0, 67000.0, 2.0, 1001.0)
        assert executor.trade_count == 2

    def test_apply_pnl(self) -> None:
        executor = PaperExecutor(_default_config())
        executor.apply_pnl(50.0)
        assert executor.balance == 10050.0
        executor.apply_pnl(-30.0)
        assert executor.balance == 10020.0

    def test_reset(self) -> None:
        executor = PaperExecutor(_default_config())
        executor.apply_pnl(500.0)
        executor.total_fees = 10.0
        executor.trade_count = 5
        executor.reset()
        assert executor.balance == 10000.0
        assert executor.total_fees == 0.0
        assert executor.trade_count == 0

    def test_config_from_yaml(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "params.yaml"
        p.write_text(
            """
paper_trading:
  initial_balance_usd: 10000
  position_size_usd: 1000
  maker_fee: 0.0001
  taker_fee: 0.0006
  slippage_factor: 0.5
"""
        )
        cfg = PaperTradingConfig.from_yaml(str(p))
        assert cfg.initial_balance_usd == 10000.0
        assert cfg.taker_fee == 0.0006
