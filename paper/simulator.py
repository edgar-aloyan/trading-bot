"""Paper trading config — параметры виртуальной торговли.

Комиссии, slippage, размер позиции — всё конфигурируется через params.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml


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
