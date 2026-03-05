"""Голосование популяции → итоговый сигнал.

Каждый бот голосует LONG/SHORT/HOLD.
Итоговый сигнал определяется по порогам из конфига.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from core.signals import Signal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VotingConfig:
    """Пороги голосования из params.yaml."""

    threshold_long: float  # доля LONG голосов для итогового LONG (0.65 = 65%)
    threshold_short: float  # доля SHORT голосов для итогового SHORT (0.65 = 65%)

    @staticmethod
    def from_yaml(path: str) -> VotingConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        v = raw["voting"]
        return VotingConfig(
            threshold_long=v["threshold_long"],
            threshold_short=v["threshold_short"],
        )


# ---------------------------------------------------------------------------
# Voting result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VotingResult:
    """Результат голосования."""

    signal: Signal
    long_ratio: float  # доля LONG голосов
    short_ratio: float  # доля SHORT голосов
    confidence: float  # сила сигнала [0, 1]
    total_voters: int


# ---------------------------------------------------------------------------
# Voting logic
# ---------------------------------------------------------------------------


def compute_vote(
    signals: list[tuple[int, Signal]], config: VotingConfig
) -> VotingResult:
    """Подсчитывает голоса и определяет итоговый сигнал."""
    total = len(signals)
    if total == 0:
        return VotingResult(
            signal=Signal.HOLD,
            long_ratio=0.0,
            short_ratio=0.0,
            confidence=0.0,
            total_voters=0,
        )

    long_votes = sum(1 for _, s in signals if s == Signal.LONG)
    short_votes = sum(1 for _, s in signals if s == Signal.SHORT)

    long_ratio = long_votes / total
    short_ratio = short_votes / total

    if long_ratio >= config.threshold_long:
        signal = Signal.LONG
        confidence = long_ratio
    elif short_ratio >= config.threshold_short:
        signal = Signal.SHORT
        confidence = short_ratio
    else:
        signal = Signal.HOLD
        confidence = 0.0

    return VotingResult(
        signal=signal,
        long_ratio=long_ratio,
        short_ratio=short_ratio,
        confidence=confidence,
        total_voters=total,
    )
