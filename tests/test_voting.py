"""Тесты для ensemble/voting.py."""

from __future__ import annotations

from core.signals import Signal
from ensemble.voting import VotingConfig, compute_vote


def _default_config() -> VotingConfig:
    return VotingConfig(threshold_long=0.65, threshold_short=0.65)


class TestComputeVote:
    def test_empty_signals(self) -> None:
        result = compute_vote([], _default_config())
        assert result.signal == Signal.HOLD
        assert result.total_voters == 0

    def test_strong_long_consensus(self) -> None:
        signals = [(i, Signal.LONG) for i in range(8)]
        signals += [(i + 8, Signal.HOLD) for i in range(2)]
        result = compute_vote(signals, _default_config())
        assert result.signal == Signal.LONG
        assert result.long_ratio == 0.8
        assert result.confidence == 0.8

    def test_strong_short_consensus(self) -> None:
        signals = [(i, Signal.SHORT) for i in range(8)]
        signals += [(i + 8, Signal.HOLD) for i in range(2)]
        result = compute_vote(signals, _default_config())
        assert result.signal == Signal.SHORT

    def test_no_consensus_hold(self) -> None:
        """50/50 — ни один порог не достигнут → HOLD."""
        signals = [(i, Signal.LONG) for i in range(5)]
        signals += [(i + 5, Signal.SHORT) for i in range(5)]
        result = compute_vote(signals, _default_config())
        assert result.signal == Signal.HOLD

    def test_all_hold(self) -> None:
        signals = [(i, Signal.HOLD) for i in range(10)]
        result = compute_vote(signals, _default_config())
        assert result.signal == Signal.HOLD
        assert result.long_ratio == 0.0
        assert result.short_ratio == 0.0

    def test_exact_threshold_long(self) -> None:
        """Ровно на пороге — должен быть LONG."""
        config = VotingConfig(threshold_long=0.60, threshold_short=0.60)
        signals = [(i, Signal.LONG) for i in range(6)]
        signals += [(i + 6, Signal.HOLD) for i in range(4)]
        result = compute_vote(signals, config)
        assert result.signal == Signal.LONG
