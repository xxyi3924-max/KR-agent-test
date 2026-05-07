"""Decay math + credibility weighting for news.aggregator."""

import math

import pandas as pd

from news_quant.news.aggregator import ScoredEvent, aggregate


def _ev(ts, direction, confidence=1.0, magnitude=100, credibility=1.0, ticker="AAPL"):
    return ScoredEvent(
        ts_utc=pd.Timestamp(ts, tz="UTC"),
        ticker=ticker,
        direction=direction,
        confidence=confidence,
        magnitude_bps=magnitude,
        credibility=credibility,
    )


def test_empty_returns_zero():
    out = aggregate([], pd.Timestamp("2025-01-01", tz="UTC"), half_life_min=60)
    assert out["signal"] == 0.0
    assert out["n"] == 0


def test_no_lookahead():
    """Events strictly after at_time must NOT contribute."""
    at = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    events = [
        _ev("2025-01-01 09:30", direction=+1.0),
        _ev("2025-01-01 10:30", direction=-1.0),  # FUTURE — must be ignored
    ]
    out = aggregate(events, at, half_life_min=60)
    assert out["n"] == 1
    assert out["signal"] > 0  # only the past +1 event remains


def test_decay_halves_at_one_half_life():
    """At one half-life past, an event's weight is exactly half the fresh weight."""
    at = pd.Timestamp("2025-01-01 11:00", tz="UTC")
    fresh = _ev("2025-01-01 11:00", direction=+1.0, magnitude=200)
    aged = _ev("2025-01-01 10:00", direction=-1.0, magnitude=200)  # 60min old at half_life=60
    out = aggregate([fresh, aged], at, half_life_min=60)
    # Fresh weight = 1.0, aged weight = 0.5; signal = (1 - 0.5) / (1 + 0.5) = 1/3
    assert math.isclose(out["signal"], 1 / 3.0, abs_tol=1e-9)


def test_credibility_dominates_low_quality_source():
    at = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    high = _ev("2025-01-01 10:00", direction=+1.0, credibility=1.0)
    low = _ev("2025-01-01 10:00", direction=-1.0, credibility=0.3)
    out = aggregate([high, low], at, half_life_min=60)
    # weighted avg: (1*1 + (-1)*0.3) / (1+0.3) = 0.7/1.3 ≈ 0.538
    assert out["signal"] > 0.5
