"""Per-ticker EWMA decay aggregator over scored news items.

For each (ticker, t), produces a continuous signal in [-1, +1]:
    signal_t = sum_i  decay(t - t_i) * direction_i * confidence_i * credibility_i
            -------------------------------------------------------
                       sum_i  decay(t - t_i) * credibility_i

where decay(dt) = exp(-ln(2) * dt / half_life).

Magnitude is the credibility-weighted mean of magnitude_bps_i over the same window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class ScoredEvent:
    ts_utc: pd.Timestamp
    ticker: str
    direction: float
    confidence: float
    magnitude_bps: float
    credibility: float


def aggregate(
    events: list[ScoredEvent],
    at_time: pd.Timestamp,
    half_life_min: int,
    max_lookback_min: int = 24 * 60,
) -> dict:
    """Aggregate all events at or before `at_time` (no look-ahead)."""
    if not events:
        return {"signal": 0.0, "magnitude_bps": 0.0, "n": 0}
    cutoff = at_time - pd.Timedelta(minutes=max_lookback_min)
    relevant = [e for e in events if cutoff <= e.ts_utc <= at_time]
    if not relevant:
        return {"signal": 0.0, "magnitude_bps": 0.0, "n": 0}
    half_life = max(half_life_min, 1)
    ln2 = math.log(2.0)
    num_dir = 0.0
    den = 0.0
    num_mag = 0.0
    den_mag = 0.0
    for e in relevant:
        dt_min = (at_time - e.ts_utc).total_seconds() / 60.0
        decay = math.exp(-ln2 * dt_min / half_life)
        w_dir = decay * e.confidence * e.credibility
        num_dir += w_dir * e.direction
        den += w_dir
        w_mag = decay * e.credibility
        num_mag += w_mag * e.magnitude_bps
        den_mag += w_mag
    signal = num_dir / den if den > 0 else 0.0
    mag = num_mag / den_mag if den_mag > 0 else 0.0
    return {"signal": signal, "magnitude_bps": mag, "n": len(relevant)}
