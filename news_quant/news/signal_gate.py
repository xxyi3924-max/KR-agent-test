"""Signal threshold + tradability gate.

Decides whether a scored event triggers a trade.

Gates (in evaluation order):
  1. ticker is in the tradable universe set
  2. no existing open position on this ticker (cash-parking constraint)
  3. daily cycle cap not exceeded
  4. composite score = direction × confidence × (mag/1000) × credibility
     above threshold in absolute value

Long/short is supported: the gate fires for both signs of `score`. The
caller decides whether to go long (score>0) or short (score<0). Earnings
exclusion happens upstream (we filter Item 2.02 in the source poller, not
here).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    fire: bool
    score: float
    side: str          # "long" | "short" | "none"
    reason: str = ""


def evaluate(
    *,
    direction: float,
    confidence: float,
    magnitude_bps: float,
    credibility: float,
    threshold: float,
    ticker: str,
    tradable_universe: set[str],
    open_positions: set[str],
    cycles_today: int,
    max_cycles_per_day: int,
) -> GateResult:
    score = direction * confidence * (magnitude_bps / 1000.0) * credibility

    if not ticker:
        return GateResult(False, score, "none", "no ticker")
    if tradable_universe and ticker not in tradable_universe:
        return GateResult(False, score, "none", f"ticker {ticker} not tradable")
    if ticker in open_positions:
        return GateResult(False, score, "none", f"already have position in {ticker}")
    if cycles_today >= max_cycles_per_day:
        return GateResult(False, score, "none", f"daily cycle cap ({cycles_today}/{max_cycles_per_day})")
    if abs(score) < threshold:
        return GateResult(False, score, "none", f"|score|={abs(score):.3f} < threshold {threshold}")

    side = "long" if score > 0 else "short"
    return GateResult(True, score, side, "ok")
