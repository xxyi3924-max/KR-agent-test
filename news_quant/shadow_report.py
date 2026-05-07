"""Phase 5: forward-shadow report.

Compares live LLM scores (logged during dry-run executor) against realized
forward returns (fetched from Polygon).  Outputs:
  - count of shadow events in window
  - signed-score vs realized-return correlation
  - hit rate at top-30% confidence
  - decision: shadow correlation > 0.5 (relative to backtest IC) → can advance to live

Runtime is wall-clock-bound (3 months minimum per plan).  This module is the
analyser only — it consumes a `shadow_log.parquet` written by `executor.py
--dry-run`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from news_quant.analysis.ic import compute_ic


def report(shadow_path: Path, window_days: int = 30) -> dict:
    if not shadow_path.exists():
        return {"error": f"shadow log missing: {shadow_path}"}
    df = pd.read_parquet(shadow_path)
    if df.empty:
        return {"error": "empty"}

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    cutoff = df["ts_utc"].max() - pd.Timedelta(days=window_days)
    sub = df[df["ts_utc"] >= cutoff]
    if sub.empty:
        return {"error": f"no events in last {window_days}d"}

    horizons = [c for c in sub.columns if c.startswith("fwd_ret_") and c.endswith("_bps")]
    rows = [compute_ic(sub, h) for h in horizons]
    return {"n": len(sub), "window_days": window_days, "horizons": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shadow", default="news_quant/data/shadow_log.parquet")
    p.add_argument("--window", type=int, default=30)
    args = p.parse_args()

    out = report(Path(args.shadow), window_days=args.window)
    if "error" in out:
        print(out["error"]); return
    print(f"shadow events in last {out['window_days']}d: n={out['n']}")
    for h in out["horizons"]:
        print(f"  {h.get('fwd_col','?'):20s}  IC={h.get('spearman_ic',0):+.4f}  p={h.get('spearman_p',1):.3f}  hit_top30={h.get('hit_rate_top30pct',0):.3f}")


if __name__ == "__main__":
    main()
