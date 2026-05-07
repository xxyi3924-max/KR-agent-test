"""Compute forward returns for each scored 8-K event using Polygon 5m bars.

For each event with acceptance_dt_utc T:
  - Find first 5m bar with start_time >= T (the next-trade-window open).
  - Compute returns at +1h, +2h, +4h, +1day from that bar's open.

Returns are issuer-equity returns (ticker), in bps.  Handles after-hours/weekend
filings by snapping to next regular session open.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulate_etf_monitor import fetch_bars_polygon  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
BARS_CACHE = DATA_DIR / "bars"
BARS_CACHE.mkdir(exist_ok=True)

HORIZONS_MIN = [60, 120, 240, 1440]


def _bars_for_ticker(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
) -> pd.DataFrame:
    """Wrap fetch_bars_polygon, return DataFrame with UTC-aware index."""
    bars = fetch_bars_polygon(ticker, start, end, api_key, multiplier=5, use_cache=True)
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def _fwd_return_bps(
    bars: pd.DataFrame, anchor_ts: pd.Timestamp, horizon_min: int
) -> float | None:
    if bars.empty:
        return None
    after = bars[bars["ts"] >= anchor_ts]
    if after.empty:
        return None
    entry_bar = after.iloc[0]
    entry_px = float(entry_bar["open"])
    target_ts = entry_bar["ts"] + pd.Timedelta(minutes=horizon_min)
    later = bars[bars["ts"] >= target_ts]
    if later.empty:
        # Fall back to last available bar within +25% of horizon
        tolerance = entry_bar["ts"] + pd.Timedelta(minutes=int(horizon_min * 1.25))
        within = bars[(bars["ts"] >= entry_bar["ts"]) & (bars["ts"] <= tolerance)]
        if within.empty:
            return None
        exit_px = float(within.iloc[-1]["close"])
    else:
        exit_px = float(later.iloc[0]["open"])
    if entry_px <= 0:
        return None
    return (exit_px / entry_px - 1.0) * 1e4


def attach_forward_returns(
    scored: pd.DataFrame,
    polygon_api_key: str,
    horizons_min: list[int] = HORIZONS_MIN,
) -> pd.DataFrame:
    """For each scored event, fetch bars and compute forward returns.

    Caches one bar dataframe per (ticker, start_year-end_year).  For 100
    events spread across ~50 tickers and 2 years, free-tier Polygon
    (5 calls/min, 13s sleep between chunks) means ~30+ minutes wall-clock.
    """
    if scored.empty:
        return scored

    out = scored.copy()
    out["acceptance_dt_utc"] = pd.to_datetime(out["acceptance_dt_utc"], utc=True)
    for h in horizons_min:
        out[f"fwd_ret_{h}m_bps"] = pd.NA

    tickers = sorted({t for t in out["ticker"].dropna().unique() if t})
    print(f"[fwd_returns] {len(tickers)} unique tickers, {len(out)} events")

    bars_by_ticker: dict[str, pd.DataFrame] = {}
    network_calls = 0  # only count actual Polygon hits, not cache reads
    for j, t in enumerate(tickers, 1):
        sub = out[out["ticker"] == t]
        d_min = sub["acceptance_dt_utc"].min().date()
        d_max = sub["acceptance_dt_utc"].max().date() + timedelta(days=2)
        start = d_min.isoformat()
        end = d_max.isoformat()
        # Probe cache: if a CSV exists for this ticker/range, fetch_bars_polygon
        # short-circuits without an API call. Only sleep when we'll actually call out.
        from pathlib import Path as _Path
        cache_root = _Path(__file__).resolve().parents[2] / "data_cache"
        cache_hit = (cache_root / f"{t}_5m_{start}_{end}.csv").exists()
        if not cache_hit and network_calls > 0:
            # Polygon free tier: 5 calls/min → 13s spacing. Be polite.
            time.sleep(13)
        print(f"  [{j}/{len(tickers)}] {t}  {start} → {end}  ({len(sub)} events){' [cache]' if cache_hit else ''}")
        bars = _bars_for_ticker(t, start, end, polygon_api_key)
        bars_by_ticker[t] = bars
        if not cache_hit:
            network_calls += 1

    for i, row in out.iterrows():
        t = row["ticker"]
        if not t or t not in bars_by_ticker:
            continue
        bars = bars_by_ticker[t]
        for h in horizons_min:
            r = _fwd_return_bps(bars, row["acceptance_dt_utc"], h)
            if r is not None:
                out.at[i, f"fwd_ret_{h}m_bps"] = r
    return out


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--polygon-key", default=None,
                   help="Polygon API key; defaults to POLYGON_API_KEY env")
    args = p.parse_args()

    import os
    api_key = args.polygon_key or os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        sys.exit("Polygon key required: --polygon-key or POLYGON_API_KEY")

    scored = pd.read_parquet(args.scored)
    print(f"loaded {len(scored)} scored events from {args.scored}")
    out_path = Path(args.out) if args.out else Path(args.scored).with_suffix(".fwd.parquet")
    enriched = attach_forward_returns(scored, api_key)
    enriched.to_parquet(out_path, index=False)
    print(f"\nWrote {len(enriched)} → {out_path}")

    for h in HORIZONS_MIN:
        col = f"fwd_ret_{h}m_bps"
        if col in enriched.columns:
            ser = pd.to_numeric(enriched[col], errors="coerce").dropna()
            if len(ser):
                print(f"  {col}: n={len(ser)}  mean={ser.mean():.1f}bps  "
                      f"std={ser.std():.1f}bps  median={ser.median():.1f}bps")


if __name__ == "__main__":
    main()
