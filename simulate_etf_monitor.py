"""
ETFMonitor Backtest — yfinance data or Polygon.io cache.

  python3 simulate_etf_monitor.py              # last 5 trading days, 1-min bars (yfinance)
  python3 simulate_etf_monitor.py --5m         # Jan 7–Apr 2, 5-min bars (yfinance 60d)
  python3 simulate_etf_monitor.py --5m-month   # last 4 weeks, 5-min bars (yfinance 30d)
  python3 simulate_etf_monitor.py --jan-mar    # Jan–Mar 2026, 1-hour bars (yfinance)
  python3 simulate_etf_monitor.py --6m         # 6-month 5-min run from local cache (no API)
  python3 simulate_etf_monitor.py --pg5m --pg-key KEY  # Polygon live-fetch + cache

Uses signal functions imported directly from etf_monitor.py (production code).
"""

import sys
import time
import json
import csv
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

# ── Bar cache (CSV) ───────────────────────────────────────────────────────────
# Cached files live in data_cache/ next to this script.
# Filename: {ticker}_{interval}_{start}_{end}.csv
# --no-cache forces a fresh fetch and overwrites.

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")


def _cache_path(ticker: str, interval: str, start: str, end: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fname = f"{ticker}_{interval}_{start}_{end}.csv".replace("/", "-")
    return os.path.join(_CACHE_DIR, fname)


def _save_bars(bars: List[Tuple], ticker: str, interval: str, start: str, end: str) -> None:
    path = _cache_path(ticker, interval, start, end)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close"])
        for ts, o, h, l, c in bars:
            w.writerow([ts.isoformat(), o, h, l, c])
    print(f"  Cached {len(bars)} bars → {os.path.basename(path)}")


def _load_bars(ticker: str, interval: str, start: str, end: str) -> Optional[List[Tuple]]:
    path = _cache_path(ticker, interval, start, end)
    if not os.path.exists(path):
        return None
    bars = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            bars.append((
                datetime.fromisoformat(row["ts"]),
                float(row["open"]), float(row["high"]),
                float(row["low"]),  float(row["close"]),
            ))
    print(f"  Loaded {len(bars)} bars from cache ({os.path.basename(path)})")
    return bars if bars else None

def resample_bars(bars: List[Tuple], n: int) -> List[Tuple]:
    """Resample bars by grouping every n bars into one (OHLC merge)."""
    out = []
    for i in range(0, len(bars) - n + 1, n):
        chunk = bars[i:i + n]
        ts  = chunk[0][0]
        o   = chunk[0][1]
        h   = max(b[2] for b in chunk)
        l   = min(b[3] for b in chunk)
        c   = chunk[-1][4]
        out.append((ts, o, h, l, c))
    return out

_ET = ZoneInfo("America/New_York")   # used for DST-aware session timing

import numpy as np

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not installed — run: pip install yfinance")

sys.path.insert(0, ".")
try:
    from etf_monitor import _compute_composite, _bollinger
except ImportError:
    # etf_monitor refactored these out; only the simulate-CLI mode needs them.
    # Importers that just need fetch_bars_polygon (e.g. news_quant) keep working.
    _compute_composite = None
    _bollinger = None

# ── Shared config ─────────────────────────────────────────────────────────────

STARTING_CASH       = 249_754.0
FREE_CASH_THRESHOLD = 10_000.0
MIN_INVEST          = 100.0
COMMISSION_PCT      = 0.001      # 0.1% per side = 20 bps round-trip (KR Broker)
ENTRY_THRESHOLD     = 0.425   # 98th pctl on 5m bars — optimized via optimize_threshold.py
LATE_ENTRY_THRESHOLD = 0.50   # ~99.7th pctl — very high conviction required near close
LATE_ENTRY_MINUTES  = 120     # T-120 = 14:00 ET
EXIT_THRESHOLD      = -0.20
MIN_PROFIT_BPS      = 25
MAX_VOL_PCT         = 30.0
STOP_LOSS_PCT       = 2.0
MAX_STOP_LOSSES_DAY = 2
ETF_PRIMARY         = "QQQ"
ETF_SECONDARY       = "SPY"
MARKET_CLOSE_UTC    = (21, 0)   # 16:00 ET = 21:00 UTC
MARKET_OPEN_UTC     = (14, 30)  # 09:30 ET = 14:30 UTC

# ── Resolution-specific config ────────────────────────────────────────────────
# Session-phase params (from industry best practices):
#   open_skip_minutes     — no entries in first N min of session (gap/price-discovery noise)
#   no_entry_minutes      — no NEW entries within N min of close (no profit runway)
#   eod_close_minutes     — MOC-style force-close at N min before close
#   eod_tighten_stop_min  — within N min of close, switch to tighter stop
#   eod_tighten_stop_pct  — tighter stop % near close (less time to recover)
#   breakeven_bps         — if profit > N bps near close, floor stop at entry

MODES = {
    "1m": {
        "label":                  "Last 5 trading days — 1-minute bars",
        "interval":               "1m",
        "period":                 "5d",
        "start":                  None,
        "end":                    None,
        "bars_per_year":          252 * 390,
        "squeeze_bw_pct":         0.15,
        "min_hold_bars":          10,
        "eod_close_minutes":      15,
        "no_entry_minutes":       90,
        "open_skip_minutes":      30,
        "eod_tighten_stop_min":   60,
        "eod_tighten_stop_pct":   1.0,
        "breakeven_bps":          10,
        "stop_loss_cooldown":     60,
        "bar_unit":               "min",
        "warmup_bars":            60,
    },
    "5m": {
        "label":                  "Jan 7 – Apr 2 2026 — 5-minute bars",
        "interval":               "5m",
        "period":                 "60d",
        "start":                  None,
        "end":                    None,
        "bars_per_year":          252 * 78,
        "squeeze_bw_pct":         0.40,
        "min_hold_bars":          2,           # 10 min = 2 × 5-min bars
        "eod_close_minutes":      15,
        "no_entry_minutes":       120,
        "open_skip_minutes":      30,
        "eod_tighten_stop_min":   60,
        "eod_tighten_stop_pct":   1.0,
        "breakeven_bps":          10,
        "stop_loss_cooldown":     60,
        "bar_unit":               "×5m",
        "warmup_bars":            60,
    },
    "1m-month": {
        "label":                  "Past month (~4 weeks) — 1-minute bars",
        "interval":               "1m",
        "period":                 None,      # uses chunked fetching
        "start":                  None,
        "end":                    None,
        "bars_per_year":          252 * 390,
        "squeeze_bw_pct":         0.15,
        "min_hold_bars":          10,
        "eod_close_minutes":      15,
        "no_entry_minutes":       90,
        "open_skip_minutes":      30,
        "eod_tighten_stop_min":   60,
        "eod_tighten_stop_pct":   1.0,
        "breakeven_bps":          10,
        "stop_loss_cooldown":     60,
        "bar_unit":               "min",
        "warmup_bars":            60,
    },
    "pg-5m": {
        # Date range set dynamically from --pg-start / --pg-end args
        "label":                  "Polygon.io 5-min bars",
        "interval":               "5m",
        "period":                 None,
        "start":                  None,
        "end":                    None,
        "bars_per_year":          252 * 78,
        "squeeze_bw_pct":         0.40,
        "min_hold_bars":          2,
        "eod_close_minutes":      15,
        "no_entry_minutes":       120,
        "open_skip_minutes":      30,
        "eod_tighten_stop_min":   60,
        "eod_tighten_stop_pct":   1.0,
        "breakeven_bps":          10,
        "stop_loss_cooldown":     60,
        "bar_unit":               "×5m",
        "warmup_bars":            60,
    },
    "pg-10m": {
        # Resampled from cached 5m data (2 bars → 1 bar)
        "label":                  "10-min bars (resampled from 5m cache)",
        "interval":               "10m",
        "period":                 None,
        "start":                  None,
        "end":                    None,
        "bars_per_year":          252 * 39,
        "squeeze_bw_pct":         0.60,
        "min_hold_bars":          1,
        "eod_close_minutes":      15,
        "no_entry_minutes":       120,
        "open_skip_minutes":      30,
        "eod_tighten_stop_min":   60,
        "eod_tighten_stop_pct":   1.0,
        "breakeven_bps":          10,
        "stop_loss_cooldown":     60,
        "bar_unit":               "×10m",
        "warmup_bars":            40,
    },
    "pg-1m": {
        # Date range set dynamically from --pg-start / --pg-end args
        "label":                  "Polygon.io 1-min bars",
        "interval":               "1m",
        "period":                 None,
        "start":                  None,
        "end":                    None,
        "bars_per_year":          252 * 390,
        "squeeze_bw_pct":         0.15,
        "min_hold_bars":          10,
        "eod_close_minutes":      15,
        "no_entry_minutes":       90,
        "open_skip_minutes":      30,
        "eod_tighten_stop_min":   60,
        "eod_tighten_stop_pct":   1.0,
        "breakeven_bps":          10,
        "stop_loss_cooldown":     60,
        "bar_unit":               "min",
        "warmup_bars":            60,
    },
    "5m-month": {
        "label":                  "Past month (~4 weeks) — 5-minute bars",
        "interval":               "5m",
        "period":                 "30d",
        "start":                  None,
        "end":                    None,
        "bars_per_year":          252 * 78,
        "squeeze_bw_pct":         0.40,
        "min_hold_bars":          2,
        "eod_close_minutes":      15,
        "no_entry_minutes":       120,
        "open_skip_minutes":      30,
        "eod_tighten_stop_min":   60,
        "eod_tighten_stop_pct":   1.0,
        "breakeven_bps":          10,
        "stop_loss_cooldown":     60,
        "bar_unit":               "×5m",
        "warmup_bars":            60,
    },
    "1h": {
        "label":                  "Jan–Mar 2026 — 1-hour bars",
        "interval":               "1h",
        "period":                 None,
        "start":                  "2026-01-01",
        "end":                    "2026-04-01",
        "bars_per_year":          252 * 7,
        "squeeze_bw_pct":         1.0,
        "min_hold_bars":          1,
        "eod_close_minutes":      60,
        "no_entry_minutes":       120,
        "open_skip_minutes":      60,
        "eod_tighten_stop_min":   120,
        "eod_tighten_stop_pct":   1.0,
        "breakeven_bps":          10,
        "stop_loss_cooldown":     240,
        "bar_unit":               "hr",
        "warmup_bars":            21,
    },
}

# ── Data helpers ──────────────────────────────────────────────────────────────

def _to_bars(df) -> List[Tuple[datetime, float, float, float, float]]:
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    bars = []
    for ts, row in df.iterrows():
        ts_utc = (
            ts.to_pydatetime().astimezone(timezone.utc)
            if hasattr(ts, "tzinfo") and ts.tzinfo
            else ts.to_pydatetime().replace(tzinfo=timezone.utc)
        )
        bars.append((ts_utc, float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])))
    return bars


def fetch_bars_chunked(ticker: str, weeks: int = 4) -> List[Tuple]:
    """
    Fetch up to `weeks` weeks of 1-minute bars by making one request per week.
    Yahoo Finance allows 1m data up to 30 days back, max 7 calendar days per request.
    """
    from datetime import date
    today = datetime.now(timezone.utc).date()
    all_bars: List[Tuple] = []
    seen_ts: set = set()
    print(f"  Fetching {ticker} 1m in {weeks} weekly chunks...", end=" ", flush=True)
    for w in range(weeks, 0, -1):
        start = (today - timedelta(weeks=w)).isoformat()
        end   = (today - timedelta(weeks=w-1)).isoformat()
        df = yf.download(ticker, start=start, end=end, interval="1m",
                         progress=False, auto_adjust=True)
        if df.empty:
            continue
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)
        for bar in _to_bars(df):
            if bar[0] not in seen_ts:
                seen_ts.add(bar[0])
                all_bars.append(bar)
    all_bars.sort(key=lambda b: b[0])
    if all_bars:
        print(f"{len(all_bars)} bars  {all_bars[0][0].strftime('%Y-%m-%d')} → {all_bars[-1][0].strftime('%Y-%m-%d')}")
    else:
        print("NO DATA")
    return all_bars


def fetch_bars(ticker: str, mode: dict) -> List[Tuple]:
    print(f"  Fetching {ticker} ({mode['interval']})...", end=" ", flush=True)
    if mode["period"]:
        df = yf.download(ticker, period=mode["period"], interval=mode["interval"],
                         progress=False, auto_adjust=True)
    else:
        df = yf.download(ticker, start=mode["start"], end=mode["end"],
                         interval=mode["interval"], progress=False, auto_adjust=True)
    if df.empty:
        print("NO DATA")
        return []
    bars = _to_bars(df)
    print(f"{len(bars)} bars  {bars[0][0].strftime('%Y-%m-%d')} → {bars[-1][0].strftime('%Y-%m-%d')}")
    return bars

# ── Polygon.io fetcher (1-min and 5-min) ─────────────────────────────────────
# Free tier: 2 years history, 5 calls/min, unlimited daily calls.
# Sign up: https://polygon.io/dashboard/signup

def fetch_bars_polygon(ticker: str, start: str, end: str, api_key: str,
                       multiplier: int = 1, use_cache: bool = True) -> List[Tuple]:
    """
    Fetch bars from Polygon.io Aggregates API.
    multiplier=1 → 1-min bars; multiplier=5 → 5-min bars.
    start/end: 'YYYY-MM-DD'
    Chunks automatically; free tier 5 calls/min → 13s sleep between chunks.
    Saves to data_cache/ and reloads on subsequent calls unless use_cache=False.
    """
    interval = f"{multiplier}m"
    if use_cache:
        cached = _load_bars(ticker, interval, start, end)
        if cached is not None:
            return cached

    from datetime import date as date_type

    base    = "https://api.polygon.io/v2/aggs/ticker"
    all_bars: List[Tuple] = []
    seen: set = set()

    # For 5m bars each chunk holds ~1,600 bars/month → safe under 50k limit.
    chunk_days = 30
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d   = datetime.strptime(end,   "%Y-%m-%d").date()
    chunk_start = start_d
    chunks: List[Tuple[date_type, date_type]] = []
    while chunk_start < end_d:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end_d)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)

    label = f"{multiplier}m"
    print(f"  Fetching {ticker} {label} via Polygon ({start} → {end}, {len(chunks)} chunk(s))...",
          flush=True)

    for i, (cs, ce) in enumerate(chunks):
        # extended_hours=false → regular session only (09:30–16:00 ET)
        url = (f"{base}/{ticker}/range/{multiplier}/minute/{cs}/{ce}"
               f"?adjusted=true&sort=asc&limit=50000"
               f"&extended_hours=false&apiKey={api_key}")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"    chunk {cs}→{ce}: HTTP error — {e}"); continue

        status = data.get("status", "")
        if status == "ERROR":
            print(f"    chunk {cs}→{ce}: API error — {data.get('error', data.get('message', '?'))}")
            continue
        if status == "NOT_AUTHORIZED":
            print(f"    NOT_AUTHORIZED — check your API key or plan tier"); continue
        if status == "DELAYED":
            pass  # free-tier returns DELAYED but results are still populated

        results = data.get("results", [])
        count = 0
        for r in results:
            ts_utc = datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc)
            if ts_utc in seen:
                continue
            seen.add(ts_utc)
            all_bars.append((ts_utc, float(r["o"]), float(r["h"]),
                              float(r["l"]), float(r["c"])))
            count += 1
        has_more = bool(data.get("next_url"))
        print(f"    {cs} → {ce}: {count} bars{' ⚠ truncated' if has_more else ''}")

        if i < len(chunks) - 1:
            time.sleep(13)   # 5 calls/min free tier

    all_bars.sort(key=lambda b: b[0])
    if all_bars:
        print(f"    Total: {len(all_bars)} bars  "
              f"{all_bars[0][0].strftime('%Y-%m-%d')} → {all_bars[-1][0].strftime('%Y-%m-%d')}")
        if use_cache:
            _save_bars(all_bars, ticker, interval, start, end)
    else:
        print("    NO DATA")
    return all_bars


# ── Regime filter (daily Golden Cross + VIX) ────────────────────────────────

def fetch_regime_data() -> Dict[str, bool]:
    """
    Per-day regime gate built from 2 years of daily QQQ + VIX data.
    A day is BLOCKED (False) if ANY of:
      • QQQ price  < 200-day SMA       (price below long-term trend)
      • QQQ 50-day EMA < 200-day EMA   (Golden Cross broken → bear)
      • VIX close  >= 25               (elevated fear / panic zone)
      • VIX spiked >= 15% vs prior day (intraday panic signal)
    Sources: Cracking Markets momentum system; Zarattini et al. (2024 SSRN);
             Rob Hanna VIX-in-portfolio research.
    """
    import pandas as pd
    print("  Fetching regime data (QQQ+VIX daily 2y)...", end=" ", flush=True)
    qqq_d = yf.download("QQQ", period="2y", interval="1d", progress=False, auto_adjust=True)
    vix_d = yf.download("^VIX", period="2y", interval="1d", progress=False, auto_adjust=True)
    for df in (qqq_d, vix_d):
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)

    closes  = qqq_d["Close"]
    ema50   = closes.ewm(span=50,  adjust=False).mean()
    ema200  = closes.ewm(span=200, adjust=False).mean()
    sma200  = closes.rolling(200).mean()

    vix_cl  = vix_d["Close"] if not vix_d.empty else pd.Series(dtype=float)
    vix_prv = vix_cl.shift(1)

    regime: Dict[str, bool] = {}
    reasons: Dict[str, List[str]] = {}
    for dt in closes.index:
        ds = str(dt)[:10]
        c   = float(closes.loc[dt])
        e50 = float(ema50.loc[dt])
        e200= float(ema200.loc[dt])
        s200= float(sma200.loc[dt]) if not pd.isna(sma200.loc[dt]) else 0.0

        blocked = []
        if s200 > 0 and c < s200:
            blocked.append("below-200sma")
        if e50 < e200:
            blocked.append("death-cross")
        if dt in vix_cl.index:
            vv = float(vix_cl.loc[dt])
            vp = float(vix_prv.loc[dt]) if not pd.isna(vix_prv.loc[dt]) else vv
            if vv >= 25.0:
                blocked.append(f"VIX={vv:.0f}")
            if vv >= vp * 1.15:
                blocked.append(f"VIX-spike({vv:.0f}>{vp:.0f})")

        regime[ds]  = len(blocked) == 0
        reasons[ds] = blocked

    ok  = sum(1 for v in regime.values() if v)
    tot = len(regime)
    blk = tot - ok
    # count unique reason types
    from collections import Counter
    all_reasons: List[str] = []
    for bl in reasons.values():
        all_reasons.extend(bl)
    reason_counts = Counter(r.split("=")[0].split("(")[0] for r in all_reasons)
    detail = "  ".join(f"{k}:{v}" for k, v in reason_counts.most_common())
    print(f"{ok}/{tot} days green  ({blk} blocked — {detail})")
    return regime


# ── Realised vol (resolution-aware) ──────────────────────────────────────────

def realized_vol_annual(prices: List[float], period: int, bars_per_year: int) -> float:
    if len(prices) < 2:
        return 0.0
    arr = np.array(prices[-period:], dtype=float)
    rets = np.diff(arr) / arr[:-1]
    return float(np.std(rets)) * np.sqrt(bars_per_year) * 100.0

# ── Simulation engine ─────────────────────────────────────────────────────────

class SimPosition:
    def __init__(self, ticker, qty, entry_price, bar_idx, ts):
        self.ticker = ticker; self.qty = qty
        self.entry_price = entry_price; self.bar_idx = bar_idx; self.ts = ts


def run_simulation(qqq_bars: List[Tuple], spy_bars: List[Tuple], mode: dict,
                   regime: Dict[str, bool],
                   entry_threshold: float = ENTRY_THRESHOLD,
                   min_profit_bps: float  = MIN_PROFIT_BPS,
                   verbose: bool = True,
                   vol_target_enabled: bool = False,
                   validate: bool = False) -> dict:
    spy_map = {b[0]: b for b in spy_bars}
    last_spy_price: float = spy_bars[0][4] if spy_bars else 0.0   # fallback for data gaps
    bars_per_year         = mode["bars_per_year"]
    squeeze_bw_pct        = mode["squeeze_bw_pct"]
    min_hold_bars         = mode["min_hold_bars"]
    eod_close_min         = mode["eod_close_minutes"]
    no_entry_min          = mode["no_entry_minutes"]
    open_skip_min         = mode["open_skip_minutes"]
    eod_tighten_stop_min  = mode["eod_tighten_stop_min"]
    eod_tighten_stop_pct  = mode["eod_tighten_stop_pct"]
    breakeven_bps         = mode["breakeven_bps"]
    sl_cooldown_min       = mode["stop_loss_cooldown"]
    bar_unit              = mode["bar_unit"]
    warmup_bars           = mode["warmup_bars"]

    cash = STARTING_CASH
    position: Optional[SimPosition] = None
    trades: List[dict] = []
    closes_buf: Dict[str, List[float]] = {ETF_PRIMARY: [], ETF_SECONDARY: []}

    squeeze_blocks = vol_blocks = cash_blocks = cooldown_blocks = regime_blocks = regime_flip_exits = 0
    entry_blocked_until: Optional[datetime] = None
    stop_losses_today = 0
    stop_loss_date: Optional[str] = None

    out = print if verbose else (lambda *a, **k: None)
    out("\n" + "=" * 72)
    out(f"SIMULATION: {mode['label']}")
    out("=" * 72)

    for i, (ts, o, h, l, c) in enumerate(qqq_bars):
        closes_buf[ETF_PRIMARY].append(c)
        if ts in spy_map:
            last_spy_price = spy_map[ts][4]
            closes_buf[ETF_SECONDARY].append(last_spy_price)
        if len(closes_buf[ETF_PRIMARY]) > 200:
            closes_buf[ETF_PRIMARY] = closes_buf[ETF_PRIMARY][-200:]
        if len(closes_buf[ETF_SECONDARY]) > 200:
            closes_buf[ETF_SECONDARY] = closes_buf[ETF_SECONDARY][-200:]

        if len(closes_buf[ETF_PRIMARY]) < warmup_bars:
            continue

        signals: Dict[str, float] = {}
        for tkr in (ETF_PRIMARY, ETF_SECONDARY):
            buf = closes_buf[tkr]
            if len(buf) >= 21:
                signals[tkr] = _compute_composite(buf)
        if not signals:
            continue

        # Price: use held ticker's price for exit checks.
        # If SPY has a data gap at this timestamp, use last known SPY price (not QQQ price).
        if position is not None and position.ticker == ETF_SECONDARY:
            price = spy_map[ts][4] if ts in spy_map else last_spy_price
        else:
            price = c

        _, _, _, bw_pct = _bollinger(closes_buf[ETF_PRIMARY], 20, 2.0)
        vol = realized_vol_annual(closes_buf[ETF_PRIMARY], 60, bars_per_year)

        # Session-phase timing — DST-aware (EDT = UTC-4, EST = UTC-5)
        _ts_et    = ts.astimezone(_ET)
        _close_et = _ts_et.replace(hour=16, minute=0,  second=0, microsecond=0)
        _open_et  = _ts_et.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_utc = _close_et.astimezone(timezone.utc)
        open_utc  = _open_et.astimezone(timezone.utc)
        mins_to_close   = (close_utc - ts).total_seconds() / 60.0 if ts < close_utc else None
        mins_since_open = (ts - open_utc).total_seconds() / 60.0  if ts >= open_utc else None

        # Market-hours gate: skip extended-hours bars from trading logic.
        # Still updates closes_buf above for signal continuity.
        # Exits (stop, EOD, regime-flip) are still allowed outside hours so an
        # overnight-held position can be closed at first regular-session open.
        in_session = mins_since_open is not None and mins_to_close is not None

        if position is None:
            if not in_session:
                continue   # never enter outside regular session

            best_tkr = max(signals, key=signals.__getitem__)
            best_sig = signals[best_tkr]

            # Opening skip: wait for price discovery after the bell
            if mins_since_open < open_skip_min:
                continue
            # No-entry window: not enough time to develop a trade before close
            if mins_to_close <= no_entry_min:
                continue

            # Late-session gate: require higher conviction when runway is short
            effective_threshold = (
                LATE_ENTRY_THRESHOLD
                if mins_to_close is not None and mins_to_close <= LATE_ENTRY_MINUTES
                else entry_threshold
            )
            if best_sig > effective_threshold:
                if not regime.get(ts.strftime("%Y-%m-%d"), True):
                    regime_blocks += 1; continue
                if entry_blocked_until and ts < entry_blocked_until:
                    cooldown_blocks += 1; continue
                if vol >= MAX_VOL_PCT:
                    vol_blocks += 1; continue
                if bw_pct < squeeze_bw_pct:
                    squeeze_blocks += 1; continue
                invest = cash - FREE_CASH_THRESHOLD
                _vol_target_window = 60
                _target_vol_pct    = 15.0
                _vol_min_mult      = 0.25
                _vol_max_mult      = 1.0
                _rv = realized_vol_annual(closes_buf[best_tkr], _vol_target_window, bars_per_year)
                if vol_target_enabled and _rv > 0:
                    _mult = max(_vol_min_mult, min(_vol_max_mult, _target_vol_pct / _rv))
                    invest *= _mult
                else:
                    _mult = 1.0
                if invest < MIN_INVEST:
                    cash_blocks += 1; continue
                if best_tkr == ETF_SECONDARY:
                    entry_price = spy_map[ts][4] if ts in spy_map else last_spy_price
                else:
                    entry_price = closes_buf[best_tkr][-1]
                commission_buy = invest * COMMISSION_PCT
                qty = (invest - commission_buy) / entry_price
                position = SimPosition(best_tkr, qty, entry_price, i, ts)
                cash -= invest   # invest includes commission
                trades.append({
                    "ts": ts, "side": "BUY", "ticker": best_tkr,
                    "price": entry_price, "qty": qty, "invest": invest,
                    "commission": commission_buy,
                    "signal": best_sig, "bw_pct": bw_pct, "vol": vol,
                    "vol_mult": _mult, "realized_vol": _rv,
                })
        else:
            bars_held = i - position.bar_idx
            if bars_held < min_hold_bars:
                continue

            sig       = signals.get(position.ticker, 0.0)
            near_close = mins_to_close is not None and mins_to_close <= eod_tighten_stop_min

            # Session-phase stop: tighter near close
            stop_pct  = eod_tighten_stop_pct if near_close else STOP_LOSS_PCT
            sl_px     = position.entry_price * (1 - stop_pct / 100)

            # Breakeven stop: if profitable near close, floor stop at entry
            current_profit_bps = (price - position.entry_price) / position.entry_price * 10_000
            if near_close and current_profit_bps >= breakeven_bps:
                sl_px = max(sl_px, position.entry_price)

            force_eod  = mins_to_close is not None and mins_to_close <= eod_close_min
            force_stop = price <= sl_px

            # Regime-flip exit: regime turned red on a day after entry → exit at
            # first bar of that day (simulates "exit at next open" rule).
            today_str = ts.strftime("%Y-%m-%d")
            entry_str = position.ts.strftime("%Y-%m-%d")
            force_regime_flip = (today_str != entry_str and
                                  not regime.get(today_str, True))

            if force_regime_flip:
                reason = f"regime-flip({today_str})"
                regime_flip_exits += 1
            elif not force_eod and not force_stop:
                if sig >= EXIT_THRESHOLD:
                    continue
                if price <= position.entry_price * (1 + min_profit_bps / 10_000):
                    continue
                reason = f"signal(comp={sig:.3f})"
            elif force_stop:
                tag = "EOD-stop" if near_close else "stop-loss"
                reason = f"{tag}({stop_pct}%,${price:.2f}≤${sl_px:.2f})"
            else:
                reason = f"MOC({mins_to_close:.0f}m left)"

            gross_proceeds  = position.qty * price
            commission_sell = gross_proceeds * COMMISSION_PCT
            proceeds        = gross_proceeds - commission_sell
            # P&L in bps: net proceeds vs original invested capital (both legs net of commission)
            invested        = position.qty * position.entry_price  # net shares × entry
            profit_bps      = (proceeds - invested) / invested * 10_000
            cash += proceeds
            trades.append({
                "ts": ts, "side": "SELL", "ticker": position.ticker,
                "price": price, "qty": position.qty, "proceeds": proceeds,
                "profit_bps": profit_bps, "bars_held": bars_held,
                "signal": sig, "bw_pct": bw_pct,
                "entry_price": position.entry_price, "reason": reason,
                "commission": commission_sell,
            })
            position = None
            if force_stop:
                today = ts.strftime("%Y-%m-%d")
                if stop_loss_date != today:
                    stop_loss_date = today; stop_losses_today = 0
                stop_losses_today += 1
                if stop_losses_today >= MAX_STOP_LOSSES_DAY:
                    next_open = (ts + timedelta(days=1)).replace(
                        hour=14, minute=30, second=0, microsecond=0)
                    while next_open.weekday() >= 5:
                        next_open += timedelta(days=1)
                    entry_blocked_until = next_open
                else:
                    entry_blocked_until = ts + timedelta(minutes=sl_cooldown_min)
            else:
                entry_blocked_until = None

    # ── Trade log ────────────────────────────────────────────────────────────
    if trades:
        for t in trades:
            ts_str = t["ts"].strftime("%a %m/%d %H:%M")
            if t["side"] == "BUY":
                out(f"  BUY  {t['ticker']} @ ${t['price']:.2f}  "
                      f"invest=${t['invest']:,.0f} ({t['qty']:.2f}sh)  "
                      f"comp={t['signal']:.3f}  bw={t['bw_pct']:.3f}%  "
                      f"vol={t['vol']:.1f}%  [{ts_str}]")
            else:
                out(f"  SELL {t['ticker']} @ ${t['price']:.2f}  "
                      f"P&L={t['profit_bps']:+.1f}bps  "
                      f"held={t['bars_held']}{bar_unit}  "
                      f"{t['reason']}  "
                      f"proceeds=${t['proceeds']:,.0f}  [{ts_str}]")
    else:
        out("  NO TRADES EXECUTED")

    if position:
        lp = spy_bars[-1][4] if (position.ticker == ETF_SECONDARY and spy_bars) \
             else qqq_bars[-1][4]
        unr = (lp - position.entry_price) / position.entry_price * 10_000
        mv  = position.qty * lp
        out(f"\n  OPEN AT PERIOD END: {position.qty:.2f}sh {position.ticker} "
              f"@ ${position.entry_price:.2f}  mkt=${lp:.2f}  "
              f"P&L={unr:+.1f}bps  mkt_value=${mv:,.0f}")

    out(f"\n  Filter blocks — regime:{regime_blocks}  squeeze:{squeeze_blocks}  "
          f"vol:{vol_blocks}  cash:{cash_blocks}  cooldown:{cooldown_blocks}"
          f"  regime-flip-exits:{regime_flip_exits}")

    # ── Performance summary ──────────────────────────────────────────────────
    out("\n" + "=" * 72)
    out("PERFORMANCE SUMMARY")
    out("=" * 72)

    buys  = [t for t in trades if t["side"] == "BUY"]
    sells = [t for t in trades if t["side"] == "SELL"]
    out(f"  Period:   {mode['label']}")
    out(f"  Trades:   {len(buys)} entries / {len(sells)} exits")

    avg_win = avg_loss = 0.0
    if sells:
        pnl_bps  = [t["profit_bps"] for t in sells]
        wins     = [p for p in pnl_bps if p > 0]
        stops    = [t for t in sells if "stop-loss" in t["reason"]]
        eods     = [t for t in sells if "EOD" in t["reason"]]
        signals_ = [t for t in sells if "signal" in t["reason"]]
        rflips   = [t for t in sells if "regime-flip" in t["reason"]]
        out(f"  Win rate: {len(wins)}/{len(sells)} = {len(wins)/len(sells)*100:.0f}%")
        out(f"  Exit breakdown: signal={len(signals_)}  stop-loss={len(stops)}  "
              f"EOD={len(eods)}  regime-flip={len(rflips)}")
        out(f"  Closed P&L: {sum(pnl_bps):+.1f} bps total  "
              f"(best={max(pnl_bps):+.1f}  worst={min(pnl_bps):+.1f})")
        avg_win  = sum(p for p in pnl_bps if p > 0) / len(wins) if wins else 0
        avg_loss = sum(p for p in pnl_bps if p <= 0) / max(1, len(sells) - len(wins))
        out(f"  Avg win:  {avg_win:+.1f} bps   Avg loss: {avg_loss:+.1f} bps")
        out(f"  Avg hold: {sum(t['bars_held'] for t in sells)/len(sells):.1f} {bar_unit}s")
        gross_usd   = sum((t["profit_bps"] / 10_000) * (t["qty"] * t["entry_price"]) for t in sells)
        total_comm  = sum(t.get("commission", 0) for t in trades)   # buy + sell commissions
        out(f"  Gross P&L: ${gross_usd:+,.2f}  (commission paid: ${total_comm:,.0f}  "
              f"= {total_comm/STARTING_CASH*100:.2f}% of capital)")

    if vol_target_enabled and buys:
        vol_mults = [t["vol_mult"] for t in buys]
        out(f"  Avg vol_mult: {sum(vol_mults)/len(vol_mults):.2f}  "
              f"(range: {min(vol_mults):.2f}–{max(vol_mults):.2f})")

    final = cash
    if position:
        lp = spy_bars[-1][4] if (position.ticker == ETF_SECONDARY and spy_bars) \
             else qqq_bars[-1][4]
        final += position.qty * lp
    out(f"  Starting cash: ${STARTING_CASH:,.0f}")
    out(f"  Final value:   ${final:,.0f}  ({(final/STARTING_CASH - 1)*100:+.3f}%)")

    # ── Alpha / risk metrics ─────────────────────────────────────────────────
    if sells:
        out("\n" + "=" * 72)
        out("RISK-ADJUSTED METRICS")
        out("=" * 72)

        # Per-trade fractional returns (relative to invested capital each trade)
        trade_rets = np.array([
            (t["profit_bps"] / 10_000) for t in sells
        ], dtype=float)

        # Equity curve (cumulative product of 1 + trade_return)
        # Each trade compounds the prior equity
        eq_factors = 1.0 + trade_rets
        cum_eq = np.cumprod(eq_factors)           # normalised (starts at 1.0)
        peak   = np.maximum.accumulate(cum_eq)
        dd_series = (cum_eq - peak) / peak
        max_dd = float(dd_series.min())            # negative number

        # Annualisation factor: trades/year based on actual period length
        period_days = (qqq_bars[-1][0] - qqq_bars[0][0]).days or 1
        years        = period_days / 365.25
        trades_per_yr = len(sells) / years

        RF_ANNUAL = 0.053                          # ~5.3% risk-free (2026 T-bill)
        rf_per_trade = RF_ANNUAL / trades_per_yr
        excess = trade_rets - rf_per_trade

        sharpe  = float(np.mean(excess) / (np.std(trade_rets) + 1e-10) * np.sqrt(trades_per_yr))

        down = trade_rets[trade_rets < 0]
        down_std = float(np.std(down)) if len(down) > 1 else float(np.std(trade_rets))
        sortino = float(np.mean(excess) / (down_std + 1e-10) * np.sqrt(trades_per_yr))

        total_ret = final / STARTING_CASH - 1
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else total_ret
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else float("inf")

        # Profit factor (gross USD profit / gross USD loss)
        gp = sum((t["profit_bps"]/10_000)*(t["qty"]*t["entry_price"]) for t in sells if t["profit_bps"] > 0)
        gl = abs(sum((t["profit_bps"]/10_000)*(t["qty"]*t["entry_price"]) for t in sells if t["profit_bps"] <= 0))
        profit_factor = gp / gl if gl > 0 else float("inf")

        # R-multiple (avg_win / |avg_loss| in bps)
        r_multiple = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")

        # Expected value per trade
        win_rate_f = len([p for p in pnl_bps if p > 0]) / len(pnl_bps)
        ev_bps = win_rate_f * avg_win + (1 - win_rate_f) * avg_loss

        # Alpha vs QQQ buy-and-hold over same period
        qqq_bh = (qqq_bars[-1][4] - qqq_bars[warmup_bars][4]) / qqq_bars[warmup_bars][4]
        alpha   = total_ret - qqq_bh

        # SPY buy-and-hold
        spy_bh = (spy_bars[-1][4] - spy_bars[0][4]) / spy_bars[0][4] if spy_bars else 0.0

        out(f"  Sharpe ratio (ann.):  {sharpe:+.2f}")
        out(f"  Sortino ratio (ann.): {sortino:+.2f}")
        out(f"  Max drawdown:         {max_dd*100:+.2f}%")
        out(f"  Calmar ratio:         {calmar:+.2f}")
        out(f"  Profit factor:        {profit_factor:.2f}x")
        out(f"  R-multiple:           {r_multiple:.3f}  (avg_win/|avg_loss| in bps)")
        out(f"  Expected value/trade: {ev_bps:+.1f} bps")
        out(f"  ─────────────────────────────────────────────────")
        out(f"  Strategy total return:{total_ret*100:+.2f}%  (ann. {annual_ret*100:+.1f}%)")
        out(f"  QQQ buy-and-hold:     {qqq_bh*100:+.2f}%  over same period")
        out(f"  SPY buy-and-hold:     {spy_bh*100:+.2f}%  over same period")
        out(f"  Alpha vs QQQ:         {alpha*100:+.2f}%")

        # Equity curve peak & worst drawdown window
        worst_idx = int(np.argmin(dd_series))
        peak_idx  = int(np.argmax(cum_eq[:worst_idx+1])) if worst_idx > 0 else 0
        if worst_idx < len(sells):
            pk_trade = sells[peak_idx]["ts"].strftime("%m/%d") if peak_idx < len(sells) else "?"
            tr_trade = sells[worst_idx]["ts"].strftime("%m/%d")
            out(f"  Drawdown window:      peak={pk_trade} → trough={tr_trade}")

        # ── Statistical validation (DSR + Bootstrap) ─────────────────────────
        if validate and len(sells) >= 2:
            from stat_validate import deflated_sharpe_ratio, stationary_block_bootstrap

            _NUM_TRIALS = 16   # 8 thresholds × 2 windows from recent threshold sweep
            # Both DSR and bootstrap use excess returns (trade_rets − rf_per_trade,
            # already computed above) so Sharpe figures are consistent with the
            # annualised Sharpe printed in RISK-ADJUSTED METRICS above.
            # freq=trades_per_yr converts per-trade excess returns to annualised SR.
            _dsr = deflated_sharpe_ratio(excess, num_trials=_NUM_TRIALS,
                                         freq=trades_per_yr)
            _bbs = stationary_block_bootstrap(excess, n_resamples=1000,
                                              mean_block=10, seed=42)
            _sig = "SIGNIFICANT" if _dsr["dsr_pvalue"] < 0.05 else "not significant"

            # Bootstrap Sharpe CIs are in per-trade units; scale to annualised for display
            _sf = np.sqrt(trades_per_yr)
            _sh5  = _bbs["sharpe_ci_5"]  * _sf
            _sh50 = _bbs["sharpe_ci_50"] * _sf
            _sh95 = _bbs["sharpe_ci_95"] * _sf

            out("\n" + "=" * 72)
            out("STATISTICAL VALIDATION  (DSR + Stationary Block Bootstrap)")
            out("=" * 72)
            out(f"  Raw Sharpe:           {_dsr['sharpe']:+.2f}")
            out(f"  DSR z-score:          {_dsr['deflated_sharpe']:+.2f}  (num_trials={_NUM_TRIALS})")
            out(f"  DSR p-value:          {_dsr['dsr_pvalue']:.4f}  → {_sig} at 5%")
            out(f"  Trades:               {_dsr['n_trades']}")
            out(f"")
            out(f"  Bootstrap (1000 resamples, mean_block=10 trades):")
            out(f"    Sharpe:       median={_sh50:+.2f}  90% CI [{_sh5:+.2f}, {_sh95:+.2f}]")
            out(f"    Total return: median={_bbs['return_ci_50']*100:+.2f}%  "
                  f"90% CI [{_bbs['return_ci_5']*100:+.2f}%, {_bbs['return_ci_95']*100:+.2f}%]")
            out(f"    Max DD:       median={_bbs['maxdd_ci_50']*100:+.2f}%  "
                  f"90% CI [{_bbs['maxdd_ci_5']*100:+.2f}%, {_bbs['maxdd_ci_95']*100:+.2f}%]")

    # ── Signal distribution ──────────────────────────────────────────────────
    out("\n" + "=" * 72)
    out("SIGNAL DISTRIBUTION (after warmup)")
    out("=" * 72)
    all_comps = []
    buf: List[float] = []
    for _, _, _, _, c_bar in qqq_bars:
        buf.append(c_bar)
        if len(buf) >= warmup_bars:
            all_comps.append(_compute_composite(buf[-200:]))
    if all_comps:
        arr = np.array(all_comps)
        above   = (arr > ENTRY_THRESHOLD).sum()
        below   = (arr < EXIT_THRESHOLD).sum()
        neutral = len(arr) - above - below
        out(f"  Bars: {len(arr)}  Range: [{arr.min():.3f}, {arr.max():.3f}]  "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}")
        out(f"  > {ENTRY_THRESHOLD} (ENTRY): {above:4d} ({above/len(arr)*100:.1f}%)")
        out(f"  < {EXIT_THRESHOLD} (EXIT):  {below:4d} ({below/len(arr)*100:.1f}%)")
        out(f"  Neutral:           {neutral:4d} ({neutral/len(arr)*100:.1f}%)")

    # ── Monthly breakdown (hourly mode only) ─────────────────────────────────
    if mode["interval"] in ("1h", "5m") and sells:
        out("\n" + "=" * 72)
        out("MONTHLY BREAKDOWN")
        out("=" * 72)
        from collections import defaultdict
        monthly: Dict[str, List[float]] = defaultdict(list)
        monthly_buys: Dict[str, int] = defaultdict(int)
        for t in sells:
            key = t["ts"].strftime("%Y-%m")
            monthly[key].append(t["profit_bps"])
        for t in buys:
            monthly_buys[t["ts"].strftime("%Y-%m")] += 1

        for month in sorted(monthly):
            pnls = monthly[month]
            w = sum(1 for p in pnls if p > 0)
            out(f"  {month}:  {monthly_buys[month]:2d} entries  "
                  f"{len(pnls):2d} exits  "
                  f"win={w}/{len(pnls)}  "
                  f"P&L={sum(pnls):+.1f}bps  "
                  f"(best={max(pnls):+.1f}  worst={min(pnls):+.1f})")

    # ── BB squeeze monthly (hourly mode) ─────────────────────────────────────
    if mode["interval"] in ("1h", "5m"):
        out("\n" + "=" * 72)
        out(f"BB SQUEEZE BY MONTH (bw < {squeeze_bw_pct}%)")
        out("=" * 72)
        from collections import defaultdict
        month_bws: Dict[str, List[float]] = defaultdict(list)
        buf2: List[float] = []
        for ts_b, _, _, _, c_b in qqq_bars:
            buf2.append(c_b)
            _, _, _, bw = _bollinger(buf2[-200:] if len(buf2) > 200 else buf2, 20, 2.0)
            month_bws[ts_b.strftime("%Y-%m")].append(bw)
        for month in sorted(month_bws):
            bws = month_bws[month]
            sq = sum(1 for b in bws if b < squeeze_bw_pct)
            out(f"  {month}: squeezed {sq:3d}/{len(bws)}  "
                  f"avg_bw={sum(bws)/len(bws):.2f}%  max_bw={max(bws):.2f}%")

    # ── Monthly stats (always computed, used by optimizer) ───────────────────
    from collections import defaultdict
    _monthly: Dict[str, List[float]] = defaultdict(list)
    for t in sells:
        _monthly[t["ts"].strftime("%Y-%m")].append(t["profit_bps"])

    # ── Return summary dict (used by sweep mode) ─────────────────────────────
    return {
        "entry_thr":  entry_threshold,
        "min_profit": min_profit_bps,
        "trades":     len(sells),
        "win_rate":   len(wins) / len(sells) if sells else 0,
        "total_bps":  sum(pnl_bps) if sells else 0,
        "ev_bps":     ev_bps if sells else 0,
        "sharpe":     sharpe if sells else 0,
        "max_dd":     max_dd * 100 if sells else 0,
        "total_ret":  (final / STARTING_CASH - 1) * 100,
        "alpha":      alpha * 100 if sells else 0,
        "commission": sum(t.get("commission", 0) for t in trades),
        "avg_win":    avg_win,
        "avg_loss":   avg_loss,
        "monthly":    {m: sum(v) for m, v in _monthly.items()},
        "monthly_trades": {m: len(v) for m, v in _monthly.items()},
        "monthly_wins":   {m: sum(1 for x in v if x > 0) for m, v in _monthly.items()},
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    # ── Cross-sectional rotation mode (--xs / --xs-cached) ───────────────────
    # --xs          : fetch 6-ETF universe from Polygon, run XS simulation, print report
    # --xs-cached   : run XS simulation from existing cache (no fetch)
    # Both require --pg-start / --pg-end date range (defaulting to 6-month window).
    # --xs requires --pg-key YOUR_POLYGON_API_KEY
    if "--xs" in args or "--xs-cached" in args:
        from backtest_universe import UNIVERSE_6, fetch_universe, align_bars
        from backtest_xs import run_xs_simulation, print_xs_summary

        pg_start = args[args.index("--pg-start") + 1] if "--pg-start" in args else "2025-10-01"
        pg_end   = args[args.index("--pg-end")   + 1] if "--pg-end"   in args else "2026-04-04"

        print(f"ETFMonitor XS Backtest — cross-sectional 6-ETF rotation")
        print(f"  Universe: {UNIVERSE_6}")
        print(f"  Period:   {pg_start} → {pg_end}")
        print("=" * 72)

        if "--xs" in args:
            try:
                pg_key = args[args.index("--pg-key") + 1]
            except (ValueError, IndexError):
                sys.exit("--xs requires --pg-key YOUR_POLYGON_API_KEY")
            print("Fetching 6-ETF universe from Polygon...")
            bars_by_ticker = fetch_universe(UNIVERSE_6, pg_start, pg_end, pg_key,
                                            multiplier=5, use_cache=True)
        else:
            # --xs-cached: load from existing cache files
            print("Loading 6-ETF universe from cache...")
            bars_by_ticker = {}
            for tkr in UNIVERSE_6:
                cached = _load_bars(tkr, "5m", pg_start, pg_end)
                if cached:
                    bars_by_ticker[tkr] = cached
                else:
                    print(f"  WARNING: no cache for {tkr} {pg_start}→{pg_end} — skipping")

        if not bars_by_ticker:
            sys.exit("No data loaded. Fetch first with --xs --pg-key KEY")

        print(f"\nLoaded {len(bars_by_ticker)} tickers: {list(bars_by_ticker.keys())}")
        for tkr, bars in bars_by_ticker.items():
            print(f"  {tkr}: {len(bars)} bars  "
                  f"{bars[0][0].strftime('%Y-%m-%d')} → {bars[-1][0].strftime('%Y-%m-%d')}")

        print("\nFetching regime data...")
        regime = fetch_regime_data()

        print("\nRunning cross-sectional simulation...")
        result = run_xs_simulation(bars_by_ticker, regime)
        print_xs_summary(result, validate=True)
        sys.exit(0)

    # ── Cached Polygon mode (--6m) ────────────────────────────────────────────
    # Reads directly from data_cache/ without hitting the Polygon API.
    # Falls back to a clear error if the cache files are missing.
    # Usage: python3 simulate_etf_monitor.py --6m [--pg-start YYYY-MM-DD --pg-end YYYY-MM-DD]
    if "--10m" in args:
        pg_start = args[args.index("--pg-start") + 1] if "--pg-start" in args else "2025-10-01"
        pg_end   = args[args.index("--pg-end")   + 1] if "--pg-end"   in args else "2026-04-04"
        _FULL_START, _FULL_END = "2025-10-01", "2026-04-04"
        mode = MODES["pg-10m"].copy()
        mode["label"] = f"10-min bars (resampled)  {pg_start} → {pg_end}"

        print(f"ETFMonitor Backtest — {mode['label']}")
        print("=" * 72)
        print("Loading cached 5m bar data and resampling to 10m...")
        qqq_5m = _load_bars(ETF_PRIMARY,   "5m", _FULL_START, _FULL_END)
        spy_5m = _load_bars(ETF_SECONDARY, "5m", _FULL_START, _FULL_END)
        if qqq_5m is None:
            sys.exit("Cache miss: no QQQ 5m data. Run --pg5m first.")
        # Slice to requested date range
        start_dt = datetime.strptime(pg_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(pg_end,   "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
        qqq_5m = [b for b in qqq_5m if start_dt <= b[0] < end_dt]
        spy_5m = [b for b in spy_5m if start_dt <= b[0] < end_dt] if spy_5m else []
        qqq = resample_bars(qqq_5m, 2)
        spy = resample_bars(spy_5m, 2) if spy_5m else []
        print(f"  QQQ: {len(qqq_5m)} × 5m → {len(qqq)} × 10m bars")
        if spy:
            print(f"  SPY: {len(spy_5m)} × 5m → {len(spy)} × 10m bars")

    elif "--6m" in args or "--6m-1m" in args:
        pg_start = args[args.index("--pg-start") + 1] if "--pg-start" in args else "2025-10-01"
        pg_end   = args[args.index("--pg-end")   + 1] if "--pg-end"   in args else "2026-04-04"
        if "--6m-1m" in args:
            mode = MODES["pg-1m"].copy()
            interval = "1m"
            mode["label"] = f"Cached Polygon 1-min  {pg_start} → {pg_end}"
        else:
            mode = MODES["pg-5m"].copy()
            interval = "5m"
            mode["label"] = f"Cached Polygon 5-min  {pg_start} → {pg_end}"

        print(f"ETFMonitor Backtest — {mode['label']}")
        print("=" * 72)
        print("Loading cached bar data...")
        qqq = _load_bars(ETF_PRIMARY,   interval, pg_start, pg_end)
        spy = _load_bars(ETF_SECONDARY, interval, pg_start, pg_end)
        if qqq is None:
            sys.exit(f"Cache miss: no QQQ {interval} data for {pg_start}→{pg_end}.\n"
                     f"Run first with: python3 simulate_etf_monitor.py --pg --pg-key KEY "
                     f"--pg-start {pg_start} --pg-end {pg_end}")
        if spy is None:
            print(f"  ⚠ No SPY cache for {pg_start}→{pg_end} — SPY signals will fall back to QQQ")
            spy = []

    # ── Polygon.io live-fetch mode ─────────────────────────────────────────────
    elif "--pg" in args or "--pg5m" in args:
        # Usage: python3 simulate_etf_monitor.py --pg    --pg-key KEY [--pg-start YYYY-MM-DD --pg-end YYYY-MM-DD]
        #        python3 simulate_etf_monitor.py --pg5m  --pg-key KEY [--pg-start YYYY-MM-DD --pg-end YYYY-MM-DD]
        try:
            pg_key = args[args.index("--pg-key") + 1]
        except (ValueError, IndexError):
            sys.exit("--pg / --pg5m requires --pg-key YOUR_POLYGON_API_KEY")

        pg_start = args[args.index("--pg-start") + 1] if "--pg-start" in args else "2025-10-01"
        pg_end   = args[args.index("--pg-end")   + 1] if "--pg-end"   in args else "2026-04-04"

        if "--pg5m" in args:
            mode = MODES["pg-5m"].copy()
            mode["label"] = f"Polygon.io 5-min  {pg_start} → {pg_end}"
            mult = 5
        else:
            mode = MODES["pg-1m"].copy()
            mode["label"] = f"Polygon.io 1-min  {pg_start} → {pg_end}"
            mult = 1

        print(f"ETFMonitor Backtest — {mode['label']}")
        print("=" * 72)
        print("Fetching data...")
        use_cache = "--no-cache" not in args
        qqq = fetch_bars_polygon(ETF_PRIMARY,   pg_start, pg_end, pg_key, multiplier=mult, use_cache=use_cache)
        # Sleep between tickers to avoid Polygon 429 on free tier (5 calls/min)
        if not use_cache or not os.path.exists(_cache_path(ETF_SECONDARY, f"{mult}m", pg_start, pg_end)):
            time.sleep(15)
        spy = fetch_bars_polygon(ETF_SECONDARY, pg_start, pg_end, pg_key, multiplier=mult, use_cache=use_cache)

    # ── yfinance modes ────────────────────────────────────────────────────────
    else:
        if "--jan-mar" in args:
            mode = MODES["1h"]
        elif "--5m-month" in args:
            mode = MODES["5m-month"]
        elif "--5m" in args:
            mode = MODES["5m"]
        elif "--1m-month" in args:
            mode = MODES["1m-month"]
        else:
            mode = MODES["1m"]

        print(f"ETFMonitor Backtest — {mode['label']}")
        print("=" * 72)
        print("Fetching data...")
        if mode["interval"] == "1m" and mode["period"] is None:
            qqq = fetch_bars_chunked(ETF_PRIMARY)
            spy = fetch_bars_chunked(ETF_SECONDARY)
        else:
            qqq = fetch_bars(ETF_PRIMARY,  mode)
            spy = fetch_bars(ETF_SECONDARY, mode)

    # ── Per-run parameter overrides ───────────────────────────────────────────
    def _arg(flag, default):
        return type(default)(args[args.index(flag) + 1]) if flag in args else default

    regime = fetch_regime_data()
    if not qqq:
        sys.exit("No QQQ data")

    # ── Parameter sweep mode ──────────────────────────────────────────────────
    if "--sweep" in args:
        entry_thresholds = [0.40, 0.45, 0.50, 0.55]
        min_profits      = [5, 15, 25, 35, 50]
        print(f"\n{'='*88}")
        print(f"PARAMETER SWEEP  (commission={COMMISSION_PCT*100:.1f}% per side = {COMMISSION_PCT*2*10000:.0f} bps round-trip)")
        print(f"{'='*88}")
        print(f"{'entry':>7} {'minP':>5} {'trd':>4} {'win%':>5} {'EV/t':>7} {'total':>8} {'ret%':>7} {'alpha%':>8} {'sharpe':>7} {'maxDD%':>7}")
        print(f"{'-'*88}")
        for et in entry_thresholds:
            for mp in min_profits:
                r = run_simulation(qqq, spy, mode, regime,
                                   entry_threshold=et, min_profit_bps=mp, verbose=False)
                flag = " ✓" if r["ev_bps"] > 0 and r["total_ret"] > 0 else ""
                print(f"  {et:.2f}   {mp:>3}bps  {r['trades']:>4}  "
                      f"{r['win_rate']*100:>4.0f}%  {r['ev_bps']:>+6.1f}bps  "
                      f"{r['total_bps']:>+7.0f}bps  {r['total_ret']:>+6.2f}%  "
                      f"{r['alpha']:>+7.2f}%  {r['sharpe']:>+6.2f}  "
                      f"{r['max_dd']:>+6.2f}%{flag}")
        print(f"{'='*88}")
        print("✓ = positive EV and positive total return")
    else:
        # Single run
        et  = _arg("--entry-threshold", ENTRY_THRESHOLD)
        mp  = _arg("--min-profit-bps",  MIN_PROFIT_BPS)
        vt  = "--vol-target" in args
        val = "--validate" in args
        run_simulation(qqq, spy, mode, regime, entry_threshold=et, min_profit_bps=mp,
                       vol_target_enabled=vt, validate=val)
