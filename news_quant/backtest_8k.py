"""Phase 3 backtest: 8-K events → simulated QQQ↔name cycles.

For each scored event that passes the signal gate:
  1. At T = acceptance_dt + entry_lag_min, the strategy:
     a) sells the QQQ default position,
     b) buys ticker XYZ at next 5m bar open,
     c) holds with TP/SL or until horizon_min elapsed,
     d) sells XYZ,
     e) rebuys QQQ.
  2. Cycle PnL (bps) = (XYZ return during hold) − (QQQ return during hold) − friction.
  3. Excess return vs always-hold-QQQ is what we measure.

Stat gates (from plan):
  - n cycles ≥ 200
  - DSR p < 0.10
  - Bootstrap 95% CI lower bound > 0
  - Walk-forward 4-fold stability
  - Hit rate ≥ 0.55
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stat_validate import deflated_sharpe_ratio, stationary_block_bootstrap  # noqa: E402
from simulate_etf_monitor import fetch_bars_polygon  # noqa: E402

from news_quant.config_loader import load as load_config  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Cycle:
    event_idx: int
    ticker: str
    direction: float
    confidence: float
    magnitude_bps: float
    score: float
    acceptance_dt: pd.Timestamp
    entry_dt: pd.Timestamp
    exit_dt: pd.Timestamp
    name_ret_bps: float
    qqq_ret_bps: float
    friction_bps: float
    excess_ret_bps: float
    exit_reason: str  # "tp", "sl", "timeout"
    hit: bool


def _bars_df(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    bars = fetch_bars_polygon(ticker, start, end, api_key, multiplier=5, use_cache=True)
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def _simulate_cycle(
    name_bars: pd.DataFrame,
    qqq_bars: pd.DataFrame,
    acceptance_dt: pd.Timestamp,
    direction: float,  # +1 long the name, -1 short the name
    horizon_min: int,
    tp_bps: float,
    sl_bps: float,
    entry_lag_min: int,
    friction_bps: float,
    allow_short: bool = False,
) -> tuple[Cycle | None, str]:
    """Walk forward through 5m bars; return Cycle or None+reason."""
    if name_bars.empty or qqq_bars.empty:
        return None, "no_bars"
    entry_anchor = acceptance_dt + pd.Timedelta(minutes=entry_lag_min)
    after_name = name_bars[name_bars["ts"] >= entry_anchor]
    if after_name.empty:
        return None, "entry_after_data"
    entry_bar = after_name.iloc[0]
    entry_px = float(entry_bar["open"])
    entry_dt = entry_bar["ts"]

    # Walk bars until TP/SL or horizon
    horizon_end = entry_dt + pd.Timedelta(minutes=horizon_min)
    inside = after_name[(after_name["ts"] >= entry_dt) & (after_name["ts"] <= horizon_end)]
    if inside.empty:
        return None, "no_inside_bars"

    if direction == 0:
        return None, "no_direction"
    if direction < 0 and not allow_short:
        return None, "no_short_supported"

    is_long = direction > 0
    # TP is in the trade direction; SL is against it.
    if is_long:
        tp_px = entry_px * (1 + tp_bps / 1e4)
        sl_px = entry_px * (1 - sl_bps / 1e4)
    else:
        tp_px = entry_px * (1 - tp_bps / 1e4)  # short profits when price falls
        sl_px = entry_px * (1 + sl_bps / 1e4)

    exit_reason = "timeout"
    exit_px = float(inside.iloc[-1]["close"])
    exit_dt = inside.iloc[-1]["ts"]
    for _, b in inside.iterrows():
        hi, lo = float(b["high"]), float(b["low"])
        if is_long:
            # Conservative: SL checked before TP within a bar (worst case for long)
            if lo <= sl_px:
                exit_px = sl_px; exit_dt = b["ts"]; exit_reason = "sl"; break
            if hi >= tp_px:
                exit_px = tp_px; exit_dt = b["ts"]; exit_reason = "tp"; break
        else:
            # Short: SL is the upper bound (price rising); TP is the lower bound.
            # Conservative: SL checked before TP within a bar.
            if hi >= sl_px:
                exit_px = sl_px; exit_dt = b["ts"]; exit_reason = "sl"; break
            if lo <= tp_px:
                exit_px = tp_px; exit_dt = b["ts"]; exit_reason = "tp"; break

    raw_name_ret_bps = (exit_px / entry_px - 1.0) * 1e4
    # Trade-direction return: long gains on rise, short gains on fall.
    name_ret_bps = raw_name_ret_bps if is_long else -raw_name_ret_bps

    # QQQ baseline return over the same hold window
    qq_entry_after = qqq_bars[qqq_bars["ts"] >= entry_dt]
    qq_exit_after = qqq_bars[qqq_bars["ts"] >= exit_dt]
    if qq_entry_after.empty or qq_exit_after.empty:
        return None, "qqq_no_overlap"
    qqq_entry_px = float(qq_entry_after.iloc[0]["open"])
    qqq_exit_px = float(qq_exit_after.iloc[0]["open"])
    qqq_ret_bps = (qqq_exit_px / qqq_entry_px - 1.0) * 1e4

    # Cash-parking framing: we left QQQ to take the trade, so we forgo QQQ's return
    # while in the name. Net excess = trade-direction PnL on the name minus what
    # QQQ would have earned during the same window, minus friction.
    excess_bps = name_ret_bps - qqq_ret_bps - friction_bps
    cycle = Cycle(
        event_idx=-1,
        ticker="",
        direction=direction,
        confidence=0.0,
        magnitude_bps=0.0,
        score=0.0,
        acceptance_dt=acceptance_dt,
        entry_dt=entry_dt,
        exit_dt=exit_dt,
        name_ret_bps=name_ret_bps,
        qqq_ret_bps=qqq_ret_bps,
        friction_bps=friction_bps,
        excess_ret_bps=excess_bps,
        exit_reason=exit_reason,
        hit=excess_bps > 0,
    )
    return cycle, "ok"


def run_backtest(
    scored: pd.DataFrame,
    polygon_api_key: str,
    threshold: float,
    horizon_min: int,
    tp_bps: float,
    sl_bps: float,
    entry_lag_min: int,
    friction_bps: float,
    allow_short: bool = False,
    exclude_item_prefixes: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg_default_index = "QQQ"
    scored = scored.copy()
    scored["acceptance_dt_utc"] = pd.to_datetime(scored["acceptance_dt_utc"], utc=True)
    if exclude_item_prefixes:
        items = scored["items"].fillna("")
        mask = pd.Series(False, index=scored.index)
        for pref in exclude_item_prefixes:
            mask = mask | items.str.startswith(pref)
        before = len(scored)
        scored = scored[~mask].reset_index(drop=True)
        print(f"[backtest] excluded items {exclude_item_prefixes}: {before} → {len(scored)}")
    scored["score"] = (
        pd.to_numeric(scored.get("direction"), errors="coerce")
        * pd.to_numeric(scored.get("confidence"), errors="coerce")
        * pd.to_numeric(scored.get("magnitude_bps"), errors="coerce") / 1000.0
    )
    # Threshold gate is on |score| — direction sign is handled by long/short logic.
    triggered = scored[scored["score"].abs() >= threshold].copy()
    triggered = triggered[triggered["ticker"].notna() & (triggered["ticker"] != "")]
    print(f"[backtest] scored={len(scored)}  triggered={len(triggered)}  threshold(|score|)={threshold}  allow_short={allow_short}")
    if triggered.empty:
        return pd.DataFrame(), {"error": "no_triggers"}

    # Pre-fetch QQQ bars covering the full window
    win_start = triggered["acceptance_dt_utc"].min().date().isoformat()
    win_end = (triggered["acceptance_dt_utc"].max().date() + pd.Timedelta(days=2)).isoformat()
    print(f"[backtest] fetching QQQ bars {win_start} → {win_end}")
    qqq_bars = _bars_df(cfg_default_index, win_start, win_end, polygon_api_key)
    if qqq_bars.empty:
        return pd.DataFrame(), {"error": "qqq_bars_empty"}

    cycles: list[Cycle] = []
    skip_reasons: dict[str, int] = {}
    bars_cache: dict[str, pd.DataFrame] = {}
    tickers = sorted(triggered["ticker"].unique())
    for j, tkr in enumerate(tickers, 1):
        sub = triggered[triggered["ticker"] == tkr]
        d_min = sub["acceptance_dt_utc"].min().date().isoformat()
        d_max = (sub["acceptance_dt_utc"].max().date() + pd.Timedelta(days=2)).isoformat()
        # Polygon free tier: 5 calls/min. Sleep before any call we expect to miss cache.
        cache_root = ROOT / "data_cache"
        cache_hit = (cache_root / f"{tkr}_5m_{d_min}_{d_max}.csv").exists()
        if not cache_hit and j > 1:
            import time as _time
            _time.sleep(13)
        print(f"  [{j}/{len(tickers)}] bars for {tkr}  ({len(sub)} events){' [cache]' if cache_hit else ''}")
        bars_cache[tkr] = _bars_df(tkr, d_min, d_max, polygon_api_key)

    for idx, row in triggered.iterrows():
        tkr = row["ticker"]
        bars = bars_cache.get(tkr)
        if bars is None or bars.empty:
            skip_reasons["no_name_bars"] = skip_reasons.get("no_name_bars", 0) + 1
            continue
        cyc, reason = _simulate_cycle(
            bars,
            qqq_bars,
            row["acceptance_dt_utc"],
            float(row.get("direction") or 0),
            horizon_min=horizon_min,
            tp_bps=tp_bps,
            sl_bps=sl_bps,
            entry_lag_min=entry_lag_min,
            friction_bps=friction_bps,
            allow_short=allow_short,
        )
        if cyc is None:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        cyc.event_idx = int(idx)
        cyc.ticker = tkr
        cyc.confidence = float(row.get("confidence") or 0)
        cyc.magnitude_bps = float(row.get("magnitude_bps") or 0)
        cyc.score = float(row["score"])
        cycles.append(cyc)

    cdf = pd.DataFrame([asdict(c) for c in cycles])
    print(f"[backtest] cycles={len(cdf)}  skips={skip_reasons}")
    if cdf.empty:
        return cdf, {"error": "no_cycles", "skip_reasons": skip_reasons}

    excess_bps = cdf["excess_ret_bps"].to_numpy()
    excess_frac = excess_bps / 1e4

    # Trades-per-year-equivalent for annualisation: estimate from window
    span_days = (cdf["entry_dt"].max() - cdf["entry_dt"].min()).total_seconds() / 86400
    trades_per_year = len(cdf) / max(span_days / 365.25, 1e-9)
    dsr = deflated_sharpe_ratio(excess_frac, num_trials=1, freq=trades_per_year)

    # Bootstrap CI on per-cycle mean (i.i.d. assumption — events are independent
    # filings, not a time-series of one asset). For autocorr-aware CIs on Sharpe
    # / total-return / drawdown, we also call stationary_block_bootstrap below.
    rng = np.random.default_rng(42)
    n = len(excess_frac)
    boot_means = np.empty(2000, dtype=float)
    for i in range(2000):
        idx = rng.integers(0, n, n)
        boot_means[i] = excess_frac[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    try:
        block_ci = stationary_block_bootstrap(excess_frac, n_resamples=1000, mean_block=max(int(np.sqrt(n)), 5))
    except Exception as e:
        block_ci = {"error": str(e)}

    # Walk-forward 4-fold by event-time
    cdf_sorted = cdf.sort_values("entry_dt").reset_index(drop=True)
    fold_idx = np.array_split(np.arange(len(cdf_sorted)), 4)
    fold_means = [
        float(cdf_sorted.iloc[ix]["excess_ret_bps"].mean()) if len(ix) else 0.0
        for ix in fold_idx
    ]
    fold_pos = sum(1 for m in fold_means if m > 0)

    summary = {
        "n_cycles": int(len(cdf)),
        "trades_per_year": float(trades_per_year),
        "mean_excess_bps": float(excess_bps.mean()),
        "median_excess_bps": float(np.median(excess_bps)),
        "std_excess_bps": float(excess_bps.std(ddof=1)),
        "hit_rate": float((excess_bps > 0).mean()),
        "sharpe": float(dsr["sharpe"]),
        "dsr_z": float(dsr["deflated_sharpe"]),
        "dsr_pvalue": float(dsr["dsr_pvalue"]),
        "boot_ci_lo_bps": float(ci_lo * 1e4),
        "boot_ci_hi_bps": float(ci_hi * 1e4),
        "block_bootstrap": block_ci,
        "fold_means_bps": fold_means,
        "fold_positive_count": fold_pos,
        "skip_reasons": skip_reasons,
        "exit_reason_counts": cdf["exit_reason"].value_counts().to_dict(),
    }
    return cdf, summary


def evaluate_gates(summary: dict) -> tuple[list[tuple[str, bool, str]], bool]:
    gates = [
        ("≥200 cycles",        summary["n_cycles"] >= 200,                f"got {summary['n_cycles']}"),
        ("DSR p<0.10",         summary["dsr_pvalue"] < 0.10,              f"p={summary['dsr_pvalue']:.4f}"),
        ("Bootstrap CI > 0",   summary["boot_ci_lo_bps"] > 0,             f"lo={summary['boot_ci_lo_bps']:.1f}bps"),
        ("Hit rate ≥ 55%",     summary["hit_rate"] >= 0.55,               f"hit={summary['hit_rate']:.3f}"),
        ("4 of 4 folds > 0",   summary["fold_positive_count"] == 4,       f"{summary['fold_positive_count']}/4 positive"),
        ("Mean excess > 0bps", summary["mean_excess_bps"] > 0,            f"{summary['mean_excess_bps']:.1f}bps"),
    ]
    overall = all(passed for _, passed, _ in gates)
    return gates, overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--polygon-key", default=None)
    p.add_argument("--threshold", type=float, default=0.0,
                   help="|score| gate; default 0.0 trades on any directional call")
    p.add_argument("--friction-bps", type=float, default=20.0)
    p.add_argument("--entry-lag-min", type=int, default=15)
    p.add_argument("--horizon-min", type=int, default=120,
                   help="max hold time before timeout exit; default 120m")
    p.add_argument("--tp-bps", type=float, default=None, help="overrides config strategy.tp_bps")
    p.add_argument("--sl-bps", type=float, default=None, help="overrides config strategy.sl_bps")
    p.add_argument("--allow-short", action="store_true",
                   help="trade direction<0 events as shorts (default off; negatives are skipped)")
    p.add_argument("--exclude-items", default="2.02",
                   help="comma-separated item-code prefixes to drop (default '2.02' = earnings)")
    args = p.parse_args()

    cfg = load_config()
    api_key = args.polygon_key or os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        sys.exit("Polygon key required: --polygon-key or POLYGON_API_KEY env")

    threshold = args.threshold
    tp_bps = args.tp_bps if args.tp_bps is not None else cfg["strategy"]["tp_bps"]
    sl_bps = args.sl_bps if args.sl_bps is not None else cfg["strategy"]["sl_bps"]
    horizon_min = args.horizon_min
    excl = [s.strip() for s in args.exclude_items.split(",") if s.strip()] or None

    scored = pd.read_parquet(args.scored)
    print(f"loaded {len(scored)} scored events")

    cdf, summary = run_backtest(
        scored,
        api_key,
        threshold=threshold,
        horizon_min=horizon_min,
        tp_bps=tp_bps,
        sl_bps=sl_bps,
        entry_lag_min=args.entry_lag_min,
        friction_bps=args.friction_bps,
        allow_short=args.allow_short,
        exclude_item_prefixes=excl,
    )
    if cdf.empty:
        print(f"no cycles produced. summary={summary}")
        return

    out_path = Path(args.out) if args.out else DATA_DIR / "backtest_cycles.parquet"
    cdf.to_parquet(out_path, index=False)
    print(f"\nWrote {len(cdf)} cycles → {out_path}")

    print("\n=== Backtest summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    gates, overall = evaluate_gates(summary)
    print("\n=== Stat gates ===")
    for name, passed, detail in gates:
        mark = "✅" if passed else "❌"
        print(f"  {mark} {name}: {detail}")
    print(f"\nOVERALL: {'✅ PASS — proceed to Phase 4' if overall else '❌ FAIL — halt'}")


if __name__ == "__main__":
    main()
