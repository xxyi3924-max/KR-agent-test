"""Phase 2 redux gate: per-cycle net expectation under explicit friction.

Tests the pre-registered hypothesis from PHASE2_REDUX_HYPOTHESIS.md:

    mean( sign(signed_score) * fwd_ret_{H}m_bps )  >  friction_bps

with bootstrap 95% CI lo > 0 and one-sided p < 0.05.

`signed_score` = direction * confidence * magnitude_bps. Rows with
`signed_score == 0` (LLM said "neutral") are dropped — we only trade
when there is a directional call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _signed(df: pd.DataFrame) -> pd.Series:
    d = pd.to_numeric(df.get("direction"), errors="coerce")
    c = pd.to_numeric(df.get("confidence"), errors="coerce")
    m = pd.to_numeric(df.get("magnitude_bps"), errors="coerce")
    return d * c * m


def evaluate(
    df: pd.DataFrame,
    horizon_min: int,
    friction_bps: float,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    fwd_col = f"fwd_ret_{horizon_min}m_bps"
    if fwd_col not in df.columns:
        raise SystemExit(f"missing column {fwd_col}")

    signed = _signed(df)
    fwd = pd.to_numeric(df[fwd_col], errors="coerce")
    mask = signed.notna() & fwd.notna() & (signed != 0)
    s = signed[mask].to_numpy()
    f = fwd[mask].to_numpy()
    n = len(s)
    if n < 20:
        return {"n": n, "error": "insufficient_data"}

    direction = np.sign(s)
    realized_gross = direction * f
    realized_net = realized_gross - friction_bps

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot_means[i] = realized_net[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    p_one_sided = float((boot_means <= 0).mean())

    win = float((realized_gross > friction_bps).mean())
    win_zero_friction = float((realized_gross > 0).mean())

    return {
        "n": n,
        "horizon_min": horizon_min,
        "friction_bps": friction_bps,
        "mean_gross_bps": float(realized_gross.mean()),
        "mean_net_bps": float(realized_net.mean()),
        "median_gross_bps": float(np.median(realized_gross)),
        "std_per_cycle_bps": float(realized_gross.std(ddof=1)),
        "ci95_net_lo": float(ci_lo),
        "ci95_net_hi": float(ci_hi),
        "p_one_sided_net_le_0": p_one_sided,
        "hit_rate_above_friction": win,
        "hit_rate_above_zero": win_zero_friction,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True, help="...fwd.parquet from forward_returns.py")
    p.add_argument("--horizon", type=int, default=120, help="forward-return horizon in minutes")
    p.add_argument("--friction-bps", type=float, default=20.0)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--exclude-items", default="2.02",
                   help="comma-separated item-code prefixes to drop (default 2.02 = earnings)")
    args = p.parse_args()

    df = pd.read_parquet(args.scored)
    print(f"loaded {len(df)} scored events from {args.scored}")

    excl = [s.strip() for s in args.exclude_items.split(",") if s.strip()]
    if excl:
        items = df["items"].fillna("")
        mask = pd.Series(False, index=df.index)
        for pref in excl:
            mask = mask | items.str.startswith(pref)
        before = len(df)
        df = df[~mask].reset_index(drop=True)
        print(f"  excluded items {excl}: {before} → {len(df)}")

    res = evaluate(df, args.horizon, args.friction_bps, n_bootstrap=args.bootstrap)
    print()
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:+.4f}")
        else:
            print(f"  {k:30s} {v}")

    if "error" in res:
        raise SystemExit(2)

    passed = (
        res["mean_net_bps"] > 0
        and res["ci95_net_lo"] > 0
        and res["p_one_sided_net_le_0"] < 0.05
    )
    print()
    if passed:
        print("PASS: net mean > 0, CI lo > 0, p < 0.05. Proceed to Gate B.")
    else:
        print("FAIL: pre-registered gate not cleared. Halt per PHASE2_REDUX_HYPOTHESIS.md.")


if __name__ == "__main__":
    main()
