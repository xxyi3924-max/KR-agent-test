"""Information Coefficient analysis: signal vs forward returns.

Phase 2 kill-switch.  Computes:
  - Spearman rank correlation (signal × confidence vs forward bps)
  - Pearson correlation
  - Hit rate when |signed_score| > threshold
  - Bootstrap 95% CI on Spearman IC

If the lower bound of the bootstrap CI is below 0, the signal has no
demonstrable information content and the project should halt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _signed_score(df: pd.DataFrame) -> pd.Series:
    """Signed score = direction × confidence × magnitude_bps.

    This is the per-event expected return contribution if the LLM
    score was perfectly calibrated.  Positive → expect price up;
    negative → expect price down.
    """
    direction = pd.to_numeric(df.get("direction"), errors="coerce")
    confidence = pd.to_numeric(df.get("confidence"), errors="coerce")
    magnitude = pd.to_numeric(df.get("magnitude_bps"), errors="coerce")
    return direction * confidence * magnitude


def compute_ic(
    df: pd.DataFrame, fwd_col: str, n_bootstrap: int = 1000, seed: int = 42
) -> dict:
    score = _signed_score(df)
    fwd = pd.to_numeric(df.get(fwd_col), errors="coerce")
    mask = score.notna() & fwd.notna()
    s = score[mask].to_numpy()
    f = fwd[mask].to_numpy()
    n = len(s)
    if n < 20:
        return {"n": n, "error": "insufficient_data"}

    spearman = stats.spearmanr(s, f)
    pearson = stats.pearsonr(s, f)

    # Bootstrap on Spearman IC
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            boots.append(stats.spearmanr(s[idx], f[idx]).correlation)
        except Exception:
            continue
    boots = np.array([b for b in boots if not np.isnan(b)])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan)

    # Hit rate: when score is non-zero, does sign(score) match sign(fwd)?
    nz = s != 0
    if nz.sum() > 0:
        hit_rate = float((np.sign(s[nz]) == np.sign(f[nz])).mean())
    else:
        hit_rate = float("nan")

    # High-confidence subset
    hi_mask = (np.abs(s) >= np.quantile(np.abs(s), 0.7))
    hi_hit = (
        float((np.sign(s[hi_mask]) == np.sign(f[hi_mask])).mean())
        if hi_mask.sum() else float("nan")
    )

    return {
        "n": int(n),
        "fwd_col": fwd_col,
        "spearman_ic": float(spearman.correlation),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "hit_rate_all": hit_rate,
        "hit_rate_top30pct": hi_hit,
        "ic_ci95_lo": float(ci_lo),
        "ic_ci95_hi": float(ci_hi),
        "fwd_mean_bps": float(np.nanmean(f)),
        "fwd_std_bps": float(np.nanstd(f, ddof=1)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True, help="scored.fwd.parquet (output of forward_returns.py)")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    df = pd.read_parquet(args.scored)
    print(f"loaded {len(df)} events from {args.scored}")

    horizons = [c for c in df.columns if c.startswith("fwd_ret_") and c.endswith("_bps")]
    if not horizons:
        raise SystemExit("no fwd_ret_*_bps columns — run forward_returns.py first")

    print("\nIC report (signed_score = direction × confidence × magnitude_bps):\n")
    rows = []
    for col in horizons:
        r = compute_ic(df, col, n_bootstrap=args.bootstrap)
        rows.append(r)
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))

    # Plain-English verdict on the strongest horizon (highest |IC|, lowest p)
    rep["abs_ic"] = rep["spearman_ic"].abs()
    best = rep.sort_values("abs_ic", ascending=False).iloc[0]
    print(f"\nBEST horizon: {best['fwd_col']}")
    print(f"  Spearman IC = {best['spearman_ic']:+.4f}  p = {best['spearman_p']:.4f}")
    print(f"  95% CI       = [{best['ic_ci95_lo']:+.4f}, {best['ic_ci95_hi']:+.4f}]")
    print(f"  Hit rate (all)        = {best['hit_rate_all']:.3f}")
    print(f"  Hit rate (top 30% conf) = {best['hit_rate_top30pct']:.3f}")

    if best["ic_ci95_lo"] > 0 and best["spearman_p"] < 0.10:
        print("\n✅ PASS: bootstrap CI excludes 0 and p<0.10. Phase 2 gate cleared.")
    elif best["spearman_p"] < 0.10:
        print("\n⚠️  WEAK: p<0.10 but CI crosses 0. Borderline; consider larger sample.")
    else:
        print("\n❌ FAIL: signal indistinguishable from 0. Halt project per plan.")


if __name__ == "__main__":
    main()
