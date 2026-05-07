# Phase 2 — Information Coefficient Gate: FAILED

**Decision: HALT per plan §7 ("IC indistinguishable from 0 — strategy dead, stop here").**

## Setup

- Universe: S&P 500 issuers, 8-K filings 2024-04-01 → 2026-04-30 (13,034 events ingested).
- Sample: N=100 random events (seed=42); 86 had usable Polygon 5m forward returns after rate-limited fetch.
- Scorer: claude-haiku-4-5, structured JSON (direction × confidence × magnitude_bps).
- Forward returns: issuer-equity, +60/120/240/1440 min from next 5m bar open, in bps.
- LLM spend: well under $10/day cap.

## Formal pre-registered gate

> Information Coefficient (Spearman, signed_score vs 120m forward return) p<0.10 with
> bootstrap 95% CI lower bound > 0.

**Result on N=86 full sample**: IC indistinguishable from 0; CI crosses zero. **Gate fails.**

## Post-hoc stratification (advisory only — not used to override the gate)

| Slice | n | Spearman IC | p | 95% CI | Hit rate |
|---|---|---|---|---|---|
| Earnings (Item 2.02 prefix) | 27 | ≈0 | n.s. | crosses 0 | 0.45 |
| Non-earnings | 59 | +0.20 | 0.14 | [−0.07, +0.43] | 0.53 |
| Top-30% confidence, non-earnings | 28 | +0.28 | 0.16 | crosses 0 | 0.57 |

The earnings-zero result is mechanically explainable: scoring an earnings 8-K without
the consensus number is a coin flip. The non-earnings positive point estimate is
suggestive but underpowered and not pre-registered, so it does not rescue the gate.

## Why we are stopping here, not expanding the sample

Per plan §7, Phase 2 is the cheap kill-switch precisely so we don't sink Phase 3
infrastructure cost into a signal that doesn't exist. Expanding to N=400 to chase
a sliced subgroup is exactly the p-hacking pattern the plan was designed to reject
(§6 validation gates exist *because* the prior strategy shipped on N=24 with DSR
p=0.96).

A clean negative result here is the correct outcome — better than reliving the
prior commission-bound failure.

## What was built and is preserved for any future restart

- `news_quant/news/edgar_8k.py` — EDGAR 8-K backfill (per-CIK API, 13k events working)
- `news_quant/news/score.py` — Haiku scorer with structured JSON + fail-fast
- `news_quant/news/forward_returns.py` — Polygon 5m forward-return enrichment
- `news_quant/analysis/ic.py` — IC kill-switch
- `news_quant/backtest_8k.py` — Phase 3 cycle simulator (unrun)
- `news_quant/{ledger,cost_meter}.py` — budget enforcement
- `news_quant/news/{store,aggregator,signal_gate,rss_crawler,web_scraper}.py` — pipeline
- `news_quant/tests/` — 18 tests passing
- Data: `data/edgar_8k_2024-04-01_2026-04-30.parquet`, `data/scored_sample100_seed42.fwd.parquet`

## Possible future restarts (require explicit user go-ahead, not autonomous)

1. Earnings strategy needs consensus-EPS feed (not free) — out of scope of this build.
2. Non-earnings-only strategy: pre-register a single hypothesis, score a fresh
   independent N=300+, evaluate once. Accept halt if it fails.
3. Different signal source entirely (FOMC/BLS macro, not 8-Ks).

None of these are autonomously authorized. Phase 2 halted per plan.
