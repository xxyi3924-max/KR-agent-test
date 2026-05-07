# Phase 2 Redux — Pre-Registration

**Written before pulling the new sample.** Once the new sample is scored, the result is reported in `PHASE2_REDUX_RESULT.md` and this file is not edited.

## Why we are reopening

`PHASE2_RESULT.md` halted under a 60 bps round-trip friction assumption inherited from `etf_monitor.py`. The actual broker friction is **20 bps round-trip**. At 20 bps, the existing N=86 sample's per-cycle net expectation flips positive on multiple slices (see plan file `it-is-20bps-round-encapsulated-gray.md` for the full table). The post-hoc positive numbers are not enough to deploy on, so we re-validate with a single pre-registered hypothesis on a fresh sample.

## Hypothesis (single, pre-registered)

> Issuer-equity 8-K filings, **excluding Item 2.02 (earnings)**, scored by `claude-haiku-4-5`, with `|signed_score| > 0` (where `signed_score = direction × confidence × magnitude_bps`), traded long/short in `sign(signed_score)` direction, exited at the first 5-min Polygon bar with timestamp `>= entry_ts + 60 minutes`, produce a mean per-cycle return greater than the **20 bps** round-trip friction, with bootstrap 95% CI lower bound > 0 and one-sided p < 0.05 against the null `mean_net_bps <= 0`.

`mean_net_bps` is computed as `mean(sign(signed) × fwd_ret_60m_bps) − 20`. We test both the gross mean and the net mean; the gate is on **net**.

## Sample plan

- Universe: `data/edgar_8k_2024-04-01_2026-04-30.parquet` (13,034 events, S&P 500 issuers).
- Filter: drop any row whose `items` field starts with `2.02`.
- Disjointness: drop any `accession` already present in `data/scored_sample100_seed42.fwd.parquet`.
- Draw: random sample with `seed=43`, target N=300 *after* the disjointness and earnings filters.
- Score with `claude-haiku-4-5` using the existing `news_quant/news/score.py:SCORING_SYSTEM` prompt. No prompt edits.
- Forward returns: `news_quant/news/forward_returns.py` at horizons `[60, 120, 240, 1440]`. Decision is made on the **60m** column only; others are recorded for diagnostics but not used to override the gate.
- Polygon bar source: same as Phase 2 (`simulate_etf_monitor.fetch_bars_polygon`, 5-minute multiplier, `use_cache=True`).

## Decision rule

After scoring + forward returns are attached, run `news_quant/analysis/net_expectation.py` with `--horizon 60 --friction-bps 20`. The script reports:
- `n` (events with both a score and a `fwd_ret_60m_bps`),
- `mean_gross_bps`, `mean_net_bps`,
- bootstrap 95% CI on the per-cycle mean (10,000 resamples, seed=42),
- one-sided p-value (`P(boot_mean <= 0)`).

**Pass:** `mean_net_bps > 0` AND `ci_lo > 0` AND `p < 0.05`.
**Fail:** anything else, including borderline (`p ∈ [0.05, 0.10]`). Borderline = halt; expanding sample after seeing the result is the post-hoc trap that killed Phase 1.

If the gate passes, proceed to Gate B (Phase 3 backtest on union seed=42 ∪ seed=43).
If the gate fails, write `PHASE2_REDUX_RESULT.md` with the numbers and halt — no slice-hunting, no horizon-shopping.

## What is *not* allowed under this pre-registration

- Looking at the new data before this file is committed.
- Picking a different horizon if 60m fails.
- Slicing on item codes (8.01-only, 1.01-only, etc.) to rescue a fail.
- Lowering the friction below 20 bps to rescue a fail.
- Inverting the score direction.
- Filtering on confidence quantiles.

These are explicitly the moves that make positive results meaningless. If the strategy works, it works on the headline 60m non-earnings cut.

## Cost budget

- Haiku 4.5: ~$0.001/filing × 300 ≈ **$0.30 LLM** (well under the $10/day cap in `cost_meter.py`).
- Polygon free tier: ~50–80 unique tickers × 13s spacing ≈ **15 min wall-clock** for forward returns.

## Files involved

- Reuse: `news_quant/news/edgar_8k.py`, `news_quant/news/score.py`, `news_quant/news/forward_returns.py`, `news_quant/cost_meter.py`.
- Edit: `news_quant/news/score.py` — add `--exclude-items` and `--exclude-accessions-from` CLI flags.
- New: `news_quant/analysis/net_expectation.py` — runs the decision rule above.
- Output: `news_quant/data/scored_sample300_seed43.parquet`, `…fwd.parquet`.
- Result doc (created after the run): `news_quant/PHASE2_REDUX_RESULT.md`.
