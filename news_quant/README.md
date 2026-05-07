# news_quant

LLM news-signal quant subproject. Standalone from `etf_monitor.py` and `decision_engine.py`.

## Scope

Default position: hold QQQ (or SPY) as cash parking. When LLM-scored news produces a high-conviction signal on a single name, liquidate the index, rotate into the name, exit at TP/SL, return to the index.

## Friction reality

20bps round-trip commission per leg × 4 legs per cycle ≈ **60bps total cycle friction**. Single-name move must clear this net to break even. See `/Users/xiao/.claude/plans/this-is-what-hapened-harmonic-beaver.md` for full survival math.

## Isolation

- Own `config.yaml` (this folder) — root `config.yaml` untouched.
- Own capital ledger (`data/ledger.sqlite`).
- Own LLM cost meter (`data/llm_costs.sqlite`).
- Own drawdown halt — auto-stops trading if equity drops > `budget.hard_stop_drawdown_pct` from high-water.
- Reuses by import only: `kr_broker.py`, `stat_validate.py`, `simulate_etf_monitor.fetch_bars_polygon`.

## Phase status

- [x] Phase 0: source survey
- [ ] Phase 1: EDGAR 8-K pipeline
- [ ] Phase 2: LLM triage + scoring (IC validation)
- [ ] Phase 3: 8-K-only backtest with stat gates
- [ ] Phase 4: Tier-2/3 web crawler
- [ ] Phase 5: 3-month forward shadow
- [ ] Phase 6: live capital (after gates pass)

## Validation gates

Before live capital, must pass all 10 gates in the plan. Non-negotiable.
