# Agent Trader — TDash v9 × KR Broker Integration

## Overview
An AI agent that bridges Trading Dashboard v9 signals with KR Broker execution.

## Architecture
```
TDash v9 (data/) ──▶ Agent Brain ──▶ KR Broker (Agora API)
       ▲                    │
       │                    ▼
       └──────────── QQQ/S&P500 ETF (free cash)
```

## Components
- `agent.py` — Main agent loop
- `tdash_connector.py` — Reads signals from TDash data files
- `kr_broker.py` — KR Broker Agora API client
- `mcda_engine.py` — Momentum-based MCDA for liquidation priority
- `decision_engine.py` — Smart order routing & execution logic
- `scheduler.py` — Runs agent multiple times per day

## Configuration
Edit `config.yaml` with your paths and preferences.

## Running
```bash
python agent.py
```

## Smart Order Logic
| Scenario | Action |
|----------|--------|
| TDash entry > current price | Market order (better entry!) |
| TDash entry < current price | Limit order at TDash price |
| Gap > 2% | Log warning, proceed with market |
| No fill in 1 hour | Escalate to market order |

## MCDA Criteria
- Price momentum (20-day)
- RSI
- Volume trend
- Fee-aware: only sell if gain > 0.1%
