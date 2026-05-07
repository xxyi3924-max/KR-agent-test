# news_quant deploy notes

## Lightsail bring-up (first time)

```bash
ssh ubuntu@<lightsail-ip>
cd ~/KR-agent-test-main && git pull           # bring latest news_quant in

# Create the venv only once
python3 -m venv news_quant/.venv
news_quant/.venv/bin/pip install -r news_quant/requirements.txt

# Add env vars to ~/.profile (one time)
cat >> ~/.profile <<'EOF'
export ANTHROPIC_API_KEY='sk-ant-...'
export ALPACA_API_KEY='PK...'           # paper-trading key
export ALPACA_SECRET_KEY='...'
# Only when broker.type==kaigora (still dry-run by default):
# export KR_BROKER_URL='http://localhost:8084'
# export KR_BROKER_API_KEY='...'
EOF
source ~/.profile

# Bootstrap ledger capital (paper account size — Alpaca paper starts at $100k)
news_quant/.venv/bin/python -c "from news_quant import ledger; ledger.init_capital(100000.0)"
```

## Day-to-day operation

```bash
bash news_quant/deploy/lightsail_start.sh   # starts in screen 'news_quant'
screen -r news_quant                        # attach
# Ctrl-a then d to detach. Daemon keeps running.

# Stop cleanly (sends SIGTERM; daemon finishes current cycle then exits):
screen -S news_quant -X quit

# Logs:
tail -f news_quant/logs/daemon-*.log
```

The screen session is parallel to the existing `kr-agent` screen — they
do not conflict (no shared ports, separate ledgers and news stores).

## Switching broker

`news_quant/config.yaml` → `broker.type`:

| value     | behaviour                                                            |
|-----------|----------------------------------------------------------------------|
| `alpaca`  | Live calls against Alpaca paper-trading sandbox (default).           |
| `kaigora` | Wraps `kr_broker` but `dry_run: true` by default — logs only.        |

Flip `kaigora.dry_run` to `false` only after the forward-shadow gate
clears (≥30 cycles, mean net ≥ 20 bps, weekly Sharpe > 0.8).

## Health checks

```bash
news_quant/.venv/bin/python -m news_quant.ledger
news_quant/.venv/bin/python -m news_quant.shadow_report
```

## Halt and resume

```bash
# Halt manually (e.g. before market open if you don't want it running):
news_quant/.venv/bin/python -c "from news_quant import ledger; ledger.set_halt('manual')"

# Clear:
news_quant/.venv/bin/python -c "from news_quant import ledger; ledger.clear_halt()"
```
