#!/usr/bin/env bash
# Start news_quant daemon in a screen session on Lightsail.
#
# Usage (on Lightsail):
#   cd ~/KR-agent-test-main
#   bash news_quant/deploy/lightsail_start.sh
#
# Sister to the kr-agent screen session that already runs on this box.
# Expects these env vars in ~/.profile or ~/.bashrc:
#   ANTHROPIC_API_KEY
#   ALPACA_API_KEY
#   ALPACA_SECRET_KEY
#   POLYGON_API_KEY            (optional — only used by backtest scripts)
#   KR_BROKER_URL              (only when broker.type=kaigora)
#   KR_BROKER_API_KEY          (only when broker.type=kaigora)
#
# The screen name is "news_quant". To attach:  screen -r news_quant
# To detach inside the session:  Ctrl-a then d
# To stop:  screen -S news_quant -X quit  (sends SIGTERM, daemon exits cleanly)

set -euo pipefail

SCREEN_NAME=news_quant
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="${REPO_DIR}/news_quant/.venv/bin/python"
LOG_DIR="${REPO_DIR}/news_quant/logs"
LOG_FILE="${LOG_DIR}/daemon-$(date -u +%Y%m%d-%H%M%S).log"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "venv python not found at ${VENV_PY}" >&2
  echo "first-time setup:  python3 -m venv news_quant/.venv && news_quant/.venv/bin/pip install -r news_quant/requirements.txt" >&2
  exit 1
fi

# Required env vars (broker-specific subset enforced inside the daemon).
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is unset}"
case "$(grep -E '^[[:space:]]*type:' "${REPO_DIR}/news_quant/config.yaml" | awk '{print $2}')" in
  alpaca)
    : "${ALPACA_API_KEY:?ALPACA_API_KEY is unset}"
    : "${ALPACA_SECRET_KEY:?ALPACA_SECRET_KEY is unset}"
    ;;
  kaigora)
    : "${KR_BROKER_URL:?KR_BROKER_URL is unset}"
    : "${KR_BROKER_API_KEY:?KR_BROKER_API_KEY is unset}"
    ;;
esac

if screen -ls | grep -q "\.${SCREEN_NAME}\b"; then
  echo "screen session '${SCREEN_NAME}' already running. Attach with:  screen -r ${SCREEN_NAME}"
  exit 0
fi

cd "${REPO_DIR}"
echo "starting news_quant daemon in screen '${SCREEN_NAME}', logging to ${LOG_FILE}"

screen -S "${SCREEN_NAME}" -dm bash -c \
  "cd '${REPO_DIR}' && '${VENV_PY}' -u -m news_quant.daemon --poll-seconds 60 --log-level INFO 2>&1 | tee -a '${LOG_FILE}'"

sleep 1
screen -ls | grep "${SCREEN_NAME}" || { echo "screen launch failed" >&2; exit 1; }
echo "ok — attach with:  screen -r ${SCREEN_NAME}"
echo "tail logs with:    tail -f ${LOG_FILE}"
