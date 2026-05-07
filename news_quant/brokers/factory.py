"""Construct the broker adapter selected by news_quant config.

Reads ``broker:`` block from the news_quant config:

    broker:
      type: alpaca | kaigora
      paper: true
      alpaca:
        api_key_env: ALPACA_API_KEY
        secret_env:  ALPACA_SECRET_KEY
        base_url:    https://paper-api.alpaca.markets   # informational only
      kaigora:
        api_url_env: KR_BROKER_URL
        api_key_env: KR_BROKER_API_KEY
        dry_run:     true        # K1 — never actually trade Kaigora until cleared
"""

from __future__ import annotations

import logging
import os

from news_quant.brokers.base import BrokerConnector

logger = logging.getLogger("news_quant.brokers.factory")


def _envget(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        raise RuntimeError(f"required env var ${name} is empty")
    return v


def create_broker(cfg: dict) -> BrokerConnector:
    bcfg = cfg.get("broker") or {}
    btype = (bcfg.get("type") or "alpaca").lower()
    paper = bool(bcfg.get("paper", True))

    if btype == "alpaca":
        from news_quant.brokers.alpaca import AlpacaBroker
        ac = bcfg.get("alpaca") or {}
        api_key = _envget(ac.get("api_key_env") or "ALPACA_API_KEY")
        secret = _envget(ac.get("secret_env") or "ALPACA_SECRET_KEY")
        return AlpacaBroker(api_key=api_key, secret_key=secret, paper=paper)

    if btype == "kaigora":
        from news_quant.brokers.kaigora import KaigoraBroker
        kc = bcfg.get("kaigora") or {}
        api_url = _envget(kc.get("api_url_env") or "KR_BROKER_URL")
        api_key = _envget(kc.get("api_key_env") or "KR_BROKER_API_KEY")
        dry_run = bool(kc.get("dry_run", True))
        if not dry_run:
            logger.warning(
                "KaigoraBroker created with dry_run=False — real-money orders will fire. "
                "Confirm forward-shadow gate cleared."
            )
        return KaigoraBroker(api_url=api_url, api_key=api_key, dry_run=dry_run)

    raise RuntimeError(f"unknown broker.type={btype!r}")
