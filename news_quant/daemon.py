"""news_quant live daemon.

Single-process loop that:
  1. Polls EDGAR + RSS sources every ``poll_seconds``.
  2. Scores each new event with Haiku (skips earnings 2.02, drops below
     credibility/score threshold).
  3. Routes scored events through ``signal_gate.evaluate``.
  4. Hands each fired signal to ``Executor.execute`` and persists results.
  5. Daily report, drawdown halt, graceful shutdown on SIGINT/SIGTERM.

Designed to run in a screen session on Lightsail. Logs to stdout (capture
with ``script -a logs/daemon.log`` or screen's logging).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

from anthropic import Anthropic

from news_quant import cost_meter, ledger
from news_quant.brokers import create_broker
from news_quant.brokers.base import BrokerConnector
from news_quant.config_loader import load as load_config
from news_quant.executor import Executor
from news_quant.news.edgar_live import EdgarLivePoller, NewsEvent
from news_quant.news.per_symbol import fetch_all_for_ticker, render_context_block
from news_quant.news.rss_live import RssLivePoller
from news_quant.news import signal_gate
from news_quant.news.score import fetch_primary_text, score_filing
from news_quant.news.universe import tradable_universe

logger = logging.getLogger("news_quant.daemon")

EXCLUDE_ITEM_PREFIXES = ("2.02",)  # earnings — coin flip without consensus EPS


class Daemon:
    def __init__(self, broker: BrokerConnector, cfg: dict):
        self.broker = broker
        self.cfg = cfg
        self.executor = Executor(broker, cfg)
        ua = cfg["http"]["user_agent"]
        edgar_rate = cfg["sources"]["edgar"]["rate_limit_per_sec"]
        self.edgar = EdgarLivePoller(ua, rate_limit_per_sec=edgar_rate, lookback_minutes=15)
        self.rss = RssLivePoller(ua, lookback_minutes=15)
        self.anthropic = Anthropic(api_key=cfg["llm"]["api_key"])
        self.score_model = cfg["llm"]["score_model"]
        self.threshold = float(cfg["strategy"]["signal_threshold"])
        self.max_cycles = int(cfg["strategy"]["max_cycles_per_day"])
        self._http = requests.Session()
        self._cycles_today = 0
        self._cycle_day: date | None = None
        self._stop = False
        # Tradable universe loaded once (S&P 500 ∪ Nasdaq-100 ≈ 516 names).
        self._universe = tradable_universe(ua)
        logger.info("daemon tradable universe size: %d", len(self._universe))

    # ── lifecycle ───────────────────────────────────────────────────────────
    def request_stop(self, *_):
        logger.info("stop requested — finishing current cycle")
        self._stop = True

    def _reset_daily_counters(self):
        today = datetime.now(timezone.utc).date()
        if self._cycle_day != today:
            self._cycle_day = today
            self._cycles_today = 0
            logger.info("date rolled to %s — cycle counter reset", today)

    # ── per-event handling ──────────────────────────────────────────────────
    def _drop_excluded(self, ev: NewsEvent) -> bool:
        if ev.source == "edgar_8k":
            for pref in EXCLUDE_ITEM_PREFIXES:
                if (ev.items or "").startswith(pref):
                    return True
        return False

    def _score_event(self, ev: NewsEvent) -> dict | None:
        """Run the LLM scorer; return scored dict or None on hard fail.

        Enriches the prompt with per-symbol headlines (Yahoo + Google) so
        Haiku sees the surrounding news context, not just the trigger event.
        """
        ua = self.cfg["http"]["user_agent"]
        snippet = ""
        if ev.primary_url:
            snippet = fetch_primary_text(ev.primary_url, ua, self._http, max_chars=8000)
        # Per-symbol enrichment: pull recent headlines for this ticker.
        try:
            recent = fetch_all_for_ticker(ev.ticker, lookback_hours=6)
        except Exception as e:
            logger.warning("per-symbol fetch failed for %s: %s", ev.ticker, e)
            recent = []
        extra_context = render_context_block(recent, max_items=8)
        if recent:
            logger.info(
                "enriched %s with %d corroborating headlines (yahoo+google)",
                ev.ticker, len(recent),
            )
        scored = score_filing(
            self.anthropic, self.score_model,
            issuer_name=ev.issuer_name,
            ticker=ev.ticker,
            acceptance_dt=ev.acceptance_dt_utc.isoformat(),
            items=ev.items,
            snippet=snippet,
            cost_tag=f"daemon-{ev.source}",
            extra_context=extra_context,
        )
        if not scored or "error" in scored:
            logger.info("score failed for %s: %s", ev.ticker, scored)
            return None
        return scored

    def handle_event(self, ev: NewsEvent) -> None:
        if self._drop_excluded(ev):
            logger.debug("drop excluded items=%s", ev.items)
            return
        if not ev.ticker:
            logger.debug("drop unticker'd event: %s", ev.headline)
            return
        # Pre-LLM universe filter: don't pay Haiku tokens or fetch per-symbol
        # context for tickers we wouldn't trade anyway (small-caps, OTC, etc.)
        if ev.ticker not in self._universe:
            logger.info("PRE-GATE drop %s — not in tradable universe", ev.ticker)
            return

        scored = self._score_event(ev)
        if scored is None:
            return

        direction = float(scored.get("direction") or 0)
        confidence = float(scored.get("confidence") or 0)
        magnitude = float(scored.get("magnitude_bps") or 0)

        # Open positions snapshot from broker (single source of truth).
        try:
            acct = self.broker.get_account_info()
            open_set = {p.ticker for p in acct.positions}
        except Exception as e:
            logger.warning("get_account_info failed in gate: %s", e)
            open_set = set()

        gate = signal_gate.evaluate(
            direction=direction,
            confidence=confidence,
            magnitude_bps=magnitude,
            credibility=ev.credibility,
            threshold=self.threshold,
            ticker=ev.ticker,
            tradable_universe=self._universe,
            open_positions=open_set,
            cycles_today=self._cycles_today,
            max_cycles_per_day=self.max_cycles,
        )
        if not gate.fire:
            logger.info(
                "GATE drop %s d=%+.2f c=%.2f m=%.0f cred=%.2f score=%+.4f reason=%s",
                ev.ticker, direction, confidence, magnitude, ev.credibility,
                gate.score, gate.reason,
            )
            return

        logger.info(
            "GATE fire %s side=%s score=%+.4f items=%s key=%s",
            ev.ticker, gate.side, gate.score, ev.items, scored.get("key_phrase", ""),
        )
        result = self.executor.execute(ev.ticker, gate.side, gate.score)
        self._cycles_today += 1
        logger.info(
            "CYCLE %s %s/%s entry=%.2f exit=%.2f realized=%+.1fbps reason=%s err=%s",
            result.cycle_id, result.ticker, result.side,
            result.entry_px, result.exit_px, result.realized_bps,
            result.exit_reason, result.error,
        )

    # ── main loop ───────────────────────────────────────────────────────────
    def run(self, poll_seconds: int = 60) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        logger.info(
            "daemon up: broker=%s paper=%s dry_run=%s threshold=%s poll=%ds",
            self.broker.name, self.broker.is_paper, self.broker.is_dry_run,
            self.threshold, poll_seconds,
        )

        while not self._stop:
            self._reset_daily_counters()
            t0 = time.time()
            try:
                # EDGAR bulk feed: 1-2s for 40 most-recent 8-Ks across all issuers.
                for ev in self.edgar.poll_bulk():
                    if self._stop:
                        break
                    self.handle_event(ev)
                for ev in self.rss.poll_once():
                    if self._stop:
                        break
                    self.handle_event(ev)
            except Exception as e:
                logger.exception("poll cycle error: %s", e)
            elapsed = time.time() - t0
            if not self._stop:
                sleep_for = max(0, poll_seconds - elapsed)
                logger.debug("cycle elapsed=%.1fs sleep=%.1fs", elapsed, sleep_for)
                # Cooperative sleep so SIGINT/SIGTERM are responsive.
                slept = 0.0
                while slept < sleep_for and not self._stop:
                    time.sleep(min(2.0, sleep_for - slept))
                    slept += 2.0

        logger.info("daemon stopped cleanly")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    cfg = load_config()
    broker = create_broker(cfg)
    Daemon(broker, cfg).run(poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
