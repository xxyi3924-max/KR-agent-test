"""Live RSS poller for BusinessWire + PRNewswire (single-name press wires).

Yields ``NewsEvent`` objects for any new entry that:
  - has a publish timestamp within the lookback window
  - resolves to a tradable ticker via ``ticker_extract.extract_ticker``
  - is not already in ``news_store.sqlite``

We deliberately drop entries we cannot ticker-resolve. Acting on
unresolved press-release headlines is a path to bad fills.

Cadence: BW/PRN rotate hundreds of releases per hour; a 60-second poll is
the right cadence and well within their fair-use limits.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

import feedparser

from news_quant.config_loader import load as load_config
from news_quant.news import store
from news_quant.news.edgar_live import NewsEvent
from news_quant.news.rss_crawler import _parse_published
from news_quant.news.ticker_extract import extract_ticker

logger = logging.getLogger("news_quant.news.rss_live")

# Subset of config.yaml `sources.rss` we care about for single-name flow.
# BusinessWire was here — dropped 2026-05-07: BW deactivated their public feed.
TARGET_SOURCES = ("prnewswire", "globenewswire", "cnbc_top")


class RssLivePoller:
    def __init__(self, user_agent: str, lookback_minutes: int = 30):
        cfg = load_config()
        all_rss = cfg["sources"]["rss"]
        self._sources: dict[str, str] = {k: all_rss[k] for k in TARGET_SOURCES if k in all_rss}
        self._credibility = cfg["sources"]["credibility"]
        self.user_agent = user_agent
        self.lookback = timedelta(minutes=lookback_minutes)
        if not self._sources:
            logger.warning("RssLivePoller: no target sources configured (looked for %s)", TARGET_SOURCES)

    def poll_once(self) -> Iterator[NewsEvent]:
        now = datetime.now(timezone.utc)
        cutoff = now - self.lookback
        seen = emitted = unresolved = 0
        for source, url in self._sources.items():
            cred = float(self._credibility.get(source, 0.5))
            try:
                feed = feedparser.parse(url, request_headers={"User-Agent": self.user_agent})
            except Exception as e:
                logger.warning("rss fetch %s failed: %s", source, e)
                continue
            for entry in feed.entries:
                ts_pub = _parse_published(entry)
                if ts_pub is None or ts_pub < cutoff:
                    continue
                seen += 1
                headline = (getattr(entry, "title", "") or "").strip()
                summary = getattr(entry, "summary", "") or ""
                link = getattr(entry, "link", "")
                ticker = extract_ticker(headline) or extract_ticker(summary)
                if not ticker:
                    unresolved += 1
                    continue
                # Persist (dedup); only emit if newly inserted.
                is_new = store.upsert(
                    headline=headline, url=link, source=source,
                    credibility=cred, ts_publish=ts_pub,
                    tickers=[ticker], body=summary,
                )
                if not is_new:
                    continue
                emitted += 1
                yield NewsEvent(
                    source=source,
                    ticker=ticker,
                    issuer_name="",          # press wire doesn't give a normalized name
                    acceptance_dt_utc=ts_pub,
                    headline=headline,
                    primary_url=link,
                    accession_or_id=link,    # use URL as dedup key for non-EDGAR
                    items="",
                    credibility=cred,
                )
        logger.info(
            "RssLivePoller poll_once: seen=%d emitted=%d unresolved=%d",
            seen, emitted, unresolved,
        )


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--lookback-min", type=int, default=120)
    args = p.parse_args()
    cfg = load_config()
    ua = cfg["http"]["user_agent"]
    poller = RssLivePoller(ua, lookback_minutes=args.lookback_min)
    n = 0
    for ev in poller.poll_once():
        n += 1
        print(f"[{ev.acceptance_dt_utc.isoformat()}] {ev.source:>13} {ev.ticker:>6}  {ev.headline}")
    print(f"\ntotal new events: {n}")


if __name__ == "__main__":
    main()
