"""Per-symbol news fetchers — Yahoo Finance + Google News RSS.

Used by the daemon to enrich a broad-feed event with corroborating
headlines from major aggregators. Both endpoints are free, public RSS;
no API keys, no scraping, no anti-bot circumvention.

Pattern: when a broad-feed event resolves to a tradable ticker, the
daemon calls ``fetch_yahoo(ticker)`` + ``fetch_google_news(ticker)`` and
appends the recent headlines to the LLM prompt as additional context.
The LLM scores the *combined* picture; the per-symbol headlines do not
themselves trigger trades.

Why not direct Bloomberg / WSJ / FT?
  - Bloomberg.com: Cloudflare anti-bot blocks naive fetchers; no free RSS.
  - WSJ / FT: paywalled HTML; no public RSS for headlines.
Yahoo and Google News index Bloomberg / WSJ / FT stories so we still see
the same coverage via these aggregators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

import feedparser

from news_quant.config_loader import load as load_config
from news_quant.news.rss_crawler import _parse_published

logger = logging.getLogger("news_quant.news.per_symbol")


@dataclass
class Headline:
    source: str
    ticker: str
    ts_publish: datetime
    headline: str
    url: str
    summary: str


def _fetch_rss_template(
    template_url: str, ticker: str, source: str, user_agent: str,
    lookback_hours: int = 6, max_items: int = 8,
) -> List[Headline]:
    url = template_url.format(ticker=ticker.upper())
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": user_agent})
    except Exception as e:
        logger.warning("%s fetch %s failed: %s", source, ticker, e)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    out: List[Headline] = []
    for entry in feed.entries[: max_items * 2]:  # over-pull, cutoff filters
        ts_pub = _parse_published(entry)
        if ts_pub is None or ts_pub < cutoff:
            continue
        out.append(
            Headline(
                source=source,
                ticker=ticker.upper(),
                ts_publish=ts_pub,
                headline=(getattr(entry, "title", "") or "").strip(),
                url=getattr(entry, "link", "") or "",
                summary=(getattr(entry, "summary", "") or "").strip(),
            )
        )
        if len(out) >= max_items:
            break
    return out


def fetch_yahoo(
    ticker: str, user_agent: str = None, lookback_hours: int = 6,
) -> List[Headline]:
    cfg = load_config()
    template = cfg["sources"]["per_symbol"]["yahoo_finance"]
    ua = user_agent or cfg["http"]["user_agent"]
    return _fetch_rss_template(template, ticker, "yahoo_finance", ua, lookback_hours)


def fetch_google_news(
    ticker: str, user_agent: str = None, lookback_hours: int = 6,
) -> List[Headline]:
    cfg = load_config()
    template = cfg["sources"]["per_symbol"]["google_news"]
    ua = user_agent or cfg["http"]["user_agent"]
    return _fetch_rss_template(template, ticker, "google_news", ua, lookback_hours)


def fetch_all_for_ticker(ticker: str, lookback_hours: int = 6) -> List[Headline]:
    """Convenience: pull from all configured per-symbol sources."""
    out: List[Headline] = []
    out.extend(fetch_yahoo(ticker, lookback_hours=lookback_hours))
    out.extend(fetch_google_news(ticker, lookback_hours=lookback_hours))
    # Sort newest-first
    out.sort(key=lambda h: h.ts_publish, reverse=True)
    return out


def render_context_block(headlines: List[Headline], max_items: int = 8) -> str:
    """Format headlines for inclusion in the Haiku scoring prompt."""
    if not headlines:
        return ""
    lines = ["Recent headlines for this issuer (last 6h, aggregator sources):"]
    for h in headlines[:max_items]:
        ts = h.ts_publish.strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"  [{ts}] ({h.source}) {h.headline}")
    return "\n".join(lines)


def main():
    """CLI: fetch + render for a given ticker."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("ticker")
    p.add_argument("--lookback-hours", type=int, default=6)
    args = p.parse_args()

    hls = fetch_all_for_ticker(args.ticker, lookback_hours=args.lookback_hours)
    print(f"\n{len(hls)} headlines for {args.ticker} in last {args.lookback_hours}h:\n")
    print(render_context_block(hls))


if __name__ == "__main__":
    main()
