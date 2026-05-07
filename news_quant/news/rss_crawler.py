"""Multi-source RSS crawler (Tier 1 + Tier 2).

Live ingestion only; no historical archive.  Polls each feed at a configurable
interval, dedups via news.store.content_hash, persists to news_store.sqlite.

Sources are read from config.yaml `sources.rss` and `sources.credibility`.
Each item gets:
  - ts_publish from the feed (CDATA-aware)
  - ts_fetch = now()
  - credibility from config (defaults to 0.3 for unknown)
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser

from news_quant.config_loader import load as load_config
from news_quant.news import store


def _parse_published(entry) -> datetime | None:
    """feedparser exposes `published_parsed` (struct_time, UTC)."""
    p = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if p is None:
        return None
    try:
        return datetime(*p[:6], tzinfo=timezone.utc)
    except Exception:
        return None


def crawl_once(rss_sources: dict[str, str], credibility_map: dict[str, float], user_agent: str) -> dict:
    """Single crawl pass over all sources.  Returns counts."""
    inserted = 0
    duplicates = 0
    errors: list[tuple[str, str]] = []

    for source, url in rss_sources.items():
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": user_agent})
            cred = float(credibility_map.get(source, 0.3))
            for entry in feed.entries:
                headline = getattr(entry, "title", "").strip()
                link = getattr(entry, "link", "")
                ts_pub = _parse_published(entry)
                if not headline or ts_pub is None:
                    continue
                body = getattr(entry, "summary", "") or ""
                ok = store.upsert(
                    headline=headline,
                    url=link,
                    source=source,
                    credibility=cred,
                    ts_publish=ts_pub,
                    body=body,
                )
                if ok:
                    inserted += 1
                else:
                    duplicates += 1
        except Exception as e:
            errors.append((source, f"{type(e).__name__}: {e}"))
    return {"inserted": inserted, "duplicates": duplicates, "errors": errors}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="single pass and exit")
    p.add_argument("--interval-seconds", type=int, default=300)
    args = p.parse_args()

    cfg = load_config()
    ua = cfg["http"]["user_agent"]
    sources = cfg["sources"]["rss"]
    credibility = cfg["sources"]["credibility"]

    while True:
        t0 = time.time()
        result = crawl_once(sources, credibility, ua)
        print(
            f"[{datetime.now(timezone.utc).isoformat()}]"
            f"  inserted={result['inserted']}  dup={result['duplicates']}  "
            f"errors={len(result['errors'])}  elapsed={time.time()-t0:.1f}s"
        )
        for src, msg in result["errors"]:
            print(f"    error {src}: {msg}")
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
