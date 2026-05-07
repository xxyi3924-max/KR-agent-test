"""Targeted web scraper for sources without RSS (Phase 4).

Used for Reuters/CNBC/MarketWatch front-page articles where we already have
a URL but not full body text.  RSS gives us headline+summary; the scraper
fills in the body when the LLM scoring needs more context.

Honors per-domain rate limits and robots.txt (delegated to a small
allowlist; we're not building a general crawler).
"""

from __future__ import annotations

import re
import time
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

# Conservative allowlist — only domains we've confirmed permit non-commercial
# scraping for research with rate-limit politeness.
ALLOWED_DOMAINS = {
    "reuters.com": 2.0,
    "www.reuters.com": 2.0,
    "cnbc.com": 2.0,
    "www.cnbc.com": 2.0,
    "marketwatch.com": 2.0,
    "www.marketwatch.com": 2.0,
    "finance.yahoo.com": 1.0,
    "businesswire.com": 1.0,
    "www.businesswire.com": 1.0,
    "prnewswire.com": 1.0,
    "www.prnewswire.com": 1.0,
}

_last_fetch_at: dict[str, float] = {}


class _MainTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._in_skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "aside"):
            self._in_skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "aside") and self._in_skip > 0:
            self._in_skip -= 1

    def handle_data(self, data):
        if self._in_skip:
            return
        s = data.strip()
        if s:
            self.parts.append(s)


def fetch_article_text(url: str, user_agent: str, max_chars: int = 8000) -> str | None:
    host = urlparse(url).hostname or ""
    if host not in ALLOWED_DOMAINS:
        return None
    delay = ALLOWED_DOMAINS[host]
    last = _last_fetch_at.get(host, 0.0)
    wait = (last + delay) - time.time()
    if wait > 0:
        time.sleep(wait)
    try:
        r = requests.get(url, headers={"User-Agent": user_agent}, timeout=15)
        _last_fetch_at[host] = time.time()
        if r.status_code != 200:
            return None
        body = r.text
    except Exception:
        return None

    p = _MainTextExtractor()
    try:
        p.feed(body)
    except Exception:
        return body[:max_chars]
    text = " ".join(p.parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
