"""Extract a US-equity ticker from a press-release headline or summary.

BW and PRN feeds typically embed the ticker in one of these forms:

  "Acme Corp (NYSE: ACME) announces …"
  "$ACME issues guidance …"
  "Acme Corporation (NASDAQ: ACME, ACME-W) reports …"
  "Acme Corp (Nasdaq:ACME) ..."  (no space)

Strategy: regex first; if nothing found, give up. We do NOT do company-name
fuzzy matching here — false positives on a tradable name are dangerous.

Returns the first plausible ticker that exists in the SEC ticker→CIK map.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from news_quant.config_loader import load as load_config
from news_quant.news.universe import fetch_sec_ticker_cik_map

# Patterns ordered most→least specific. Capture group 1 must be the ticker.
_PATTERNS = [
    # "(NYSE: ACME)" / "(Nasdaq:ACME)" / "(OTC: ACME)" — primary listing exchanges
    re.compile(r"\((?:NYSE|NASDAQ|AMEX|OTC|BATS|CBOE|ARCA)(?:\s*:\s*)([A-Z][A-Z0-9.\-]{0,5})", re.IGNORECASE),
    # Bare exchange tag: "NYSE: ACME"
    re.compile(r"(?:NYSE|NASDAQ|AMEX|OTC|BATS|CBOE|ARCA)\s*:\s*([A-Z][A-Z0-9.\-]{0,5})", re.IGNORECASE),
    # Cashtag: "$ACME"
    re.compile(r"\$([A-Z][A-Z0-9.\-]{0,5})\b"),
]


@lru_cache(maxsize=1)
def _ticker_universe() -> frozenset[str]:
    cfg = load_config()
    ua = cfg["http"]["user_agent"]
    return frozenset(fetch_sec_ticker_cik_map(ua).keys())


def extract_ticker(text: str) -> Optional[str]:
    if not text:
        return None
    universe = _ticker_universe()
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            cand = m.group(1).upper().rstrip(".,;:)]}\"'")
            # Normalize common variants
            for variant in (cand, cand.replace(".", "-"), cand.replace("-", "."), cand.replace(".", "")):
                if variant in universe:
                    return variant
    return None
