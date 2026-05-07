"""S&P 500 ticker→CIK universe.

Sources:
  - SEC company_tickers.json (current ticker→CIK mapping for all SEC filers)
  - Wikipedia "List of S&P 500 companies" (current membership)

Point-in-time historical S&P 500 membership is hard to get free; for backtest
we use current membership, which introduces survivorship bias. Documented in
SOURCES.md and accepted for the 2y backfill window (turnover ~5%/yr is small
on a 2y horizon). For 5y+ backtests we'd need a paid CRSP feed or a curated
historical list.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def fetch_sec_ticker_cik_map(user_agent: str, force: bool = False) -> dict[str, str]:
    """Return {TICKER: CIK_padded10}.  Cached to data/ticker_cik.json."""
    cache = DATA_DIR / "ticker_cik.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    r = requests.get(SEC_TICKERS_URL, headers={"User-Agent": user_agent}, timeout=30)
    r.raise_for_status()
    raw = r.json()
    out: dict[str, str] = {}
    for v in raw.values():
        ticker = v["ticker"].upper()
        cik = str(v["cik_str"]).zfill(10)
        out[ticker] = cik
    cache.write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def fetch_sp500_tickers(user_agent: str, force: bool = False) -> list[str]:
    """Current S&P 500 tickers from Wikipedia.  Cached to data/sp500.json."""
    cache = DATA_DIR / "sp500.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    r = requests.get(WIKI_SP500_URL, headers={"User-Agent": user_agent}, timeout=30)
    r.raise_for_status()
    html = r.text
    # Wikipedia tables: first tbody, rows have <td><a ... title="...">TICKER</a></td>
    # Extract from the constituents table by anchor pattern
    m = re.search(r'id="constituents"', html)
    if not m:
        raise RuntimeError("constituents table not found on Wikipedia")
    sub = html[m.start():]
    # Each row starts with the ticker link: tickers are 1-5 uppercase letters/numbers/.
    # Common pattern: href="/wiki/...">AAPL</a> in the first column
    tickers = re.findall(r'rel="nofollow" class="external text"[^>]*>([A-Z][A-Z0-9.\-]{0,5})</a>', sub)
    if len(tickers) < 400:
        # Fallback: NYSE/Nasdaq external symbol-page links
        tickers = re.findall(r'>(?:NYSE|NASDAQ):\s*([A-Z][A-Z0-9.\-]{0,5})<', sub)
    if len(tickers) < 400:
        # Final fallback: any all-caps ticker-shaped string in the table that maps to SEC
        tmap = fetch_sec_ticker_cik_map(user_agent)
        candidates = re.findall(r'<td[^>]*>([A-Z][A-Z0-9.\-]{0,5})</td>', sub[:200000])
        tickers = [t for t in candidates if t in tmap]
    # Dedup preserving order
    seen, out = set(), []
    for t in tickers:
        # Normalize BRK.B → BRK-B for some sources; keep both forms callers can map
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    if len(out) < 400 or len(out) > 600:
        raise RuntimeError(f"S&P 500 parse implausible: got {len(out)} tickers")
    cache.write_text(json.dumps(out, indent=2))
    return out


def fetch_nasdaq100_tickers(user_agent: str, force: bool = False) -> list[str]:
    """Current Nasdaq-100 tickers from Wikipedia.  Cached to data/nasdaq100.json.

    Lower validation bound (80) than S&P 500 because the index is exactly 100
    names; we accept 80–110 to allow for Wikipedia formatting wobble.
    """
    cache = DATA_DIR / "nasdaq100.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    r = requests.get(WIKI_NASDAQ100_URL, headers={"User-Agent": user_agent}, timeout=30)
    r.raise_for_status()
    html = r.text
    # The "Components"/"Constituents" section has rows with external links
    # of the form: rel="nofollow" class="external text">TICK</a>
    m = re.search(r'id="Components"', html) or re.search(r'id="Constituents"', html)
    sub = html[m.start():] if m else html
    tickers = re.findall(r'rel="nofollow"[^>]*>([A-Z][A-Z0-9.\-]{0,5})</a>', sub)
    if len(tickers) < 80:
        # Fallback: raw td cells
        tickers = re.findall(r'<td[^>]*>([A-Z][A-Z0-9.\-]{0,5})</td>', sub[:200000])
    seen, out = set(), []
    for t in tickers:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    if len(out) < 80 or len(out) > 110:
        raise RuntimeError(f"Nasdaq-100 parse implausible: got {len(out)} tickers")
    cache.write_text(json.dumps(out, indent=2))
    return out


def tradable_universe(user_agent: str) -> set[str]:
    """Return the set of tickers we are willing to trade live.

    Today: S&P 500 ∪ Nasdaq-100 ≈ 550 unique names. Liquidity floor implicit
    in membership. Cached files are refreshed only when the user manually
    deletes ``data/sp500.json`` / ``data/nasdaq100.json``.
    """
    sp = set(fetch_sp500_tickers(user_agent))
    n100 = set(fetch_nasdaq100_tickers(user_agent))
    return sp | n100


def sp500_cik_set(user_agent: str) -> set[str]:
    """Return CIKs (padded 10) for S&P 500 members that resolve in SEC mapping."""
    tmap = fetch_sec_ticker_cik_map(user_agent)
    sp500 = fetch_sp500_tickers(user_agent)
    ciks = set()
    unresolved = []
    for t in sp500:
        # Try direct match, then dot/dash swaps (e.g. BRK.B vs BRK-B vs BRKB)
        for candidate in (t, t.replace(".", "-"), t.replace(".", ""), t.replace("-", ".")):
            if candidate in tmap:
                ciks.add(tmap[candidate])
                break
        else:
            unresolved.append(t)
    if unresolved:
        print(f"[universe] {len(unresolved)} S&P 500 tickers unresolved in SEC map: {unresolved[:10]}{'...' if len(unresolved)>10 else ''}")
    return ciks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from news_quant.config_loader import load
    cfg = load()
    ua = cfg["http"]["user_agent"]

    tmap = fetch_sec_ticker_cik_map(ua)
    print(f"SEC ticker→CIK map: {len(tmap)} entries")
    sp = fetch_sp500_tickers(ua)
    print(f"S&P 500 tickers: {len(sp)}  sample: {sp[:10]}")
    ciks = sp500_cik_set(ua)
    print(f"S&P 500 CIKs resolved: {len(ciks)}")
