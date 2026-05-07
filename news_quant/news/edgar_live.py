"""Live EDGAR 8-K poller.

Two modes:

  - ``poll_bulk()`` (DEFAULT for live ops). Single HTTP call to SEC's
    `getcurrent` Atom feed which returns the ~40 most-recent 8-K filings
    across all issuers. ~1-2 seconds per poll cycle. The bulk feed gives
    us CIK + accession + items hint + updated timestamp; we look up the
    ticker via the cached ticker_cik map and use the filing's index URL
    as the primary URL for scoring.

  - ``poll_once()`` (per-CIK loop, original implementation). Walks every
    S&P 500 CIK's submissions endpoint. Slower (~5 min for 500 CIKs at the
    SEC fair-use limit) but produces full primary_document URLs and is the
    fallback if `getcurrent` is unavailable.

Both paths persist to ``news_store.sqlite`` and dedup via SHA-based
content_hash so a process restart does not double-emit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests

from news_quant.config_loader import load as load_config
from news_quant.news import store
from news_quant.news.edgar_8k import extract_8k_filings, fetch_submissions
from news_quant.news.universe import fetch_sec_ticker_cik_map, sp500_cik_set

import re
import xml.etree.ElementTree as ET

logger = logging.getLogger("news_quant.news.edgar_live")

EDGAR_GETCURRENT_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&output=atom"
)
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
TITLE_CIK_RE = re.compile(r"\((\d{4,10})\)")
SUMMARY_ITEM_RE = re.compile(r"Item\s+(\d+\.\d+)")
SUMMARY_ACC_RE = re.compile(r"AccNo:\s*(\d{10}-\d{2}-\d{6})")


@dataclass
class NewsEvent:
    """Normalized event produced by any source poller (EDGAR or RSS).

    The aggregator and signal gate both consume this shape.
    """

    source: str                 # "edgar_8k" | "businesswire" | "prnewswire" | …
    ticker: str
    issuer_name: str
    acceptance_dt_utc: datetime
    headline: str               # short identifier — for EDGAR, the form+items
    primary_url: str
    accession_or_id: str        # SEC accession or RSS link (used as dedup key)
    items: str                  # SEC item codes; "" for non-SEC sources
    credibility: float          # config-driven, EDGAR=1.0, news wires lower


class EdgarLivePoller:
    def __init__(
        self,
        user_agent: str,
        rate_limit_per_sec: int = 10,
        lookback_minutes: int = 30,
    ):
        self.user_agent = user_agent
        self.sleep_per_req = 1.0 / max(1, rate_limit_per_sec)
        self.lookback = timedelta(minutes=lookback_minutes)
        self._ciks: list[str] = []
        self._session = requests.Session()
        self._loaded_universe = False
        self._cik_to_ticker: dict[str, str] = {}

    def _ensure_universe(self) -> None:
        if self._loaded_universe:
            return
        self._ciks = sorted(sp500_cik_set(self.user_agent))
        # Build CIK→ticker reverse map from the cached SEC company_tickers.json.
        tmap = fetch_sec_ticker_cik_map(self.user_agent)  # {TICKER: CIK_padded10}
        for tk, cik in tmap.items():
            # Prefer the first ticker we see for each CIK (alphabetical from SEC).
            self._cik_to_ticker.setdefault(cik, tk)
        self._loaded_universe = True
        logger.info(
            "EdgarLivePoller universe loaded: %d S&P 500 CIKs, %d total CIK→ticker entries",
            len(self._ciks), len(self._cik_to_ticker),
        )

    def poll_bulk(self) -> Iterator[NewsEvent]:
        """Single-call bulk poll via SEC's `getcurrent` 8-K Atom feed.

        Returns ~40 most recent 8-K filings across all issuers. Wall-clock is
        ~1-2 seconds, dominated by HTTP latency. Use this as the default for
        live ops; fall back to ``poll_once()`` only if `getcurrent` is down.
        """
        self._ensure_universe()
        now = datetime.now(timezone.utc)
        cutoff = now - self.lookback
        try:
            r = self._session.get(
                EDGAR_GETCURRENT_URL,
                headers={"User-Agent": self.user_agent},
                timeout=15,
            )
            r.raise_for_status()
        except Exception as e:
            logger.warning("getcurrent fetch failed: %s", e)
            return
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError as e:
            logger.warning("getcurrent parse failed: %s", e)
            return

        seen = emitted = no_ticker = 0
        for entry in root.findall("a:entry", ATOM_NS):
            seen += 1
            title = (entry.findtext("a:title", default="", namespaces=ATOM_NS) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=ATOM_NS) or "").strip()
            updated = (entry.findtext("a:updated", default="", namespaces=ATOM_NS) or "").strip()
            link_el = entry.find("a:link[@rel='alternate']", ATOM_NS)
            link = link_el.get("href") if link_el is not None else ""

            # Acceptance time
            try:
                acc_dt = datetime.fromisoformat(updated).astimezone(timezone.utc)
            except ValueError:
                continue
            if acc_dt < cutoff:
                continue

            # CIK from title: "8-K - ACME CORP (0001234567) (Filer)"
            m_cik = TITLE_CIK_RE.search(title)
            if not m_cik:
                continue
            cik = m_cik.group(1).zfill(10)

            ticker = self._cik_to_ticker.get(cik, "")
            if not ticker:
                no_ticker += 1
                continue

            # Issuer name = whatever is between "8-K - " and " (CIK)"
            issuer = title.split(" - ", 1)[-1]
            issuer = issuer.split("(")[0].strip()

            # Items list from summary, e.g. "Item 5.07 ..." → "5.07"
            items = ",".join(SUMMARY_ITEM_RE.findall(summary))
            # Accession from summary or atom id
            m_acc = SUMMARY_ACC_RE.search(summary)
            accession = m_acc.group(1) if m_acc else ""

            headline = f"8-K {issuer} ({ticker}) items={items}"
            is_new = store.upsert(
                headline=headline,
                url=link or accession,
                source="edgar_8k",
                credibility=1.0,
                ts_publish=acc_dt,
                tickers=[ticker],
                body=summary,
            )
            if not is_new:
                continue
            emitted += 1
            yield NewsEvent(
                source="edgar_8k",
                ticker=ticker,
                issuer_name=issuer,
                acceptance_dt_utc=acc_dt,
                headline=headline,
                primary_url=link,
                accession_or_id=accession,
                items=items,
                credibility=1.0,
            )
        logger.info(
            "EdgarLivePoller poll_bulk: seen=%d emitted=%d no_ticker=%d",
            seen, emitted, no_ticker,
        )

    def poll_once(self) -> Iterator[NewsEvent]:
        """One full sweep of all S&P 500 CIKs.

        Yields events whose acceptance time falls in the last ``lookback_minutes``
        and that we have not previously seen (dedup via store.upsert returning
        True). The dedup key is the SEC accession number embedded in
        ``primary_url`` so we never re-emit a filing already in the news store.
        """
        self._ensure_universe()
        now = datetime.now(timezone.utc)
        cutoff = now - self.lookback
        # 8-K filing_date is YYYY-MM-DD; today and yesterday cover lookback unless
        # process is paused for >1 day.
        d_today = now.date().isoformat()
        d_yest = (now - timedelta(days=1)).date().isoformat()
        emitted = 0
        seen = 0
        errors = 0

        for cik in self._ciks:
            try:
                sub = fetch_submissions(cik, self.user_agent, self._session)
            except Exception as e:
                errors += 1
                logger.debug("fetch_submissions(%s) failed: %s", cik, e)
                time.sleep(self.sleep_per_req)
                continue

            filings = extract_8k_filings(sub, d_yest, d_today)
            for f in filings:
                acc_dt_raw = f.get("acceptance_dt_utc")
                if acc_dt_raw is None:
                    continue
                try:
                    acc_dt = pd.to_datetime(acc_dt_raw, utc=True).to_pydatetime()
                except Exception:
                    continue
                if acc_dt < cutoff:
                    continue
                seen += 1
                ticker = (f.get("ticker") or "").strip()
                accession = f.get("accession") or ""
                primary_url = f.get("primary_url") or ""
                items = f.get("items") or ""
                issuer_name = f.get("issuer_name") or ""
                headline = f"8-K {issuer_name} ({ticker}) items={items}"

                # Persist; if upsert returns True the event is new.
                is_new = store.upsert(
                    headline=headline,
                    url=primary_url or accession,
                    source="edgar_8k",
                    credibility=1.0,
                    ts_publish=acc_dt,
                    tickers=[ticker] if ticker else (),
                    body="",
                )
                if not is_new:
                    continue
                emitted += 1
                yield NewsEvent(
                    source="edgar_8k",
                    ticker=ticker,
                    issuer_name=issuer_name,
                    acceptance_dt_utc=acc_dt,
                    headline=headline,
                    primary_url=primary_url,
                    accession_or_id=accession,
                    items=items,
                    credibility=1.0,
                )
            time.sleep(self.sleep_per_req)

        logger.info(
            "EdgarLivePoller poll_once: seen=%d emitted=%d errors=%d",
            seen, emitted, errors,
        )


def main():
    """Standalone CLI: poll once via bulk feed (default) and print new events."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--lookback-min", type=int, default=120)
    p.add_argument("--per-cik", action="store_true",
                   help="use the slow per-CIK loop instead of the bulk feed (debug)")
    args = p.parse_args()

    cfg = load_config()
    ua = cfg["http"]["user_agent"]
    rate = cfg["sources"]["edgar"]["rate_limit_per_sec"]

    poller = EdgarLivePoller(ua, rate_limit_per_sec=rate, lookback_minutes=args.lookback_min)
    iterator = poller.poll_once() if args.per_cik else poller.poll_bulk()
    n = 0
    for ev in iterator:
        n += 1
        print(f"[{ev.acceptance_dt_utc.isoformat()}] {ev.ticker:>6} {ev.items:<14} {ev.headline}")
    print(f"\ntotal new events: {n}")


if __name__ == "__main__":
    main()
