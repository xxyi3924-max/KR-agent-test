"""SEC EDGAR 8-K backfill via per-CIK submissions API.

Why per-CIK and not the quarterly full-index?  Because the full-index gives us
form.idx rows but NOT acceptanceDateTime, which is the only point-in-time-correct
timestamp.  Submissions API gives form, accession, primaryDocument, AND acceptance
time in one call.  500 CIKs × 1 request each = 50s wall-clock at 10 req/s.

Output: parquet file at news_quant/data/edgar_8k_<start>_<end>.parquet with cols:
  cik, ticker, accession, form, filing_date, acceptance_dt_utc,
  primary_document, primary_url

The primary_url points at the filing's main HTML/XML; body fetch happens lazily
during Phase 2 LLM scoring (avoids storing GB of raw filings on disk).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from news_quant.config_loader import load as load_config
from news_quant.news.universe import fetch_sec_ticker_cik_map, sp500_cik_set

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


def fetch_submissions(cik: str, user_agent: str, session: requests.Session) -> dict:
    url = SUBMISSIONS_URL.format(cik=cik)
    r = session.get(url, headers={"User-Agent": user_agent}, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_8k_filings(
    submissions: dict, start_date: str, end_date: str
) -> list[dict]:
    """Return list of dicts for 8-K filings in [start, end].

    Note: SEC submissions API returns the most recent ~1000 filings in `recent`.
    For very active filers this may not cover 2y; we'd need to read the
    `files` field (paginated older history).  Most S&P 500 issuers file
    <100 docs/yr so 1000 covers ~10 years.  Documented assumption.
    """
    cik = submissions.get("cik", "").zfill(10)
    name = submissions.get("name", "")
    tickers = submissions.get("tickers", [])
    primary_ticker = tickers[0] if tickers else ""

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    fdates = recent.get("filingDate", [])
    acc_times = recent.get("acceptanceDateTime", [])
    prims = recent.get("primaryDocument", [])
    items = recent.get("items", [])

    out = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        fdate = fdates[i]
        if fdate < start_date or fdate > end_date:
            continue
        accession = accs[i]
        acc_no_dash = accession.replace("-", "")
        primary = prims[i] if i < len(prims) else ""
        primary_url = (
            f"{ARCHIVES_BASE}/{int(cik)}/{acc_no_dash}/{primary}" if primary else ""
        )
        out.append(
            {
                "cik": cik,
                "issuer_name": name,
                "ticker": primary_ticker,
                "accession": accession,
                "form": form,
                "filing_date": fdate,
                "acceptance_dt_utc": acc_times[i] if i < len(acc_times) else None,
                "items": items[i] if i < len(items) else "",
                "primary_document": primary,
                "primary_url": primary_url,
            }
        )
    return out


def backfill(
    start_date: str,
    end_date: str,
    user_agent: str,
    rate_limit_per_sec: int = 10,
) -> pd.DataFrame:
    """Walk all S&P 500 CIKs and collect 8-K filings in [start, end]."""
    ciks = sorted(sp500_cik_set(user_agent))
    print(f"[backfill] {len(ciks)} S&P 500 CIKs, window {start_date} → {end_date}")

    session = requests.Session()
    sleep_per_req = 1.0 / rate_limit_per_sec
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_start = time.time()
    for i, cik in enumerate(ciks, 1):
        try:
            sub = fetch_submissions(cik, user_agent, session)
            new = extract_8k_filings(sub, start_date, end_date)
            rows.extend(new)
        except Exception as e:
            failures.append((cik, f"{type(e).__name__}: {e}"))
        if i % 50 == 0:
            elapsed = time.time() - t_start
            print(
                f"  {i}/{len(ciks)} CIKs  elapsed={elapsed:.1f}s  "
                f"events={len(rows)}  failures={len(failures)}"
            )
        time.sleep(sleep_per_req)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["acceptance_dt_utc"] = pd.to_datetime(df["acceptance_dt_utc"], utc=True)
        df = df.sort_values("acceptance_dt_utc").reset_index(drop=True)
    print(
        f"[backfill] done in {time.time()-t_start:.1f}s  "
        f"events={len(df)}  failures={len(failures)}"
    )
    if failures:
        for cik, msg in failures[:5]:
            print(f"  fail CIK={cik}  {msg}")
    return df


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2026-04-30")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config()
    ua = cfg["http"]["user_agent"]
    rate = cfg["sources"]["edgar"]["rate_limit_per_sec"]

    df = backfill(args.start, args.end, ua, rate_limit_per_sec=rate)

    if df.empty:
        print("NO DATA — aborting write.")
        return

    out = Path(args.out) if args.out else DATA_DIR / f"edgar_8k_{args.start}_{args.end}.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df)} events → {out}")
    print("\nSummary:")
    print(f"  unique CIKs:       {df['cik'].nunique()}")
    print(f"  unique tickers:    {df['ticker'].nunique()}")
    print(f"  earliest:          {df['acceptance_dt_utc'].min()}")
    print(f"  latest:            {df['acceptance_dt_utc'].max()}")
    annual = df.groupby(df["acceptance_dt_utc"].dt.year).size()
    print(f"  by year:\n{annual.to_string()}")
    items_top = df["items"].value_counts().head(10)
    print(f"  top 10 Item codes:\n{items_top.to_string()}")


if __name__ == "__main__":
    main()
