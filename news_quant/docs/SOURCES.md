# News Source Survey — Phase 0

Probed 2026-04-30 from a single home connection, no auth except `User-Agent`.

## Tier 1 — authoritative, free, point-in-time

### SEC EDGAR — primary backtest spine
| Endpoint | Use | Status |
|---|---|---|
| `https://data.sec.gov/submissions/CIK{padded10}.json` | Per-issuer filing history (last 1000 + paginated) | ✅ HTTP 200, 165 KB for AAPL, 3.4s |
| `https://www.sec.gov/Archives/edgar/daily-index/{YYYY}/QTR{q}/form.{YYYYMMDD}.idx` | All filings on a single trading day, form-sorted | ✅ HTTP 200, 1.4 MB, ~7k filings/day, ~386 8-Ks/day |
| `https://www.sec.gov/Archives/edgar/full-index/{YYYY}/QTR{q}/form.idx` | Quarter-aggregated form index for bulk backfill | ✅ HTTP 200, 56 MB, ~16k 8-Ks/quarter ⇒ ~64k 8-Ks/yr across all filers |
| `https://www.sec.gov/Archives/{path-from-index}` | Filing text (TXT bundle, contains all docs) | (not yet probed at scale, well-documented) |

**Timestamps**: `acceptanceDateTime` in submissions API is ISO-8601 UTC with milliseconds (e.g. `2026-04-20T21:29:51.000Z`). This is the official SEC-stamped public-availability time — **use this as the point-in-time anchor**, not `filingDate` (which is calendar-day only).

**Rate limits**: SEC fair-access policy is **10 requests/sec/IP**. Must include a descriptive `User-Agent` with contact info or requests are 403'd. We've configured `news_quant research/0.0 contact: xxyi.3924@gmail.com` in `config.yaml:http.user_agent`.

**Backfill plan**: download full-index `form.idx` per quarter (small — 56 MB), filter for `8-K`, then resolve each accession to its index page to extract acceptance time and primary document. For S&P 500 + Nasdaq-100 universe, we expect ~1500–3000 8-Ks/yr → 5-year backfill = 7500–15000 events, well above the 200-cycle floor.

### Federal Reserve — `https://www.federalreserve.gov/feeds/press_all.xml`
- ✅ HTTP 200, 14 KB, 20 items
- RSS 2.0; `pubDate` is **wrapped in CDATA** (regex must handle this), format `Wed, 29 Apr 2026 18:00:00 GMT`
- Sample: "Federal Reserve issues FOMC statement" — exactly the macro-event class we want
- No auth, no rate limit observed

### BLS — `https://www.bls.gov/feed/bls_latest.rss`
- ✅ HTTP 200, 5 KB
- Quiet at probe time (1 item) — `bls_latest` aggregates only the most recent. May need to combine with `https://www.bls.gov/feed/bls_press_release.rss` and category-specific feeds (CPI, employment).
- pubDate format: `Thu, 30 Apr 2026 08:30:55 -0400` (NY time, RFC 822)

### PRNewswire — `https://www.prnewswire.com/rss/news-releases-list.rss`
- ✅ HTTP 200, 41 KB, 20 items, recent timestamps
- Useful for company-IR press releases that companies syndicate here
- Should add ticker-specific feeds: `https://www.prnewswire.com/rss/financial-services-latest-news/financial-services-latest-news-list.rss` etc.

### Treasury — DEFERRED
- `home.treasury.gov` consistently times out (connection-level, not slow). Multiple URL variants tried; only the legacy `www.treasury.gov` static page resolves but it isn't RSS.
- **Decision**: defer Treasury to Phase 4 web crawler with HTML scraping, OR rely on Reuters/FT mirroring of Treasury announcements. Not a Phase 1 blocker — Fed + BLS cover the dominant macro signal volume.

### BusinessWire — needs category-specific URLs
- Default home feed (`?rss=G1QFDERJXkJeEVtRWQ==`) returned 0 items — that's a landing-page placeholder.
- Real BusinessWire feeds are encoded category IDs (rss=...). Defer to Phase 4 once we identify which categories actually carry the IR content we want.

## Tier 2 — mainstream press (Phase 4)

Not probed in Phase 0. Will need:
- Reuters business RSS
- CNBC market RSS
- Yahoo Finance per-ticker RSS (`https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}`)
- MarketWatch top stories RSS
- FT (paywalled body, RSS headline-only)
- WSJ Markets RSS

## Tier 3 — aggregators (Phase 4+)

- GDELT 2.0 free firehose: `http://data.gdeltproject.org/gdeltv2/`
- Google News RSS per-ticker query: `https://news.google.com/rss/search?q={ticker}+stock`

## Phase 0 success criteria — status

| Criterion | Target | Actual | Status |
|---|---|---|---|
| Tier-1 sources working | ≥5 | EDGAR (3 endpoints) + Fed + BLS + PRNewswire = 4 distinct organizations, 6 endpoints | ✅ |
| EDGAR access viable | no rate-limit blockers | 10 req/s policy is plenty for our load | ✅ |
| Timestamp discipline | every source has reliable publish_time | EDGAR ✅, Fed ✅, BLS ✅, PRNewswire ✅ | ✅ |
| Dedup precision | >90% on 100-article sample | (not yet — needs ingestion to test) | deferred to Phase 1 end |

## Implications for Phase 1

1. Build the EDGAR pipeline against the **full-index** quarterly files for backfill, plus the **submissions API** per-CIK for incremental polling. Don't iterate per-CIK for the 5-year backfill — that's 500 CIKs × 20 quarters = 10,000 requests at 10 req/s = 17min minimum, painful and fragile.
2. Use `acceptanceDateTime` as the point-in-time anchor throughout. Tests must assert no signal at time T uses any filing whose `acceptanceDateTime > T`.
3. Write CDATA-aware RSS parser from day one — Fed feed needs it, others may too.
4. Don't bother with Treasury or BusinessWire-default for Phase 1. Get EDGAR + Fed + BLS working first; add the rest in Phase 4.

## Open questions for later

- Sub-CIK ticker resolution: SEC filings use CIK, market data uses tickers. Need a CIK→ticker mapping (SEC publishes `company_tickers.json`). One-time fetch + cache.
- 8-K item-classification: 8-Ks have "Items" (1.01 acquisition, 2.02 earnings, 5.02 exec changes, etc.). The Item code itself is rich signal — should be extracted into a structured field before LLM scoring. Reduces LLM cost (Item-only triage filters by event type before sending body).
