"""Point-in-time news store.

SQLite table `news` with separate `ts_publish` and `ts_fetch` columns.
Backtest joins use `ts_publish`; live ops use `ts_fetch`.  Tests in
test_news_pit.py assert the discipline.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DB_PATH = Path(__file__).parent.parent / "data" / "news_store.sqlite"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS news (
            content_hash TEXT PRIMARY KEY,
            ts_publish TEXT NOT NULL,
            ts_fetch TEXT NOT NULL,
            source TEXT NOT NULL,
            credibility REAL NOT NULL,
            url TEXT,
            tickers TEXT,
            headline TEXT NOT NULL,
            body TEXT
        )"""
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_publish ON news(ts_publish)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fetch ON news(ts_fetch)")
    return c


def content_hash(headline: str, url: str) -> str:
    h = hashlib.sha256()
    h.update(headline.strip().lower().encode("utf-8"))
    h.update(b"\n")
    h.update((url or "").strip().lower().encode("utf-8"))
    return h.hexdigest()


def upsert(
    *,
    headline: str,
    url: str,
    source: str,
    credibility: float,
    ts_publish: datetime,
    ts_fetch: datetime | None = None,
    tickers: Iterable[str] = (),
    body: str = "",
) -> bool:
    """Returns True if inserted, False if duplicate."""
    if ts_fetch is None:
        ts_fetch = datetime.now(timezone.utc)
    key = content_hash(headline, url)
    with _conn() as c:
        try:
            c.execute(
                "INSERT INTO news VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    ts_publish.astimezone(timezone.utc).isoformat(),
                    ts_fetch.astimezone(timezone.utc).isoformat(),
                    source,
                    float(credibility),
                    url,
                    ",".join(sorted(set(tickers))),
                    headline,
                    body,
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def query_for_backtest(
    ticker: str, as_of_publish_ts: datetime
) -> list[dict]:
    """Return all news for `ticker` published at or before as_of_publish_ts."""
    cutoff = as_of_publish_ts.astimezone(timezone.utc).isoformat()
    with _conn() as c:
        rows = c.execute(
            "SELECT ts_publish, source, credibility, url, headline, body, tickers "
            "FROM news WHERE ts_publish <= ? AND tickers LIKE ?",
            (cutoff, f"%{ticker}%"),
        ).fetchall()
    cols = ["ts_publish", "source", "credibility", "url", "headline", "body", "tickers"]
    return [dict(zip(cols, r)) for r in rows]
