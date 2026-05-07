"""Point-in-time discipline: news_store.query_for_backtest must NOT return
items whose ts_publish > the as-of timestamp.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from news_quant.news import store


@pytest.fixture(autouse=True)
def _temp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test_news.sqlite")
    yield


def _ts(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_no_lookahead_in_query():
    base = _ts("2025-04-01T12:00:00")
    # Past event
    store.upsert(
        headline="Past news for AAPL",
        url="https://x/1",
        source="reuters",
        credibility=0.9,
        ts_publish=base - timedelta(hours=2),
        tickers=["AAPL"],
    )
    # FUTURE event — must not show up at as_of=base
    store.upsert(
        headline="Future news for AAPL",
        url="https://x/2",
        source="reuters",
        credibility=0.9,
        ts_publish=base + timedelta(hours=2),
        tickers=["AAPL"],
    )
    rows = store.query_for_backtest("AAPL", base)
    assert len(rows) == 1
    assert rows[0]["headline"] == "Past news for AAPL"


def test_dedup_same_headline_url():
    base = _ts("2025-04-01T12:00:00")
    a = store.upsert(
        headline="Apple beats earnings",
        url="https://x/dup",
        source="reuters",
        credibility=0.9,
        ts_publish=base,
        tickers=["AAPL"],
    )
    b = store.upsert(
        headline="Apple beats earnings",
        url="https://x/dup",
        source="cnbc",  # different source, same content -> dedup
        credibility=0.7,
        ts_publish=base,
        tickers=["AAPL"],
    )
    assert a is True and b is False


def test_fetch_time_records_separately(monkeypatch):
    base = _ts("2025-04-01T12:00:00")
    fetch = _ts("2025-04-01T12:05:00")  # 5 minutes after publish
    store.upsert(
        headline="Article with delayed fetch",
        url="https://x/3",
        source="reuters",
        credibility=0.9,
        ts_publish=base,
        ts_fetch=fetch,
        tickers=["AAPL"],
    )
    rows = store.query_for_backtest("AAPL", base)
    assert len(rows) == 1
    # Backtest discipline: ts_publish is what's used for joining; ts_fetch is for live ops only.
    assert rows[0]["ts_publish"].startswith("2025-04-01T12:00")
