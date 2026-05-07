"""Per-symbol news fetcher tests with mocked feedparser."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from news_quant.news.per_symbol import (
    Headline,
    _fetch_rss_template,
    fetch_all_for_ticker,
    render_context_block,
)


class _FakeEntry:
    def __init__(self, title, link, summary, ts):
        self.title = title
        self.link = link
        self.summary = summary
        self.published_parsed = ts.timetuple()
        self.updated_parsed = ts.timetuple()


class _FakeFeed:
    def __init__(self, entries):
        self.entries = entries


def test_fetch_template_filters_old_entries_and_caps_count():
    now = datetime.now(timezone.utc)
    entries = [
        _FakeEntry("FRESH 1", "u1", "s1", now - timedelta(minutes=5)),
        _FakeEntry("FRESH 2", "u2", "s2", now - timedelta(minutes=30)),
        _FakeEntry("STALE",   "u3", "s3", now - timedelta(hours=24)),
    ]
    with patch("news_quant.news.per_symbol.feedparser.parse",
               return_value=_FakeFeed(entries)):
        out = _fetch_rss_template(
            "https://example.test/{ticker}", "AAPL",
            "yahoo_finance", "ua", lookback_hours=6, max_items=10,
        )
    assert len(out) == 2
    assert all(h.ticker == "AAPL" for h in out)
    assert all(h.source == "yahoo_finance" for h in out)
    assert {h.headline for h in out} == {"FRESH 1", "FRESH 2"}


def test_render_context_block_handles_empty():
    assert render_context_block([]) == ""


def test_render_context_block_formats_headlines():
    h = Headline(
        source="yahoo_finance", ticker="AAPL",
        ts_publish=datetime(2026, 5, 7, 14, 30, tzinfo=timezone.utc),
        headline="AAPL hits new high",
        url="http://example.test/x", summary="",
    )
    out = render_context_block([h])
    assert "AAPL hits new high" in out
    assert "yahoo_finance" in out
    assert "2026-05-07" in out


def test_fetch_all_returns_combined_sorted():
    now = datetime.now(timezone.utc)
    yh = [_FakeEntry("Y old",  "y1", "", now - timedelta(hours=2))]
    gn = [_FakeEntry("G new",  "g1", "", now - timedelta(minutes=5))]
    with patch("news_quant.news.per_symbol.feedparser.parse",
               side_effect=[_FakeFeed(yh), _FakeFeed(gn)]):
        out = fetch_all_for_ticker("AAPL", lookback_hours=6)
    assert len(out) == 2
    # Newest first
    assert out[0].headline == "G new"
    assert out[1].headline == "Y old"
