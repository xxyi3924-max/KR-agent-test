"""LLM spend meter with daily cap.  Halts scoring when cap is exceeded."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "llm_costs.sqlite"

# Anthropic pricing as of 2026 (USD per 1M tokens) — keep in sync with API
PRICING = {
    "claude-haiku-4-5":   {"in": 1.00, "out":  5.00},
    "claude-sonnet-4-6":  {"in": 3.00, "out": 15.00},
}


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS costs (
            ts_utc TEXT NOT NULL,
            day TEXT NOT NULL,
            model TEXT NOT NULL,
            tokens_in INTEGER NOT NULL,
            tokens_out INTEGER NOT NULL,
            cost_usd REAL NOT NULL,
            tag TEXT
        )"""
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_day ON costs(day)")
    return c


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = PRICING.get(model)
    if not p:
        # Conservative fallback: assume Sonnet pricing
        p = PRICING["claude-sonnet-4-6"]
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000.0


def record(model: str, tokens_in: int, tokens_out: int, tag: str = "") -> float:
    cost = estimate_cost(model, tokens_in, tokens_out)
    now = datetime.now(timezone.utc)
    with _conn() as c:
        c.execute(
            "INSERT INTO costs VALUES (?,?,?,?,?,?,?)",
            (now.isoformat(), now.date().isoformat(), model, tokens_in, tokens_out, cost, tag),
        )
    return cost


def today_spend_usd() -> float:
    today = date.today().isoformat()
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM costs WHERE day=?", (today,)
        ).fetchone()
    return float(row[0]) if row else 0.0


def session_spend_usd(since_iso: str) -> float:
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM costs WHERE ts_utc >= ?",
            (since_iso,),
        ).fetchone()
    return float(row[0]) if row else 0.0


def assert_budget(daily_cap_usd: float) -> None:
    spent = today_spend_usd()
    if spent >= daily_cap_usd:
        raise RuntimeError(
            f"LLM daily cap exceeded: spent ${spent:.4f} >= cap ${daily_cap_usd:.2f}.  Halting."
        )


if __name__ == "__main__":
    print(f"today spend: ${today_spend_usd():.4f}")
