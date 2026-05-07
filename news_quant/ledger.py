"""Capital ledger with high-water-mark drawdown halt.

Subproject-isolated equity tracking.  The executor must consult `is_halted()`
before opening any position and call `record_trade()` after every fill.
A `HALTED` flag in the ledger is sticky — only `clear_halt()` (manual) re-arms.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "ledger.sqlite"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS equity (
            ts_utc TEXT NOT NULL,
            equity_usd REAL NOT NULL,
            note TEXT
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS trades (
            ts_utc TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            fees_usd REAL NOT NULL,
            cycle_id TEXT,
            note TEXT
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )"""
    )
    return c


def init_capital(amount_usd: float, note: str = "initial deposit") -> None:
    with _conn() as c:
        existing = c.execute("SELECT COUNT(*) FROM equity").fetchone()[0]
        if existing > 0:
            return
        c.execute(
            "INSERT INTO equity VALUES (?,?,?)",
            (datetime.now(timezone.utc).isoformat(), amount_usd, note),
        )


def record_equity(equity_usd: float, note: str = "") -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO equity VALUES (?,?,?)",
            (datetime.now(timezone.utc).isoformat(), equity_usd, note),
        )


def record_trade(
    ticker: str,
    side: str,
    qty: float,
    price: float,
    fees_usd: float,
    cycle_id: str = "",
    note: str = "",
) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                ticker.upper(),
                side.upper(),
                qty,
                price,
                fees_usd,
                cycle_id,
                note,
            ),
        )


def current_equity() -> float | None:
    with _conn() as c:
        row = c.execute(
            "SELECT equity_usd FROM equity ORDER BY ts_utc DESC LIMIT 1"
        ).fetchone()
    return float(row[0]) if row else None


def high_water_mark() -> float | None:
    with _conn() as c:
        row = c.execute("SELECT MAX(equity_usd) FROM equity").fetchone()
    return float(row[0]) if row and row[0] is not None else None


def drawdown_pct() -> float:
    eq = current_equity()
    hwm = high_water_mark()
    if eq is None or hwm is None or hwm <= 0:
        return 0.0
    return (1.0 - eq / hwm) * 100.0


def is_halted() -> bool:
    with _conn() as c:
        row = c.execute("SELECT value FROM state WHERE key='halted'").fetchone()
    return row is not None and row[0] == "1"


def set_halt(reason: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO state VALUES ('halted', '1')"
        )
        c.execute(
            "INSERT OR REPLACE INTO state VALUES ('halt_reason', ?)", (reason,)
        )


def clear_halt() -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO state VALUES ('halted', '0')")


def check_drawdown_halt(max_dd_pct: float) -> bool:
    """If drawdown exceeds threshold, halt and return True."""
    dd = drawdown_pct()
    if dd >= max_dd_pct:
        set_halt(f"drawdown {dd:.2f}% >= cap {max_dd_pct:.2f}%")
        return True
    return False


if __name__ == "__main__":
    eq = current_equity()
    hwm = high_water_mark()
    dd = drawdown_pct()
    halt = is_halted()
    print(f"equity:    {eq}")
    print(f"high-water:{hwm}")
    print(f"drawdown:  {dd:.2f}%")
    print(f"halted:    {halt}")
