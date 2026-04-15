"""
TDash Connector — Reads signals from Trading Dashboard v9 JSON output.

TDash saves run files as:
    {data_dir}/portfolio-run-{YYYYMMDD}-{HHMMSS}-{user}.json

We find the most recent one by sorting filenames descending (timestamp is
lexicographically sortable), load it, and map its structure to dataclasses
the decision engine can work with directly.
"""

import json
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """One actionable signal from a TDash run."""
    ticker: str
    action: str           # ADD | NEW | TRIM | SELL  (HOLD is filtered out)
    shares: float         # shares_to_buy or shares_to_sell
    entry_price: float    # entry_price_limit (limit order price)
    take_profit: float    # stop_levels.take_profit
    stop_loss: float      # stop_levels.stop_loss
    priority: int         # order within the run (1 = first)
    risk_flags: List[str]
    notes: str = ""
    # --- idempotency / context ---
    signal_id: str = ""       # "{run_id}:{ticker}:{action}" — unique per signal
    run_id: str = ""
    generated_at: str = ""    # run datetime ISO string
    composite: float = 0.0    # TDash composite score (0–1)
    timing_price: float = 0.0 # last_price at signal generation time
    trim_pct: float = 0.0     # % to trim (for TRIM actions only)
    dollar_amount: float = 0.0  # pre-computed dollar amount from TDash


@dataclass
class Holding:
    """A current holding as reported by the TDash run."""
    ticker: str
    shares: float
    avg_cost: float
    entry_date: str
    current_value: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Portfolio:
    """Full portfolio state from a TDash run."""
    market: str
    period: str
    risk_profile: str
    capital: float
    cash: float           # deploy_now / rebalance_budget
    holdings: List[Holding]
    signals: List[Signal]
    macro_context: str
    run_timestamp: str
    run_id: str
    token_usage: Dict


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class TDashConnector:
    """
    Reads the most recent TDash v9 portfolio run file.

    File naming convention:  portfolio-run-{YYYYMMDD}-{HHMMSS}-{user}.json
    These sort lexicographically by filename = chronologically by run time.
    """

    def __init__(self, data_dir: str, user: str = "admin"):
        self.data_dir = Path(os.path.expanduser(data_dir))
        self.user = user
        self._cached_run_id: str = ""
        self._cached_data: Optional[dict] = None

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _get_latest_run_file(self) -> Optional[Path]:
        """
        Return path to the most recent run JSON for this user.

        Supports two filename conventions:
          Old: portfolio-run-YYYYMMDD-HHMMSS-{user}.json  (local dev)
          New: YYYYMMDD-HHMMSS-{user}.json                (Lightsail/production)
        Both sort lexicographically = chronologically.
        """
        # New format (Lightsail): YYYYMMDD-HHMMSS-{user}.json
        pattern_new = str(self.data_dir / f"*-{self.user}.json")
        files = sorted(glob.glob(pattern_new), reverse=True)
        if files:
            return Path(files[0])

        # Old format (legacy): portfolio-run-YYYYMMDD-HHMMSS-{user}.json
        pattern_old = str(self.data_dir / f"portfolio-run-*-{self.user}.json")
        files = sorted(glob.glob(pattern_old), reverse=True)
        if files:
            return Path(files[0])

        # Last resort: any JSON in the runs dir
        pattern_any = str(self.data_dir / "*.json")
        files = sorted(glob.glob(pattern_any), reverse=True)
        if files:
            return Path(files[0])

        return None

    def _load_run(self) -> dict:
        """Load the most recent run JSON (with simple in-memory cache)."""
        run_file = self._get_latest_run_file()
        if run_file is None:
            raise FileNotFoundError(
                f"No portfolio-run-*.json found in {self.data_dir}"
            )

        run_id = run_file.stem  # filename without .json
        if run_id == self._cached_run_id and self._cached_data is not None:
            return self._cached_data

        with open(run_file) as f:
            data = json.load(f)

        self._cached_run_id = run_id
        self._cached_data = data
        return data

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_run_id(self) -> str:
        return self._load_run().get("run_id", "")

    def is_signal_stale(self, signal: Signal, max_age_hours: float = 72.0) -> bool:
        """True if the signal is older than max_age_hours."""
        if not signal.generated_at:
            return False
        try:
            generated = datetime.fromisoformat(signal.generated_at)
            age_hours = (datetime.now() - generated).total_seconds() / 3600.0
            return age_hours > max_age_hours
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Holdings
    # ------------------------------------------------------------------

    def load_holdings(self) -> List[Holding]:
        """Load current holdings from holdings_snapshot in the run JSON."""
        data = self._load_run()
        holdings = []
        for h in data.get("holdings_snapshot", []):
            holdings.append(Holding(
                ticker=h.get("ticker", ""),
                shares=float(h.get("shares", 0) or 0),
                avg_cost=float(h.get("cost_basis", 0) or 0),
                entry_date=h.get("entry_date", ""),
            ))
        return holdings

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def load_signals(self) -> List[Signal]:
        """
        Parse all actionable signals from the run JSON.

        Sources:
          portfolio_analysis[]  — existing holdings: ADD / TRIM / SELL (HOLD skipped)
          new_stocks[]          — new positions: NEW
        """
        data = self._load_run()
        run_id = data.get("run_id", "")
        generated_at = data.get("datetime", "")

        signals: List[Signal] = []
        priority = 1

        # --- portfolio_analysis: existing holdings ---
        for item in data.get("portfolio_analysis", []):
            action = (item.get("action") or "HOLD").upper()
            if action == "HOLD":
                continue

            sig = self._item_to_signal(item, action, run_id, generated_at, priority)
            if sig:
                signals.append(sig)
                priority += 1

        # --- new_stocks: new positions ---
        for item in data.get("new_stocks", []):
            action = (item.get("action") or "NEW").upper()
            sig = self._item_to_signal(item, action, run_id, generated_at, priority)
            if sig:
                signals.append(sig)
                priority += 1

        return signals

    def _item_to_signal(
        self,
        item: dict,
        action: str,
        run_id: str,
        generated_at: str,
        priority: int,
    ) -> Optional[Signal]:
        """Convert a portfolio_analysis or new_stocks item to a Signal."""
        ticker = item.get("ticker", "")
        if not ticker:
            return None

        alloc = item.get("allocation") or {}
        stops = item.get("stop_levels") or {}
        timing = item.get("timing") or {}

        # For TRIM we need shares_to_sell; for others shares_to_buy
        if action == "TRIM":
            shares = float(alloc.get("shares_to_sell") or alloc.get("shares_to_buy") or 0)
        else:
            shares = float(alloc.get("shares_to_buy") or alloc.get("shares_to_sell") or 0)

        return Signal(
            ticker=ticker,
            action=action,
            shares=shares,
            entry_price=float(alloc.get("entry_price_limit") or 0),
            take_profit=float(stops.get("take_profit") or 0),
            stop_loss=float(stops.get("stop_loss") or 0),
            priority=priority,
            risk_flags=[],
            signal_id=f"{run_id}:{ticker}:{action}",
            run_id=run_id,
            generated_at=generated_at,
            composite=float(item.get("composite") or 0),
            timing_price=float(timing.get("last_price") or 0),
            trim_pct=float(item.get("trim_percent") or 0),
            dollar_amount=float(alloc.get("dollar_amount") or 0),
        )

    # ------------------------------------------------------------------
    # Full portfolio
    # ------------------------------------------------------------------

    def get_portfolio(self) -> Portfolio:
        """Return the complete portfolio state from the most recent run."""
        data = self._load_run()
        return Portfolio(
            market=data.get("market", "us"),
            period=data.get("period", "short"),
            risk_profile=data.get("risk_profile", "aggressive"),
            capital=float(data.get("capital") or 0),
            cash=float(
                data.get("rebalance_budget")
                or data.get("deploy_now")
                or 0
            ),
            holdings=self.load_holdings(),
            signals=self.load_signals(),
            macro_context="",
            run_timestamp=data.get("datetime", ""),
            run_id=data.get("run_id", ""),
            token_usage=data.get("token_usage") or {},
        )

    # ------------------------------------------------------------------
    # Write-back (informational only — TDash is source of truth)
    # ------------------------------------------------------------------

    def update_cash(self, new_cash: float) -> bool:
        """Append a cash-update entry to a local log file."""
        log_file = self.data_dir / "cash_updates.log"
        try:
            with open(log_file, "a") as f:
                f.write(f"{datetime.now().isoformat()},{new_cash}\n")
            return True
        except OSError:
            return False
