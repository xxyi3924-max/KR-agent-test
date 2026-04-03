"""
TDash Connector — Reads signals and state from Trading Dashboard v9
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class Signal:
    """Trading signal from TDash v9"""
    ticker: str
    action: str          # HOLD, ADD, NEW, SELL, TRIM
    shares: float        # Number of shares
    entry_price: float   # Target entry price
    take_profit: float   # TP price
    stop_loss: float     # SL price
    priority: int        # AI ranking (1 = highest)
    risk_flags: List[str]
    notes: str = ""


@dataclass
class Holding:
    """Current holding from TDash"""
    ticker: str
    shares: float
    avg_cost: float
    entry_date: str
    current_value: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Portfolio:
    """Full portfolio state from TDash"""
    market: str
    period: str
    risk_profile: str
    capital: float
    cash: float
    holdings: List[Holding]
    signals: List[Signal]
    macro_context: str
    run_timestamp: str
    token_usage: Dict


class TDashConnector:
    """Connect to TDash v9 data files"""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(os.path.expanduser(config['tdash']['data_dir']))
        self.user = config['tdash']['user']
        self.runs_dir = self.data_dir / "runs"
        
    def get_latest_run_dir(self) -> Path:
        """Find most recent trading run"""
        # Find most recent run by timestamp in folder name
        runs = sorted(self.runs_dir.glob("*-*"), reverse=True)
        if runs:
            return runs[0]
        raise FileNotFoundError(f"No runs found in {self.runs_dir}")
    
    def load_holdings(self) -> List[Holding]:
        """Load current holdings from live_holdings.txt"""
        holdings_file = self.data_dir / "live_holdings.txt"
        
        holdings = []
        if holdings_file.exists():
            with open(holdings_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) >= 4:
                        holdings.append(Holding(
                            ticker=parts[0].strip(),
                            shares=float(parts[1]),
                            avg_cost=float(parts[2]),
                            entry_date=parts[3].strip()
                        ))
        return holdings
    
    def load_signals(self) -> List[Signal]:
        """Load trading signals from run directory"""
        run_dir = self.get_latest_run_dir()
        
        # Look for signals file (may vary by run)
        signals = []
        
        # Try reading from run summary if available
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
                signals = self._parse_signals(data)
        
        # If no signals file, parse from live_holdings annotations
        # or construct from holdings data
        return signals
    
    def _parse_signals(self, data: dict) -> List[Signal]:
        """Parse signals from run data"""
        signals = []
        
        # Parse from "top_picks" format
        if "top_picks" in data:
            for i, pick in enumerate(data["top_picks"]):
                signals.append(Signal(
                    ticker=pick.get("ticker", ""),
                    action="NEW" if i > 2 else "ADD",
                    shares=pick.get("suggested_shares", 0),
                    entry_price=pick.get("entry_price", 0),
                    take_profit=pick.get("take_profit", 0),
                    stop_loss=pick.get("stop_loss", 0),
                    priority=i + 1,
                    risk_flags=pick.get("risk_flags", []),
                    notes=pick.get("notes", "")
                ))
        
        # Parse from holdings actions
        if "actions" in data:
            for action in data["actions"]:
                signals.append(Signal(
                    ticker=action.get("ticker", ""),
                    action=action.get("type", "HOLD"),
                    shares=action.get("shares", 0),
                    entry_price=action.get("entry_price", 0),
                    take_profit=action.get("take_profit", 0),
                    stop_loss=action.get("stop_loss", 0),
                    priority=action.get("priority", 99),
                    risk_flags=action.get("risk_flags", [])
                ))
        
        return signals
    
    def get_portfolio(self) -> Portfolio:
        """Get full portfolio state"""
        run_dir = self.get_latest_run_dir()
        
        # Load holdings
        holdings = self.load_holdings()
        
        # Load signals
        signals = self.load_signals()
        
        # Load run metadata
        run_info = {}
        meta_file = run_dir / "run_info.json"
        if meta_file.exists():
            with open(meta_file) as f:
                run_info = json.load(f)
        
        return Portfolio(
            market=run_info.get("market", "US"),
            period=run_info.get("period", "short"),
            risk_profile=run_info.get("risk_profile", "aggressive"),
            capital=run_info.get("capital", 0),
            cash=run_info.get("cash", 0),
            holdings=holdings,
            signals=signals,
            macro_context=run_info.get("macro_context", ""),
            run_timestamp=run_info.get("timestamp", ""),
            token_usage=run_info.get("token_usage", {})
        )
    
    def update_cash(self, new_cash: float) -> bool:
        """Update cash in TDash (simple write)"""
        holdings_file = self.data_dir / "users" / self.user / "live_holdings.txt"
        
        # For now, just append to a log file
        log_file = self.data_dir / "users" / self.user / "cash_updates.log"
        timestamp = datetime.now().isoformat()
        
        with open(log_file, "a") as f:
            f.write(f"{timestamp},{new_cash}\n")
        
        return True
    
    def update_holdings(self, holdings: List[Holding]) -> bool:
        """Update holdings in TDash (simple write)"""
        holdings_file = self.data_dir / "users" / self.user / "live_holdings.txt"
        
        lines = ["# Ticker,Shares,AvgCost,EntryDate"]
        for h in holdings:
            lines.append(f"{h.ticker},{h.shares},{h.avg_cost},{h.entry_date}")
        
        with open(holdings_file, "w") as f:
            f.write("\n".join(lines))
        
        return True
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices from TDash data"""
        prices = {}
        
        # Try to read from latest market data
        prices_file = self.data_dir / "market_prices.json"
        if prices_file.exists():
            with open(prices_file) as f:
                market_data = json.load(f)
                for ticker in tickers:
                    if ticker in market_data:
                        prices[ticker] = market_data[ticker].get("price", 0)
        
        return prices
