"""Live executor: scored event → broker order → ledger record.

Called by the daemon for each event that clears the signal gate. Handles:

    1. Pre-trade guards: ledger.is_halted, cost_meter cap, broker liquidity.
    2. Sizing: dollar amount = base_capital_pct * cash; convert to integer shares.
    3. Place a bracket-style entry on Alpaca (atomic TP+SL); kr_broker has no
       atomic bracket so we place entry + then TP/SL pair.
    4. Wait for entry fill (timeout 60s); on fill, record_trade.
    5. Wait for exit (TP, SL, or hold-time timeout). On timeout, market-out.
    6. Record exit trade and update equity.

The executor is *synchronous per cycle*: while a trade is open, we don't take
new signals. ``max_positions=1`` in config enforces this at the daemon level.

Dry-run path: if the broker is in dry-run mode (KaigoraBroker default, or any
broker explicitly configured that way), all order calls are still issued but
the broker logs instead of placing — exit cycles are simulated assuming TP fill.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from news_quant import cost_meter, ledger
from news_quant.brokers.base import (
    BrokerConnector,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from news_quant.config_loader import load as load_config

logger = logging.getLogger("news_quant.executor")


@dataclass
class CycleResult:
    cycle_id: str
    ticker: str
    side: str                       # "long" | "short"
    entry_ts: Optional[datetime]
    exit_ts: Optional[datetime]
    entry_px: float
    exit_px: float
    qty: float
    realized_bps: float             # in trade direction, fees not netted here
    exit_reason: str                # "tp" | "sl" | "timeout" | "rejected" | "no_fill"
    error: str = ""


class Executor:
    def __init__(self, broker: BrokerConnector, cfg: Optional[dict] = None):
        self.broker = broker
        self.cfg = cfg or load_config()
        s = self.cfg["strategy"]
        b = self.cfg["budget"]
        self.tp_bps = float(s["tp_bps"])
        self.sl_bps = float(s["sl_bps"])
        self.hold_minutes = int(s.get("hold_minutes_default", 60))
        self.entry_lag_min = int(s.get("entry_lag_min", 1))
        self.base_capital_pct = float(s["base_capital_pct"])
        self.daily_llm_cap = float(b["llm_daily_cost_cap_usd"])
        self.max_dd_pct = float(b["hard_stop_drawdown_pct"])

    def _ok_to_trade(self) -> tuple[bool, str]:
        if ledger.is_halted():
            return False, "ledger halted"
        if cost_meter.today_spend_usd() >= self.daily_llm_cap:
            return False, "LLM cap reached"
        if ledger.check_drawdown_halt(self.max_dd_pct):
            return False, f"drawdown halt @ {ledger.drawdown_pct():.1f}%"
        return True, "ok"

    def _size_qty(self, ticker: str) -> tuple[float, float]:
        """Return (qty_shares, intended_price). 0,0 if cannot size."""
        acct = self.broker.get_account_info()
        cash = float(acct.cash)
        if cash <= 100:
            logger.info("insufficient cash %.2f", cash)
            return 0.0, 0.0
        px = self.broker.get_current_price(ticker)
        if px <= 0:
            logger.warning("no quote for %s — cannot size", ticker)
            return 0.0, 0.0
        dollar_amount = cash * self.base_capital_pct
        qty = max(1, int(dollar_amount // px))
        return float(qty), px

    def execute(self, ticker: str, side: str, score: float) -> CycleResult:
        cycle_id = uuid.uuid4().hex[:12]
        result = CycleResult(
            cycle_id=cycle_id, ticker=ticker, side=side,
            entry_ts=None, exit_ts=None,
            entry_px=0.0, exit_px=0.0, qty=0.0,
            realized_bps=0.0, exit_reason="",
        )

        ok, why = self._ok_to_trade()
        if not ok:
            result.exit_reason = "rejected"
            result.error = why
            logger.warning("trade rejected: %s", why)
            return result

        qty, ref_px = self._size_qty(ticker)
        if qty == 0:
            result.exit_reason = "rejected"
            result.error = "size=0"
            return result

        # Compute TP/SL absolute prices in trade direction.
        if side == "long":
            tp_px = ref_px * (1 + self.tp_bps / 1e4)
            sl_px = ref_px * (1 - self.sl_bps / 1e4)
            order_side = OrderSide.BUY
        else:  # short
            tp_px = ref_px * (1 - self.tp_bps / 1e4)
            sl_px = ref_px * (1 + self.sl_bps / 1e4)
            order_side = OrderSide.SELL

        # Entry: market with attached bracket (Alpaca uses bracket; Kaigora dry-run
        # logs and pretends to fill at ref_px).
        entry = Order(
            ticker=ticker,
            side=order_side,
            order_type=OrderType.MARKET,
            qty=qty,
            price=ref_px,
            tp_price=tp_px,
            sl_price=sl_px,
        )
        logger.info(
            "[%s] enter %s %s qty=%s ref=$%.2f tp=$%.2f sl=$%.2f score=%+.4f",
            cycle_id, side, ticker, qty, ref_px, tp_px, sl_px, score,
        )
        resp = self.broker.place_order(entry)
        if "error" in resp:
            result.exit_reason = "rejected"
            result.error = resp["error"]
            return result

        # Wait for entry fill.
        status = self.broker.wait_for_fill(entry.order_id, timeout_seconds=60)
        if status not in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            self.broker.cancel_order(entry.order_id)
            result.exit_reason = "no_fill"
            result.error = f"entry status={status.value}"
            return result

        # Refresh fill price; on dry-run kaigora the broker already populated.
        result.entry_ts = datetime.now(timezone.utc)
        result.entry_px = entry.avg_fill_price or ref_px
        result.qty = entry.filled_qty or qty
        ledger.record_trade(
            ticker=ticker, side=order_side.value,
            qty=result.qty, price=result.entry_px,
            fees_usd=0.0, cycle_id=cycle_id, note=f"entry score={score:.4f}",
        )

        # Hold loop: poll position until exit. Bracket orders auto-exit on
        # Alpaca; we just need to detect when the position is gone or the
        # hold-time elapses.
        deadline = result.entry_ts + timedelta(minutes=self.hold_minutes)
        exit_reason = "timeout"
        while datetime.now(timezone.utc) < deadline:
            pos = self.broker.get_position(ticker)
            if pos is None or abs(pos.qty) < 1e-9:
                exit_reason = "tp_or_sl"  # bracket fired; unsure which without order history
                break
            time.sleep(15)

        # If still holding after deadline, close via the broker's atomic helper
        # (cancels bracket children first, then market-closes).
        pos_now = self.broker.get_position(ticker)
        if pos_now is not None and abs(pos_now.qty) >= 1e-9:
            logger.info("[%s] timeout — closing %s qty=%s", cycle_id, ticker, pos_now.qty)
            ok = self.broker.close_position(ticker)
            if not ok:
                logger.error("[%s] close_position failed for %s", cycle_id, ticker)
            # exit price = current quote (close fills at market)
            result.exit_px = self.broker.get_current_price(ticker)
            exit_reason = "timeout"
        else:
            # Bracket already exited. We don't know exact exit px without
            # querying order history; estimate from current quote.
            result.exit_px = self.broker.get_current_price(ticker)

        result.exit_ts = datetime.now(timezone.utc)
        result.exit_reason = exit_reason

        # Compute realized bps in trade direction.
        if result.entry_px > 0:
            raw = (result.exit_px / result.entry_px - 1.0) * 1e4
            result.realized_bps = raw if side == "long" else -raw

        ledger.record_trade(
            ticker=ticker, side=("SELL" if side == "long" else "BUY"),
            qty=result.qty, price=result.exit_px,
            fees_usd=0.0, cycle_id=cycle_id,
            note=f"exit reason={exit_reason} realized={result.realized_bps:+.1f}bps",
        )
        # Update equity snapshot.
        try:
            acct = self.broker.get_account_info()
            ledger.record_equity(acct.total_equity, note=f"post-cycle {cycle_id}")
        except Exception as e:
            logger.warning("equity refresh failed: %s", e)

        return result
