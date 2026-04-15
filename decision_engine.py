"""
Decision Engine — orchestrates the TDash → KR Broker pipeline.

Execution flow each cycle:
  1. Load agent_state (last_run_id, tp_sl_pairs)
  2. Check market hours (skip if closed)
  3. Load TDash signals; skip if run_id already processed
  4. Check signal staleness
  5. Fetch current account from KR Broker
  6. For each actionable signal:
       - score filter: composite >= confidence_threshold
       - smart order routing: limit if price not reached, market if better
       - execute order
       - on BUY fill: place TP/SL pair, store in state
  7. Cancel orphaned TP/SL legs (when one fills, cancel other)
  8. Auto-invest free cash > threshold into QQQ/SPY
  9. Persist state
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import yaml

from kr_broker import (
    KRBrokerConnector,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from mcda_engine import MCDAMomentumEngine
from tdash_connector import Portfolio, Signal, TDashConnector

# ETFMonitor is imported lazily to avoid circular dependency issues
# at module load time; type hint only here.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from etf_monitor import ETFMonitor

logger = logging.getLogger(__name__)

STATE_FILE = "agent_state.json"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExecutionPlan:
    ticker: str
    action: str           # BUY | SELL | TRIM
    qty: float
    order_type: OrderType
    price: float          # limit price (0 for market)
    dollar_amount: float = 0.0
    tp_price: float = 0.0
    sl_price: float = 0.0
    trigger_price: float = 0.0


@dataclass
class ExecutionResult:
    ticker: str
    action: str
    qty: float
    fill_price: float
    success: bool
    message: str = ""
    order_id: str = ""


# ---------------------------------------------------------------------------
# Agent state persistence
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    """Load persisted agent state (run_id tracker + TP/SL pair map)."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_run_id": "", "tp_sl_pairs": {}, "entry_dates": {}}


def _save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        logger.warning(f"Could not save agent state: {e}")


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def _is_market_open() -> bool:
    """
    US market hours: Mon–Fri 09:30–16:00 ET.
    Approximated as 14:30–21:00 UTC (ignores DST; ±1h acceptable for this use).
    """
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:           # Saturday=5, Sunday=6
        return False
    open_h, open_m = 14, 30
    close_h, close_m = 21, 0
    after_open = (now.hour, now.minute) >= (open_h, open_m)
    before_close = (now.hour, now.minute) < (close_h, close_m)
    return after_open and before_close


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Orchestrates the full signal-to-execution pipeline.
    Reads config.yaml; instantiates TDashConnector and KRBrokerConnector.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        tdash_cfg = self.config.get("tdash", {})
        self.tdash = TDashConnector(
            data_dir=os.path.expanduser(tdash_cfg.get("data_dir", ".")),
            user=tdash_cfg.get("user", "admin"),
        )
        # Portfolio sync: write bot state back to TDash after each cycle
        # so future TDash runs see the correct holdings and free cash budget.
        bot_user = tdash_cfg.get("bot_user", "")
        quant_data = os.path.expanduser(tdash_cfg.get("quant_data_dir", "~/quant_data"))
        self._portfolio_sync_file: Optional[str] = (
            os.path.join(quant_data, "users", bot_user, "portfolio_us.json")
            if bot_user else None
        )
        self._portfolio_period = tdash_cfg.get("period", "short")
        self._portfolio_risk = tdash_cfg.get("risk_profile", "aggressive")

        broker_cfg = self.config.get("kr_broker", {})
        self.broker = KRBrokerConnector(
            api_url=broker_cfg.get("api_url", "http://localhost:8084"),
            api_key=broker_cfg.get("api_key", ""),
        )

        self.mcda = MCDAMomentumEngine(self.config)

        trading_cfg = self.config.get("trading", {})
        self.free_cash_threshold = trading_cfg.get("free_cash_threshold", 10_000)
        self.etf_preference = trading_cfg.get("etf_preference", "QQQ")
        self.etf_alternate = trading_cfg.get("etf_alternate", "SPY")
        self.max_slippage_pct = trading_cfg.get("max_slippage_pct", 2.0)
        self.tp_enabled = trading_cfg.get("tp_enabled", True)
        self.sl_enabled = trading_cfg.get("sl_enabled", True)
        # Signal selection rules
        self.max_positions = trading_cfg.get("max_positions", 10)
        self.max_new_per_cycle = trading_cfg.get("max_new_per_cycle", 2)
        self.min_position_dollars = trading_cfg.get("min_position_dollars", 8_000)

        # TDash composite score threshold for ADD/NEW signals.
        # SELL/TRIM always pass regardless of composite.
        # Kept in trading section; LLM confidence_threshold is unrelated.
        self.composite_threshold = trading_cfg.get("composite_threshold", 0.60)

        agent_cfg = self.config.get("agent", {})
        self.signal_refresh_days = agent_cfg.get("signal_refresh_days", 3)

        # Set by agent.py after ETFMonitor is constructed; if set, passive
        # auto-invest is skipped in favour of the real-time monitor.
        self.etf_monitor: Optional["ETFMonitor"] = None

        logger.info("DecisionEngine initialised")

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> dict:
        summary: dict = {
            "timestamp": datetime.now().isoformat(),
            "skipped": None,
            "run_id": None,
            "plans_created": 0,
            "executions": [],
            "auto_invest": [],
            "cash_liquidations": [],
            "errors": [],
        }

        state = _load_state()

        # 1. Market hours guard
        if not _is_market_open():
            msg = "Market closed — skipping execution"
            logger.info(msg)
            summary["skipped"] = msg
            # Still sync portfolio so TDash always has current data
            try:
                self._sync_portfolio_to_tdash(self.broker.get_account_info())
            except Exception as e:
                logger.warning(f"Portfolio sync (market closed): {e}")
            return summary

        # 2. Load TDash portfolio
        try:
            portfolio = self.tdash.get_portfolio()
        except Exception as e:
            logger.error(f"Failed to load TDash portfolio: {e}")
            summary["errors"].append(f"TDash load: {e}")
            return summary

        summary["run_id"] = portfolio.run_id

        # 3. Idempotency: skip if this run was already processed
        if portfolio.run_id and portfolio.run_id == state.get("last_run_id"):
            msg = f"Run {portfolio.run_id} already processed — skipping"
            logger.info(msg)
            summary["skipped"] = msg
            # Still run cash management even if signals are stale
            self._manage_free_cash(summary)
            return summary

        # 4. Filter signals (staleness + confidence)
        signals = self._filter_signals(portfolio.signals)

        if not signals:
            logger.info("No actionable signals after filtering")
            summary["skipped"] = "No actionable signals"
            summary["signals"] = {"total": len(portfolio.signals), "actionable": 0, "actions": []}
            self._manage_free_cash(summary)
            return summary

        # 5. Fetch account

        try:
            account = self.broker.get_account_info()
            summary["account"] = {
                "cash": account.cash,
                "equity": account.total_equity,
                "positions": len(account.positions),
            }
        except Exception as e:
            logger.error(f"Failed to fetch account: {e}")
            summary["errors"].append(f"Account fetch: {e}")
            return summary

        # 6. Select signals (position cap, new-per-cycle cap, min size, tradable check)
        tradable = self.broker.get_tradable_tickers()
        pending_buys = state.get("pending_buys", {})
        signals, selection_log = self._select_signals(signals, account, tradable, pending_buys)
        summary["signals"] = {
            "total": len(portfolio.signals),
            "actionable": len(signals),
            "actions": list({s.action for s in signals}),
            "selection": selection_log,
        }
        if not signals:
            logger.info("No signals selected after position/size rules")
            summary["skipped"] = "No signals passed selection rules"
            self._manage_free_cash(summary)
            return summary

        # 7. Cancel orphaned TP/SL legs before placing new ones
        self._reconcile_tp_sl_pairs(state, account)

        # 8. Build and execute plans
        plans = self._build_plans(signals, account)
        summary["plans_created"] = len(plans)

        if plans:
            # Check if we need to raise cash for BUY orders.
            # Subtract expected SELL/TRIM proceeds so we don't over-liquidate ETFs.
            required_cash = sum(
                p.dollar_amount or (p.qty * p.price)
                for p in plans if p.action == "BUY"
            )
            sell_proceeds = sum(
                p.qty * p.price
                for p in plans if p.action == "SELL" and p.price > 0
            )
            net_cash_needed = required_cash - sell_proceeds
            if account.cash < net_cash_needed * 1.02:  # 2% buffer
                shortfall = net_cash_needed * 1.02 - account.cash
                logger.info(f"Cash shortfall ${shortfall:.2f} (buy ${required_cash:.0f} - sell proceeds ${sell_proceeds:.0f}) — liquidating QQQ/SPY")
                liq_results = self._liquidate_etf_for_cash(shortfall, account)

                summary["cash_liquidations"] = [
                    {"ticker": r.ticker, "qty": r.qty, "raised": r.fill_price * r.qty}
                    for r in liq_results
                ]

            results = self._execute_plans(plans, state)
            summary["executions"] = [
                {
                    "action": r.action,
                    "ticker": r.ticker,
                    "qty": r.qty,
                    "success": r.success,
                    "message": r.message,
                }
                for r in results
            ]

        # 9. Auto-invest free cash
        self._manage_free_cash(summary)

        # 10. Persist state with updated run_id
        state["last_run_id"] = portfolio.run_id
        _save_state(state)

        # 11. Sync portfolio back to TDash bot account
        try:
            account2 = self.broker.get_account_info()
            self.tdash.update_cash(account2.cash)
            self._sync_portfolio_to_tdash(account2)
        except Exception as e:
            logger.warning(f"Portfolio sync failed: {e}")

        return summary

    # ------------------------------------------------------------------
    # Signal filtering
    # ------------------------------------------------------------------

    def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Keep signals that pass staleness and composite checks.

        SELL / TRIM : staleness check only — exits are never blocked by score.
        ADD / NEW   : staleness + composite >= composite_threshold.
        HOLD        : always dropped (TDashConnector skips them for
                      portfolio_analysis; new_stocks HOLD are caught here).
        """
        max_age_hours = self.signal_refresh_days * 24.0
        filtered = []
        for sig in signals:
            if sig.action == "HOLD":
                continue
            if self.tdash.is_signal_stale(sig, max_age_hours=max_age_hours):
                logger.warning(f"Stale signal dropped: {sig.ticker} {sig.action}")
                continue
            if sig.action not in ("SELL", "TRIM") and sig.composite < self.composite_threshold:
                logger.info(
                    f"Low composite dropped: {sig.ticker} {sig.action} "
                    f"{sig.composite:.3f} < {self.composite_threshold}"
                )
                continue
            filtered.append(sig)
        return filtered

    # ------------------------------------------------------------------
    # Signal selection (position cap, new-per-cycle, min size)
    # ------------------------------------------------------------------

    def _select_signals(
        self, signals: List[Signal], account, tradable: set = None,
        pending_buys: dict = None,
    ) -> Tuple[List[Signal], List[dict]]:
        """
        Apply portfolio construction rules after confidence filtering.

        SELL / TRIM → always selected (exits never blocked).

        ADD + NEW   → ranked together by composite score (highest first),
                      then each checked in order:
                        1. ticker already has a pending limit buy → skip (no double-order)
                        2. ticker in Kaigora tradable universe (if known)
                        3. size >= min_position_dollars
                        4. NEW only: total positions < max_positions
                        5. NEW only: new names this cycle < max_new_per_cycle
                      ADD does not consume a position slot (it's an existing name).
                      NEW consumes one slot and one new-per-cycle token.

        Returns (selected_signals, selection_log).
        """
        log: List[dict] = []
        selected: List[Signal] = []
        new_count = 0

        # SELLs free a slot this cycle — count optimistically
        sells_this_cycle = sum(1 for s in signals if s.action == "SELL")
        effective_count = len(account.positions) - sells_this_cycle

        # ── Step 1: exits always pass ─────────────────────────────────
        for sig in signals:
            if sig.action in ("SELL", "TRIM"):
                selected.append(sig)
                log.append({"ticker": sig.ticker, "action": sig.action,
                            "composite": sig.composite, "result": "selected",
                            "reason": "exit — always executed"})

        # ── Step 2: entries ranked by composite score (best first) ────
        entries = sorted(
            [s for s in signals if s.action in ("ADD", "NEW")],
            key=lambda s: -s.composite,
        )

        for sig in entries:
            size = sig.dollar_amount or (sig.shares * sig.entry_price)
            tag = f"composite={sig.composite:.2f} ${size:.0f}"

            # Check 1: skip if a limit buy for this ticker is already pending fill
            if pending_buys and sig.ticker in pending_buys:
                log.append({"ticker": sig.ticker, "action": sig.action,
                            "composite": sig.composite, "result": "dropped",
                            "reason": "limit buy already pending — no double order"})
                continue

            # Check 2: tradable on Kaigora (skip if universe known and ticker absent)
            if tradable and sig.ticker not in tradable:
                log.append({"ticker": sig.ticker, "action": sig.action,
                            "composite": sig.composite, "result": "dropped",
                            "reason": "not in Kaigora asset universe"})
                continue

            # Check 2: minimum position size (applies to both ADD and NEW)
            if size < self.min_position_dollars:
                log.append({"ticker": sig.ticker, "action": sig.action,
                            "composite": sig.composite, "result": "dropped",
                            "reason": f"size ${size:.0f} < min ${self.min_position_dollars:.0f}"})
                continue

            if sig.action == "NEW":
                # Check 3: position cap
                if effective_count + new_count >= self.max_positions:
                    log.append({"ticker": sig.ticker, "action": "NEW",
                                "composite": sig.composite, "result": "dropped",
                                "reason": f"position cap {self.max_positions} reached"})
                    continue
                # Check 4: new-per-cycle cap
                if new_count >= self.max_new_per_cycle:
                    log.append({"ticker": sig.ticker, "action": "NEW",
                                "composite": sig.composite, "result": "dropped",
                                "reason": f"new-per-cycle cap {self.max_new_per_cycle} reached"})
                    continue
                new_count += 1

            selected.append(sig)
            log.append({"ticker": sig.ticker, "action": sig.action,
                        "composite": sig.composite, "result": "selected",
                        "reason": tag})

        for entry in log:
            level = logging.INFO if entry["result"] == "selected" else logging.WARNING
            logger.log(
                level,
                f"Select {entry['action']:4s} {entry['ticker']:6s} "
                f"[{entry['result']}] {entry['reason']}"
            )

        return selected, log

    # ------------------------------------------------------------------
    # Plan building
    # ------------------------------------------------------------------

    def _build_plans(
        self, signals: List[Signal], account
    ) -> List[ExecutionPlan]:
        plans = []
        positions_by_ticker = {p.ticker: p for p in account.positions}

        for sig in signals:
            plan = self._signal_to_plan(sig, positions_by_ticker)
            if plan:
                plans.append(plan)

        return plans

    def _signal_to_plan(
        self, sig: Signal, positions_by_ticker: Dict[str, Position]
    ) -> Optional[ExecutionPlan]:
        """Convert a Signal to an ExecutionPlan with smart order routing."""
        action = sig.action  # ADD | NEW | TRIM | SELL

        if action in ("ADD", "NEW"):
            return self._plan_buy(sig)

        elif action == "SELL":
            position = positions_by_ticker.get(sig.ticker)
            if not position:
                logger.warning(f"SELL signal for {sig.ticker} but no position found — skipping")
                return None
            return self._plan_sell(sig, position.qty)

        elif action == "TRIM":
            position = positions_by_ticker.get(sig.ticker)
            if not position:
                logger.warning(f"TRIM signal for {sig.ticker} but no position found — skipping")
                return None
            # Use TDash trim_pct if provided, otherwise fall back to signal.shares
            if sig.trim_pct > 0:
                trim_qty = position.qty * sig.trim_pct / 100.0
            elif sig.shares > 0:
                trim_qty = sig.shares
            else:
                logger.warning(f"TRIM for {sig.ticker}: no trim_pct or shares — skipping")
                return None
            trim_qty = min(trim_qty, position.qty)  # can't sell more than we have
            return self._plan_sell(sig, trim_qty)

        return None

    def _plan_buy(self, sig: Signal) -> Optional[ExecutionPlan]:
        """Smart order routing for BUY: market if current price <= entry, limit otherwise."""
        if sig.shares <= 0:
            logger.warning(f"BUY signal for {sig.ticker} has shares=0 — skipping")
            return None

        try:
            current_price = self.broker.get_current_price(sig.ticker)
        except ValueError as e:
            logger.error(f"Cannot plan BUY for {sig.ticker}: {e}")
            return None

        entry = sig.entry_price or current_price

        # Gap check
        if entry > 0:
            gap_pct = abs(current_price - entry) / entry * 100
            if gap_pct > self.max_slippage_pct:
                logger.warning(
                    f"{sig.ticker}: price gap {gap_pct:.1f}% > {self.max_slippage_pct}% — proceeding"
                )

        # Smart routing
        if current_price <= entry:
            # Current price is at or better than entry → market order
            dollar_amount = sig.dollar_amount or (sig.shares * current_price)
            return ExecutionPlan(
                ticker=sig.ticker,
                action="BUY",
                qty=sig.shares,
                order_type=OrderType.MARKET,
                price=current_price,
                dollar_amount=dollar_amount,
                tp_price=sig.take_profit,
                sl_price=sig.stop_loss,
            )
        else:
            # Price not yet at entry → limit order
            return ExecutionPlan(
                ticker=sig.ticker,
                action="BUY",
                qty=sig.shares,
                order_type=OrderType.LIMIT,
                price=entry,
                dollar_amount=sig.dollar_amount,
                tp_price=sig.take_profit,
                sl_price=sig.stop_loss,
            )

    def _plan_sell(self, sig: Signal, qty: float) -> Optional[ExecutionPlan]:
        """SELL / TRIM → always market order (intentional exit)."""
        if qty <= 0:
            return None
        return ExecutionPlan(
            ticker=sig.ticker,
            action="SELL",
            qty=qty,
            order_type=OrderType.MARKET,
            price=0.0,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_plans(
        self, plans: List[ExecutionPlan], state: dict
    ) -> List[ExecutionResult]:
        results = []
        for plan in plans:
            result = self._execute_one(plan, state)
            results.append(result)
        return results

    def _execute_one(self, plan: ExecutionPlan, state: dict) -> ExecutionResult:
        try:
            order = Order(
                ticker=plan.ticker,
                side=OrderSide.BUY if plan.action == "BUY" else OrderSide.SELL,
                order_type=plan.order_type,
                qty=plan.qty,
                price=plan.price,
                trigger_price=plan.trigger_price,
                dollar_amount=plan.dollar_amount,
                tp_price=plan.tp_price,
                sl_price=plan.sl_price,
            )

            api_result = self.broker.place_order(order)
            order_id = str(api_result.get("id") or api_result.get("orderIds", [""])[0])

            # Record entry date for portfolio sync
            if plan.action == "BUY":
                state.setdefault("entry_dates", {})[plan.ticker] = (
                    datetime.now().strftime("%Y-%m-%d")
                )

            if plan.action == "BUY" and (plan.tp_price or plan.sl_price):
                if plan.order_type == OrderType.MARKET:
                    # Market orders fill immediately — place TP/SL now
                    self._place_and_store_tp_sl(plan, plan.price, state)
                else:
                    # Limit order — defer TP/SL until fill confirmed in background_check
                    state.setdefault("pending_buys", {})[plan.ticker] = {
                        "qty": plan.qty,
                        "tp_price": plan.tp_price,
                        "sl_price": plan.sl_price,
                        "limit_price": plan.price,
                        "placed_at": datetime.now().isoformat(),
                    }
                    logger.info(f"Limit BUY {plan.ticker} queued — TP/SL deferred to fill")

            return ExecutionResult(
                ticker=plan.ticker,
                action=plan.action,
                qty=plan.qty,
                fill_price=plan.price,
                success=True,
                message=f"Order placed: {api_result}",
                order_id=order_id,
            )

        except Exception as e:
            logger.error(f"Failed to execute {plan.action} {plan.ticker}: {e}")
            return ExecutionResult(
                ticker=plan.ticker,
                action=plan.action,
                qty=plan.qty,
                fill_price=plan.price,
                success=False,
                message=str(e),
            )

    # ------------------------------------------------------------------
    # TP/SL management
    # ------------------------------------------------------------------

    def _place_and_store_tp_sl(
        self, plan: ExecutionPlan, fill_price: float, state: dict
    ) -> None:
        """Place TP + SL conditional orders and record their IDs in state."""
        if not (self.tp_enabled or self.sl_enabled):
            return

        tp_price = plan.tp_price if self.tp_enabled else 0
        sl_price = plan.sl_price if self.sl_enabled else 0
        ref_price = fill_price or plan.price

        if not ref_price:
            logger.warning(f"Cannot set TP/SL for {plan.ticker}: no reference price")
            return

        try:
            tp_id, sl_id = self.broker.place_tp_sl_pair(
                ticker=plan.ticker,
                qty=plan.qty,
                take_profit=tp_price,
                stop_loss=sl_price,
                current_price=ref_price,
            )
            if tp_id or sl_id:
                state.setdefault("tp_sl_pairs", {})[plan.ticker] = {
                    "tp_id": tp_id,
                    "sl_id": sl_id,
                    "qty": plan.qty,
                }
                logger.info(f"TP/SL stored for {plan.ticker}: tp={tp_id} sl={sl_id}")
        except Exception as e:
            logger.error(f"Failed to place TP/SL for {plan.ticker}: {e}")

    def _reconcile_tp_sl_pairs(self, state: dict, account) -> None:
        """Cancel the orphaned TP or SL leg when a position is gone."""
        pairs: dict = state.get("tp_sl_pairs", {})
        held_tickers = {p.ticker for p in account.positions}
        to_remove = []

        for ticker, pair in pairs.items():
            if ticker not in held_tickers:
                for leg_key in ("tp_id", "sl_id"):
                    leg_id = pair.get(leg_key)
                    if leg_id is not None:
                        logger.info(f"Cancelling orphaned {leg_key} {leg_id} for {ticker}")
                        self.broker.cancel_conditional_order(leg_id)
                to_remove.append(ticker)

        for ticker in to_remove:
            del pairs[ticker]

    # ------------------------------------------------------------------
    # Background check (every 5 min) — fill detection + OCO cleanup
    # ------------------------------------------------------------------

    def background_check(self) -> None:
        """
        Called every 5 minutes by the scheduler between main cycles.
        Four jobs:
          0. Detect new TDash run → trigger intraday cycle immediately
          1. Detect filled LIMIT BUY orders → place TP/SL
          2. Escalate stale limit orders to market after 1 hour
          3. Cancel orphaned TP/SL legs (OCO simulation)
        """
        if not _is_market_open():
            return

        state = _load_state()

        # Job 0: intraday signal pickup — if a new TDash run exists, run the
        # full cycle now rather than waiting for the next scheduled run_once.
        try:
            latest_run_id = self.tdash.get_portfolio().run_id
        except Exception as e:
            logger.warning(f"background_check: TDash peek failed: {e}")
            latest_run_id = None

        if latest_run_id and latest_run_id != state.get("last_run_id"):
            logger.info(
                f"New TDash run detected in background: {latest_run_id} — triggering intraday cycle"
            )
            self.run_cycle()
            # Guard: if run_cycle exited early (no signals, broker down, etc.)
            # it may not have saved last_run_id.  Stamp it now so the same run
            # does not re-trigger on every subsequent 5-min background tick.
            fresh_state = _load_state()
            if fresh_state.get("last_run_id") != latest_run_id:
                fresh_state["last_run_id"] = latest_run_id
                _save_state(fresh_state)
            return

        changed = False

        try:
            account = self.broker.get_account_info()
        except Exception as e:
            logger.warning(f"background_check: account fetch failed: {e}")
            return

        held = {p.ticker: p for p in account.positions}
        pending_buys: dict = state.get("pending_buys", {})
        to_remove = []

        for ticker, info in pending_buys.items():
            pos = held.get(ticker)
            expected_qty = info["qty"]
            placed_at = datetime.fromisoformat(info["placed_at"])
            age_minutes = (datetime.now() - placed_at).total_seconds() / 60

            if pos and pos.qty >= expected_qty * 0.90:
                # Filled (accept ≥90% for partial fills)
                logger.info(f"LIMIT BUY {ticker} confirmed filled — placing TP/SL")
                plan = ExecutionPlan(
                    ticker=ticker,
                    action="BUY",
                    qty=pos.qty,
                    order_type=OrderType.LIMIT,
                    price=pos.avg_cost,
                    tp_price=info["tp_price"],
                    sl_price=info["sl_price"],
                )
                self._place_and_store_tp_sl(plan, pos.avg_cost, state)
                to_remove.append(ticker)
                changed = True

            elif age_minutes >= 60:
                # 1-hour timeout — escalate to market
                logger.info(
                    f"LIMIT BUY {ticker} unfilled after {age_minutes:.0f}m — escalating to market"
                )
                try:
                    escalation = Order(
                        ticker=ticker,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        qty=expected_qty,
                        price=info["limit_price"],
                        dollar_amount=expected_qty * info["limit_price"],
                        tp_price=info["tp_price"],
                        sl_price=info["sl_price"],
                    )
                    self.broker.place_order(escalation)
                    state.setdefault("entry_dates", {})[ticker] = (
                        datetime.now().strftime("%Y-%m-%d")
                    )
                    # For market orders assume fill — place TP/SL immediately
                    plan = ExecutionPlan(
                        ticker=ticker, action="BUY", qty=expected_qty,
                        order_type=OrderType.MARKET, price=info["limit_price"],
                        tp_price=info["tp_price"], sl_price=info["sl_price"],
                    )
                    self._place_and_store_tp_sl(plan, info["limit_price"], state)
                    to_remove.append(ticker)
                    changed = True
                except Exception as e:
                    logger.error(f"Escalation failed for {ticker}: {e}")

        for ticker in to_remove:
            del pending_buys[ticker]
        if to_remove:
            state["pending_buys"] = pending_buys
            # Re-fetch account so reconciliation sees positions filled by
            # escalated market orders above (avoids cancelling fresh TP/SL).
            try:
                account = self.broker.get_account_info()
            except Exception as e:
                logger.warning(f"background_check: account re-fetch after escalation failed: {e}")

        # OCO cleanup
        before = len(state.get("tp_sl_pairs", {}))
        self._reconcile_tp_sl_pairs(state, account)
        if len(state.get("tp_sl_pairs", {})) != before:
            changed = True

        if changed:
            _save_state(state)

        # Keep TDash bot portfolio in sync every background tick
        try:
            self._sync_portfolio_to_tdash(account)
        except Exception as e:
            logger.warning(f"Portfolio sync (background): {e}")

    # ------------------------------------------------------------------
    # Free cash management (MCDA: QQQ/SPY only)
    # ------------------------------------------------------------------

    def _get_etf_market_value(self, account) -> float:
        """Sum current market value of QQQ + SPY positions."""
        etf_tickers = {self.etf_preference, self.etf_alternate}
        total = 0.0
        for pos in account.positions:
            if pos.ticker in etf_tickers:
                price = pos.current_price or pos.avg_cost
                total += pos.qty * price
        return total

    def get_total_free_cash(self, account) -> float:
        """
        Total deployable capital = cash + ETF market value.
        This is what gets reported to TDash as 'budget'.
        """
        return account.cash + self._get_etf_market_value(account)

    def _pick_best_etf(self) -> str:
        """
        Use MCDA momentum score to choose between QQQ and SPY.
        Falls back to etf_preference if price history is insufficient.
        """
        try:
            pref_hist = self.broker.get_price_history(self.etf_preference)
            alt_hist = self.broker.get_price_history(self.etf_alternate)
            if len(pref_hist) >= 5 and len(alt_hist) >= 5:
                pref_score = self.mcda.score_position(
                    self.etf_preference, pref_hist
                ).total_score
                alt_score = self.mcda.score_position(
                    self.etf_alternate, alt_hist
                ).total_score
                chosen = self.etf_preference if pref_score >= alt_score else self.etf_alternate
                logger.info(
                    f"ETF selection: {self.etf_preference}={pref_score:.1f} "
                    f"{self.etf_alternate}={alt_score:.1f} → {chosen}"
                )
                return chosen
        except Exception as e:
            logger.warning(f"ETF MCDA selection failed: {e}")
        return self.etf_preference

    def _manage_free_cash(self, summary: dict) -> None:
        """
        Auto-invest idle cash into the best-momentum ETF (QQQ or SPY).
        Fee-aware: minimum $100 investment to make commission worthwhile.
        Delegates to ETFMonitor when it is running (real-time quant).
        """
        if self.etf_monitor is not None and self.etf_monitor.is_running():
            logger.debug("ETFMonitor active — passive auto-invest skipped")
            return
        if not _is_market_open():
            return
        try:
            account = self.broker.get_account_info()
            cash = account.cash

            if cash <= self.free_cash_threshold:
                logger.info(f"Cash ${cash:.2f} ≤ threshold ${self.free_cash_threshold} — no auto-invest")
                return

            invest_amount = cash - self.free_cash_threshold
            if invest_amount < 100:
                logger.info(f"Invest amount ${invest_amount:.2f} < $100 minimum — skipping")
                return

            etf = self._pick_best_etf()
            try:
                etf_price = self.broker.get_current_price(etf)
            except ValueError:
                # Fall back to the alternate
                etf = self.etf_alternate if etf == self.etf_preference else self.etf_preference
                try:
                    etf_price = self.broker.get_current_price(etf)
                except ValueError:
                    logger.warning("Both ETF prices unavailable — skipping auto-invest")
                    return

            order = Order(
                ticker=etf,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=invest_amount / etf_price,
                price=etf_price,
                dollar_amount=invest_amount,
            )
            self.broker.place_order(order)
            logger.info(f"Auto-invested ${invest_amount:.2f} in {etf} @ ${etf_price:.2f}")
            summary.setdefault("auto_invest", []).append(
                {"ticker": etf, "dollar_amount": invest_amount, "price": etf_price}
            )

        except Exception as e:
            logger.error(f"Auto-invest failed: {e}")

    def _liquidate_etf_for_cash(
        self, amount_needed: float, account
    ) -> List[ExecutionResult]:
        """
        Sell QQQ then SPY to raise the required cash for a TDash signal.
        TDash signal always takes priority — no MCDA hesitation, sell regardless of P&L.
        Sells only the portion needed (partial sell), not the full ETF position.
        """
        results = []
        raised = 0.0
        etf_order = [self.etf_preference, self.etf_alternate]  # QQQ first, then SPY

        for etf in etf_order:
            if raised >= amount_needed:
                break

            pos = next((p for p in account.positions if p.ticker == etf), None)
            if not pos or pos.qty <= 0:
                continue

            price = pos.current_price or pos.avg_cost
            if price <= 0:
                continue

            still_needed = amount_needed - raised
            # Add 2% buffer on the last tranche to cover price movement
            sell_dollars = min(still_needed * 1.02, pos.qty * price)
            sell_qty = sell_dollars / price

            try:
                order = Order(
                    ticker=etf,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    qty=sell_qty,
                    price=price,
                )
                self.broker.place_order(order)
                raised += sell_qty * price
                results.append(ExecutionResult(
                    ticker=etf,
                    action="SELL",
                    qty=sell_qty,
                    fill_price=price,
                    success=True,
                    message=f"ETF liquidation to fund trade (~${sell_qty*price:.0f})",
                ))
                logger.info(
                    f"Liquidated {sell_qty:.2f} {etf} @ ${price:.2f} "
                    f"(raised ${sell_qty*price:.0f} / needed ${amount_needed:.0f})"
                )
            except Exception as e:
                logger.error(f"ETF liquidation failed for {etf}: {e}")

        if raised < amount_needed:
            logger.warning(
                f"Could only raise ${raised:.0f} of needed ${amount_needed:.0f} "
                f"— insufficient ETF positions"
            )

        return results

    # ------------------------------------------------------------------
    # Portfolio sync → TDash bot account
    # ------------------------------------------------------------------

    def _sync_portfolio_to_tdash(self, account) -> None:
        """
        Write current broker state to the TDash bot account's portfolio file
        so the next TDash run sees the correct holdings and free-cash budget.

        holdings = all broker positions EXCEPT QQQ/SPY (those are free cash).
        budget   = total_free_cash (cash + QQQ/SPY market value).
        entry_date comes from agent_state.json (recorded on BUY execution).
        """
        if not self._portfolio_sync_file:
            return

        state = _load_state()
        entry_dates: dict = state.get("entry_dates", {})

        etf_tickers = {self.etf_preference, self.etf_alternate}
        holdings = []
        for pos in account.positions:
            if pos.ticker in etf_tickers:
                continue  # ETF positions are part of free cash, not holdings
            holdings.append({
                "ticker": pos.ticker,
                "shares": round(pos.qty, 6),
                "cost_basis": round(pos.avg_cost, 4),
                "entry_date": entry_dates.get(pos.ticker, ""),
            })

        budget = round(self.get_total_free_cash(account), 2)

        portfolio = {
            "holdings": holdings,
            "budget": budget,
            "period": self._portfolio_period,
            "risk_profile": self._portfolio_risk,
        }

        try:
            os.makedirs(os.path.dirname(self._portfolio_sync_file), exist_ok=True)
            with open(self._portfolio_sync_file, "w") as f:
                json.dump(portfolio, f, indent=2)
            logger.info(
                f"Portfolio synced to TDash: {len(holdings)} holdings, "
                f"budget=${budget:,.0f} → {self._portfolio_sync_file}"
            )
        except OSError as e:
            logger.warning(f"Could not write portfolio sync file: {e}")

