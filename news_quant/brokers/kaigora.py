"""Kaigora adapter — wraps kr_broker.KRBrokerConnector with optional dry-run.

K1 default: ``dry_run=True``. ``place_order`` and ``place_tp_sl_pair`` log
intended actions but do not call the live Kaigora API. Read-only methods
(account, position, price) pass through so we can still cross-check
quotes/fills against Alpaca paper.

Set ``dry_run=False`` only after the forward shadow on Alpaca paper clears
the gate criteria.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from kr_broker import KRBrokerConnector

from news_quant.brokers.base import (
    AccountInfo,
    BrokerConnector,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)

logger = logging.getLogger("news_quant.brokers.kaigora")


class KaigoraBroker(BrokerConnector):
    name = "kaigora"
    is_paper = False  # Kaigora has no sandbox — live or dry-run

    def __init__(self, api_url: str, api_key: str, dry_run: bool = True):
        self._inner = KRBrokerConnector(api_url=api_url, api_key=api_key)
        self.is_dry_run = dry_run
        if dry_run:
            logger.warning("KaigoraBroker in DRY-RUN — orders will be logged, not placed")

    # ── read-only passthrough ────────────────────────────────────────────────
    def get_account_info(self) -> AccountInfo:
        return self._inner.get_account_info()

    def get_position(self, ticker: str) -> Optional[Position]:
        return self._inner.get_position(ticker)

    def get_current_price(self, ticker: str) -> float:
        return self._inner.get_current_price(ticker)

    def get_tradable_tickers(self) -> set:
        return self._inner.get_tradable_tickers()

    def get_prices(self, tickers):
        return self._inner.get_prices(tickers)

    # ── mutating: gated by dry_run ───────────────────────────────────────────
    def place_order(self, order: Order) -> dict:
        if self.is_dry_run:
            logger.info(
                "[DRY-RUN] place_order ticker=%s side=%s type=%s qty=%s price=%s",
                order.ticker, order.side.value, order.order_type.value,
                order.qty, order.price,
            )
            order.order_id = f"dryrun-{id(order)}"
            order.status = OrderStatus.FILLED  # pretend it filled for downstream bookkeeping
            order.filled_qty = order.qty
            order.avg_fill_price = order.price or self.get_current_price(order.ticker)
            return {"order_id": order.order_id, "dry_run": True}
        return self._inner.place_order(order)

    def place_tp_sl_pair(
        self,
        ticker: str,
        qty: float,
        tp_price: float,
        sl_price: float,
        side: OrderSide = OrderSide.SELL,
    ) -> Tuple[Optional[str], Optional[str]]:
        if self.is_dry_run:
            logger.info(
                "[DRY-RUN] tp_sl_pair ticker=%s qty=%s tp=%s sl=%s side=%s",
                ticker, qty, tp_price, sl_price, side.value,
            )
            return f"dryrun-tp-{ticker}", f"dryrun-sl-{ticker}"
        return self._inner.place_tp_sl_pair(ticker, qty, tp_price, sl_price, side)

    def cancel_order(self, order_id: str) -> bool:
        if self.is_dry_run:
            logger.info("[DRY-RUN] cancel_order id=%s", order_id)
            return True
        return self._inner.cancel_order(order_id)

    def cancel_conditional_order(self, order_id: str) -> bool:
        if self.is_dry_run:
            logger.info("[DRY-RUN] cancel_conditional_order id=%s", order_id)
            return True
        # KRBrokerConnector signature is int; cast best-effort
        try:
            return self._inner.cancel_conditional_order(int(order_id))
        except (TypeError, ValueError):
            logger.warning("non-int conditional order id=%r — cannot cancel", order_id)
            return False

    def wait_for_fill(self, order_id: str, timeout_seconds: int = 60) -> OrderStatus:
        if self.is_dry_run:
            return OrderStatus.FILLED
        return self._inner.wait_for_fill(order_id, timeout_seconds)

    def close_position(self, ticker: str) -> bool:
        if self.is_dry_run:
            logger.info("[DRY-RUN] close_position ticker=%s", ticker)
            return True
        # kr_broker has no atomic close — emit a market sell of whatever qty we hold.
        pos = self._inner.get_position(ticker)
        if pos is None or abs(pos.qty) < 1e-9:
            return True
        from kr_broker import Order as KROrder, OrderType as KROrderType, OrderSide as KROrderSide
        close = KROrder(
            ticker=ticker,
            side=KROrderSide.SELL if pos.qty > 0 else KROrderSide.BUY,
            order_type=KROrderType.MARKET,
            qty=abs(pos.qty),
        )
        resp = self._inner.place_order(close)
        return "error" not in resp
