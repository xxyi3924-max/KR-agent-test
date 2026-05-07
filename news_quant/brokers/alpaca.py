"""Alpaca paper-trading adapter.

Maps news_quant's broker contract onto alpaca-py.

Notable differences from Kaigora:
- Alpaca supports atomic bracket orders (entry + TP + SL in one submission).
  We use that instead of the kr_broker pattern of placing TP/SL separately
  after entry fill. ``place_order(order)`` with non-zero ``order.tp_price``
  and ``order.sl_price`` becomes a bracket order; ``place_tp_sl_pair`` is
  retained for compatibility but normally unused on this venue.
- Alpaca uses fractional shares with qty (no dollar_amount path); BUY and
  SELL both pass ``qty``.
- Shorting requires the ticker to be marketable as shortable; we surface
  that via ``error`` in the response and the caller decides to skip.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, TimeInForce
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import OrderStatus as AlpacaOrderStatus
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from news_quant.brokers.base import (
    AccountInfo,
    BrokerConnector,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

logger = logging.getLogger("news_quant.brokers.alpaca")


def _map_status(s: AlpacaOrderStatus) -> OrderStatus:
    if s in (AlpacaOrderStatus.FILLED, AlpacaOrderStatus.DONE_FOR_DAY):
        return OrderStatus.FILLED
    if s == AlpacaOrderStatus.PARTIALLY_FILLED:
        return OrderStatus.PARTIAL
    if s in (AlpacaOrderStatus.CANCELED, AlpacaOrderStatus.EXPIRED, AlpacaOrderStatus.PENDING_CANCEL):
        return OrderStatus.CANCELLED
    if s == AlpacaOrderStatus.REJECTED:
        return OrderStatus.REJECTED
    return OrderStatus.PENDING


def _to_alpaca_side(side: OrderSide) -> AlpacaOrderSide:
    return AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL


class AlpacaBroker(BrokerConnector):
    name = "alpaca"

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.is_paper = paper
        self.is_dry_run = False
        self._trading = TradingClient(api_key, secret_key, paper=paper)
        # Data client needs separate auth; same keys work on the IEX free tier.
        self._data = StockHistoricalDataClient(api_key, secret_key)
        logger.info("AlpacaBroker init paper=%s", paper)

    # ── account / positions ─────────────────────────────────────────────────
    def get_account_info(self) -> AccountInfo:
        a = self._trading.get_account()
        positions = self._trading.get_all_positions()
        pos_list = []
        for p in positions:
            pos_list.append(
                Position(
                    ticker=p.symbol,
                    qty=float(p.qty),
                    avg_cost=float(p.avg_entry_price),
                    current_price=float(p.current_price or 0),
                    unrealized_pnl=float(p.unrealized_pl or 0),
                )
            )
        return AccountInfo(
            cash=float(a.cash),
            total_equity=float(a.equity),
            unrealized_pnl=sum(p.unrealized_pnl for p in pos_list),
            buying_power=float(a.buying_power),
            positions=pos_list,
        )

    def get_position(self, ticker: str) -> Optional[Position]:
        try:
            p = self._trading.get_open_position(ticker)
        except APIError as e:
            # 404 = no position; that's expected
            if "position does not exist" in str(e).lower() or "404" in str(e):
                return None
            raise
        return Position(
            ticker=p.symbol,
            qty=float(p.qty),
            avg_cost=float(p.avg_entry_price),
            current_price=float(p.current_price or 0),
            unrealized_pnl=float(p.unrealized_pl or 0),
        )

    # ── pricing ─────────────────────────────────────────────────────────────
    def get_current_price(self, ticker: str) -> float:
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quotes = self._data.get_stock_latest_quote(req)
            q = quotes.get(ticker)
            if not q:
                return 0.0
            # mid of bid/ask if both present, else whichever is non-zero
            bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            return ask or bid
        except Exception as e:
            logger.warning("get_current_price(%s) failed: %s", ticker, e)
            return 0.0

    def get_tradable_tickers(self) -> set:
        # Alpaca exposes ~10k assets; we don't enumerate. Caller should handle
        # rejected orders for non-tradable symbols.
        return set()

    # ── order placement ─────────────────────────────────────────────────────
    def place_order(self, order: Order) -> dict:
        try:
            side = _to_alpaca_side(order.side)
            tif = TimeInForce.DAY  # news_quant trades are intraday

            # If TP/SL are set on the order, submit as a bracket. Bracket is
            # only valid for entry orders (BUY long or SELL short); exit
            # orders shouldn't include nested TP/SL.
            use_bracket = (
                order.tp_price > 0
                and order.sl_price > 0
                and order.order_type == OrderType.MARKET
            )

            if order.order_type == OrderType.MARKET:
                kwargs = dict(symbol=order.ticker, qty=order.qty, side=side, time_in_force=tif)
                if use_bracket:
                    kwargs["order_class"] = OrderClass.BRACKET
                    kwargs["take_profit"] = TakeProfitRequest(limit_price=round(order.tp_price, 2))
                    kwargs["stop_loss"] = StopLossRequest(stop_price=round(order.sl_price, 2))
                req = MarketOrderRequest(**kwargs)
            elif order.order_type == OrderType.LIMIT:
                req = LimitOrderRequest(
                    symbol=order.ticker, qty=order.qty, side=side,
                    time_in_force=tif, limit_price=round(order.price, 2),
                )
            else:
                return {"error": f"unsupported order_type={order.order_type}"}

            resp = self._trading.submit_order(req)
            order.order_id = str(resp.id)
            order.status = _map_status(resp.status)
            return {"order_id": order.order_id, "raw": resp.model_dump() if hasattr(resp, "model_dump") else {}}
        except APIError as e:
            msg = str(e)
            logger.error("place_order failed ticker=%s side=%s qty=%s: %s",
                         order.ticker, order.side, order.qty, msg)
            order.status = OrderStatus.REJECTED
            return {"error": msg}

    def place_tp_sl_pair(
        self,
        ticker: str,
        qty: float,
        tp_price: float,
        sl_price: float,
        side: OrderSide = OrderSide.SELL,
    ) -> Tuple[Optional[str], Optional[str]]:
        # Alpaca prefers atomic brackets. This separate-pair API is kept for
        # parity with kr_broker but submits two independent orders.
        a_side = _to_alpaca_side(side)
        tp_id = sl_id = None
        try:
            tp_resp = self._trading.submit_order(
                LimitOrderRequest(
                    symbol=ticker, qty=qty, side=a_side,
                    time_in_force=TimeInForce.GTC, limit_price=round(tp_price, 2),
                )
            )
            tp_id = str(tp_resp.id)
        except APIError as e:
            logger.error("TP placement failed ticker=%s: %s", ticker, e)
        try:
            from alpaca.trading.requests import StopOrderRequest
            sl_resp = self._trading.submit_order(
                StopOrderRequest(
                    symbol=ticker, qty=qty, side=a_side,
                    time_in_force=TimeInForce.GTC, stop_price=round(sl_price, 2),
                )
            )
            sl_id = str(sl_resp.id)
        except APIError as e:
            logger.error("SL placement failed ticker=%s: %s", ticker, e)
        return tp_id, sl_id

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._trading.cancel_order_by_id(order_id)
            return True
        except APIError as e:
            logger.warning("cancel_order(%s) failed: %s", order_id, e)
            return False

    def cancel_conditional_order(self, order_id: str) -> bool:
        # Alpaca treats stop/limit orders the same as market — same cancel call.
        return self.cancel_order(order_id)

    def close_position(self, ticker: str) -> bool:
        # 1. Cancel any open child orders (bracket TP/SL would otherwise hold qty).
        try:
            req = GetOrdersRequest(status="open", symbols=[ticker])
            opens = self._trading.get_orders(filter=req)
            for o in opens:
                try:
                    self._trading.cancel_order_by_id(o.id)
                except APIError as e:
                    logger.warning("cancel(%s) failed: %s", o.id, e)
        except APIError as e:
            logger.warning("get_orders for %s failed: %s", ticker, e)
        # 2. If no position remains (e.g. bracket already closed), we're done.
        try:
            self._trading.get_open_position(ticker)
        except APIError as e:
            if "position does not exist" in str(e).lower() or "404" in str(e):
                return True
            logger.warning("get_open_position check: %s", e)
        # 3. Liquidate via Alpaca's per-symbol close.
        try:
            time.sleep(0.5)  # let cancel propagate
            self._trading.close_position(ticker)
            return True
        except APIError as e:
            logger.error("close_position(%s) failed: %s", ticker, e)
            return False

    def wait_for_fill(self, order_id: str, timeout_seconds: int = 60) -> OrderStatus:
        deadline = time.time() + timeout_seconds
        last_status = OrderStatus.PENDING
        while time.time() < deadline:
            try:
                o = self._trading.get_order_by_id(order_id)
                last_status = _map_status(o.status)
                if last_status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
                    return last_status
            except APIError as e:
                logger.warning("wait_for_fill poll failed for %s: %s", order_id, e)
            time.sleep(2)
        return last_status
