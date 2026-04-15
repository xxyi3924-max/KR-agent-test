"""
KR Broker Connector — calls the KR Broker Flask app (localhost:8084).

KR Broker acts as a proxy between this agent and the Kaigora Agora API.
All requests require:  Authorization: Bearer <api_key>

Agora order conventions (enforced by KR Broker):
  Market BUY  → order_amount  (dollar amount to spend)
  Market SELL → order_quantity (number of shares)
  Conditional (LIMIT / STOP_LIMIT) → order_quantity (shares) for both sides
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """Represents an order to be placed or already placed."""
    ticker: str
    side: OrderSide
    order_type: OrderType
    qty: float                          # shares
    price: float = 0.0                  # limit price (0 for market)
    trigger_price: float = 0.0          # stop trigger for STOP_LIMIT
    dollar_amount: float = 0.0          # for BUY market orders (derived: qty * price)
    tp_price: float = 0.0
    sl_price: float = 0.0
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_at: str = ""


@dataclass
class Position:
    ticker: str
    qty: float
    avg_cost: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class AccountInfo:
    cash: float
    total_equity: float
    unrealized_pnl: float
    buying_power: float
    positions: List[Position] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class KRBrokerConnector:
    """
    Thin HTTP client targeting the KR Broker Flask app.

    Args:
        api_url: Base URL of the KR Broker Flask app, e.g. "http://localhost:8084"
        api_key: Kaigora API key (passed as Bearer token)
    """

    def __init__(self, api_url: str, api_key: str):
        self.base_url = api_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._tradable_cache: Optional[set] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, timeout: int = 10) -> dict:
        resp = self.session.get(f"{self.base_url}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict = None, timeout: int = 10) -> dict:
        resp = self.session.post(
            f"{self.base_url}{path}",
            json=body or {},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Portfolio / positions
    # ------------------------------------------------------------------

    def _fetch_portfolio(self) -> dict:
        """GET /api/portfolio — raw Agora portfolio response."""
        return self._get("/api/portfolio")

    def get_account_info(self) -> AccountInfo:
        """Return current account snapshot including positions."""
        data = self._fetch_portfolio()
        positions = []
        for p in data.get("positions", []):
            qty = float(
                p.get("quantity")
                or p.get("shares")
                or p.get("qty")
                or 0
            )
            positions.append(Position(
                ticker=p.get("assetCode", ""),
                qty=qty,
                avg_cost=float(p.get("avgCost") or 0),
                current_price=float(p.get("currentPrice") or 0),
                unrealized_pnl=float(p.get("unrealizedPnl") or 0),
            ))
        return AccountInfo(
            cash=float(data.get("availableCash") or 0),
            total_equity=float(data.get("totalEquity") or 0),
            unrealized_pnl=float(data.get("unrealizedPnl") or 0),
            buying_power=float(data.get("availableCash") or 0),
            positions=positions,
        )

    def get_positions(self) -> List[Position]:
        return self.get_account_info().positions

    def get_position(self, ticker: str) -> Optional[Position]:
        for p in self.get_positions():
            if p.ticker == ticker:
                return p
        return None

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def get_tradable_tickers(self) -> set:
        """Return the set of all assetCodes available on Kaigora (cached per session).
        Pagination is handled by kr-broker; this is a single call."""
        if self._tradable_cache is not None:
            return self._tradable_cache
        try:
            # kr-broker loads all pages in parallel on first call (~4s); longer timeout needed
            data = self._get("/api/assets", timeout=30)
            tickers = {a["assetCode"] for a in data.get("assets", []) if a.get("assetCode")}
            logger.info(f"Loaded {len(tickers)} tradable tickers")
            self._tradable_cache = tickers
            return tickers
        except Exception as e:
            logger.warning(f"get_tradable_tickers failed: {e}")
            return set()

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        """GET /api/assets — return {ticker: price} for the requested tickers."""
        try:
            data = self._get("/api/assets", timeout=30)
            assets = data.get("assets", [])
            return {
                a["assetCode"]: float(a["currentPrice"])
                for a in assets
                if a.get("assetCode") in tickers and a.get("currentPrice") is not None
            }
        except Exception as e:
            logger.warning(f"get_prices failed: {e}")
            return {}

    def get_current_price(self, ticker: str) -> float:
        """Return current price for a single ticker, or raise if unavailable."""
        prices = self.get_prices([ticker])
        price = prices.get(ticker)
        if price is None:
            raise ValueError(f"Price unavailable for {ticker}")
        return price

    def get_price_history(self, ticker: str, limit: int = 50) -> List[float]:
        """
        GET /api/price-history/<ticker>
        Returns list of prices oldest-first, up to `limit` entries.
        Returns [] if unavailable (caller should handle gracefully).
        """
        try:
            data = self._get(f"/api/price-history/{ticker.upper()}")
            history = data.get("history", [])
            prices = [float(h["price"]) for h in history if h.get("price") is not None]
            return prices[-limit:]
        except Exception as e:
            logger.warning(f"Price history unavailable for {ticker}: {e}")
            return []

    # ------------------------------------------------------------------
    # Order placement — central dispatcher
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> dict:
        """
        Place an order via KR Broker.  Routes to the correct endpoint
        based on order_type:
          MARKET    → POST /api/orders
          LIMIT     → POST /api/conditional-orders
          STOP_LIMIT → POST /api/conditional-orders
        Returns the raw API response dict.
        """
        if order.order_type == OrderType.MARKET:
            return self._place_market_order(order)
        else:
            return self._place_conditional_order(order)

    def _place_market_order(self, order: Order) -> dict:
        """
        POST /api/orders
        BUY  → send order_amount  (dollars)
        SELL → send order_quantity (shares)
        """
        if order.side == OrderSide.BUY:
            # Agora BUY needs a dollar amount
            dollar_amount = order.dollar_amount
            if not dollar_amount:
                dollar_amount = order.qty * order.price if order.price else 0
            if not dollar_amount:
                raise ValueError(
                    f"Cannot place BUY market order for {order.ticker}: "
                    "no dollar_amount or price available"
                )
            body = {
                "ticker": order.ticker,
                "side": "BUY",
                "order_amount": round(dollar_amount, 2),
            }
        else:
            # Agora SELL needs share quantity
            if not order.qty:
                raise ValueError(
                    f"Cannot place SELL market order for {order.ticker}: qty=0"
                )
            body = {
                "ticker": order.ticker,
                "side": "SELL",
                "order_quantity": order.qty,
            }

        logger.info(f"Market order → {body}")
        result = self._post("/api/orders", body)
        logger.info(f"Market order result: {result}")
        return result

    def _place_conditional_order(self, order: Order) -> dict:
        """
        POST /api/conditional-orders
        Both BUY and SELL use order_quantity (shares).
        """
        if not order.qty:
            raise ValueError(
                f"Cannot place conditional order for {order.ticker}: qty=0"
            )
        if not order.price:
            raise ValueError(
                f"Cannot place conditional order for {order.ticker}: no limit_price"
            )

        body: dict = {
            "ticker": order.ticker,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "limit_price": order.price,
            "order_quantity": order.qty,
        }

        if order.order_type == OrderType.STOP_LIMIT:
            if not order.trigger_price:
                raise ValueError(
                    f"STOP_LIMIT order for {order.ticker} requires trigger_price"
                )
            body["trigger_price"] = order.trigger_price

        logger.info(f"Conditional order → {body}")
        result = self._post("/api/conditional-orders", body)
        logger.info(f"Conditional order result: {result}")
        return result

    # ------------------------------------------------------------------
    # TP / SL helpers
    # ------------------------------------------------------------------

    def place_tp_sl_pair(
        self,
        ticker: str,
        qty: float,
        take_profit: float,
        stop_loss: float,
        current_price: float,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Place a LIMIT sell for TP and a STOP_LIMIT sell for SL.
        Returns (tp_conditional_id, sl_conditional_id).
        These IDs should be stored in agent_state for cancellation when one fills.
        """
        tp_id: Optional[int] = None
        sl_id: Optional[int] = None

        if take_profit > 0 and take_profit > current_price:
            try:
                tp_order = Order(
                    ticker=ticker,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    qty=qty,
                    price=take_profit,
                )
                result = self._place_conditional_order(tp_order)
                tp_id = result.get("id")
                logger.info(f"TP set for {ticker} @ ${take_profit:.2f} (id={tp_id})")
            except Exception as e:
                logger.error(f"Failed to set TP for {ticker}: {e}")

        if stop_loss > 0 and stop_loss < current_price:
            try:
                sl_order = Order(
                    ticker=ticker,
                    side=OrderSide.SELL,
                    order_type=OrderType.STOP_LIMIT,
                    qty=qty,
                    price=stop_loss,
                    trigger_price=stop_loss,
                )
                result = self._place_conditional_order(sl_order)
                sl_id = result.get("id")
                logger.info(f"SL set for {ticker} @ ${stop_loss:.2f} (id={sl_id})")
            except Exception as e:
                logger.error(f"Failed to set SL for {ticker}: {e}")

        return tp_id, sl_id

    def cancel_conditional_order(self, order_id: int) -> bool:
        """POST /api/conditional-orders/<id>/cancel"""
        try:
            self._post(f"/api/conditional-orders/{order_id}/cancel")
            logger.info(f"Cancelled conditional order {order_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel conditional order {order_id}: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """POST /api/orders/<id>/cancel"""
        try:
            self._post(f"/api/orders/{order_id}/cancel")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Fill polling
    # ------------------------------------------------------------------

    def wait_for_fill(self, order_id: str, timeout_seconds: int = 60) -> OrderStatus:
        """Poll order status until filled, rejected, or timeout."""
        start = time.time()
        while time.time() - start < timeout_seconds:
            try:
                data = self._get(f"/api/orders/{order_id}")
                status_str = (data.get("status") or "pending").lower()
                status = OrderStatus(status_str)
                if status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED):
                    return status
            except Exception:
                pass
            time.sleep(2)
        return OrderStatus.PENDING

    # ------------------------------------------------------------------
    # Slippage verification
    # ------------------------------------------------------------------

    def verify_slippage(
        self,
        fill_price: float,
        expected_price: float,
        max_slippage_pct: float = 2.0,
    ) -> Tuple[bool, str]:
        """Return (ok, message). Raises no exceptions."""
        if expected_price <= 0:
            return True, "No expected price to compare"
        slippage = abs(fill_price - expected_price) / expected_price * 100
        if slippage > max_slippage_pct:
            return False, f"Slippage {slippage:.2f}% exceeds limit {max_slippage_pct}%"
        return True, f"Slippage {slippage:.2f}% within limit"
