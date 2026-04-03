"""
KR Broker Connector — Interacts with Kaigora Agora API
Updated to match KR Broker v2 (Apr 2026)
"""

import os
import sqlite3
import requests
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

# Module-level logger
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    qty: float
    price: float = 0      # Limit price (0 for market)
    trigger_price: float = 0  # For STOP_LIMIT
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0
    avg_fill_price: float = 0
    take_profit: float = 0   # TP price
    stop_loss: float = 0     # SL price
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Position:
    ticker: str
    qty: float
    avg_cost: float
    current_price: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0


@dataclass
class AccountInfo:
    cash: float
    total_equity: float
    unrealized_pnl: float
    buying_power: float


class KRBrokerConnector:
    """Connect to KR Broker (Kaigora Agora API)
    
    API Endpoints:
    - GET  /api/portfolio          → Get portfolio snapshot
    - GET  /api/assets             → Get all asset prices
    - POST /api/orders              → Place market order
    - POST /api/conditional-orders  → Place LIMIT/STOP_LIMIT
    - POST /api/conditional-orders/<id>/cancel → Cancel conditional
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.base_url = config['kr_broker']['api_url']
        self.api_key = config['kr_broker']['api_key']
        
        # Fix path expansion for db_path
        db_path = config['kr_broker']['db_path']
        if db_path.startswith('~'):
            db_path = os.path.expanduser(db_path)
        self.db_path = db_path
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """Get SQLite connection to local broker DB"""
        if self.db_path and os.path.exists(self.db_path):
            return sqlite3.connect(self.db_path)
        return None
    
    def _get_api_headers(self) -> dict:
        """Get headers for Agora API"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    # === Agora API Methods ===
    
    def _get_portfolio(self) -> dict:
        """GET /api/portfolio — Get portfolio from Agora API"""
        try:
            resp = requests.get(
                f"{self.base_url}/portfolio",
                headers=self._get_api_headers(),
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"API error: {e}, using local DB fallback")
            return self._get_portfolio_from_db()
    
    def _get_portfolio_from_db(self) -> dict:
        """Fallback: Get portfolio from local SQLite DB"""
        conn = self._get_db_connection()
        if not conn:
            return {"cash": 0, "positions": []}
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT total_equity, unrealized_pnl, cash 
            FROM equity_log ORDER BY timestamp DESC LIMIT 1
        """)
        row = cursor.fetchone()
        
        cursor.execute("""
            SELECT ticker, quantity, avg_price, current_price, unrealized_pnl
            FROM positions
        """)
        positions = []
        for r in cursor.fetchall():
            positions.append({
                "ticker": r[0], "qty": r[1], "avg_cost": r[2],
                "current_price": r[3], "unrealized_pnl": r[4]
            })
        
        conn.close()
        
        return {
            "cash": row[2] if row else 0,
            "total_equity": row[0] if row else 0,
            "positions": positions
        }
    
    def get_account_info(self) -> AccountInfo:
        """Get current account info"""
        portfolio = self._get_portfolio()
        return AccountInfo(
            cash=portfolio.get('cash', 0),
            total_equity=portfolio.get('total_equity', 0),
            unrealized_pnl=portfolio.get('unrealized_pnl', 0),
            buying_power=portfolio.get('cash', 0)  # Simplified
        )
    
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        portfolio = self._get_portfolio()
        positions = []
        for p in portfolio.get('positions', []):
            positions.append(Position(
                ticker=p['ticker'],
                qty=p.get('qty', p.get('quantity', 0)),
                avg_cost=p.get('avg_cost', p.get('avg_price', 0)),
                current_price=p.get('current_price', 0),
                unrealized_pnl=p.get('unrealized_pnl', 0)
            ))
        return positions
    
    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        """GET /api/assets — Get prices for multiple tickers"""
        try:
            resp = requests.get(
                f"{self.base_url}/assets",
                headers=self._get_api_headers(),
                timeout=10
            )
            resp.raise_for_status()
            assets = resp.json()
            
            prices = {}
            for asset in assets:
                if asset['asset_code'] in tickers:
                    prices[asset['asset_code']] = float(asset.get('current_price', 0))
            return prices
        except Exception as e:
            logger.warning(f"Failed to get prices from API: {e}")
            return {}
    
    def get_current_price(self, ticker: str) -> float:
        """Get current price for a single ticker"""
        prices = self.get_prices([ticker])
        return prices.get(ticker, 0)
    
    # === Order Execution ===
    
    def place_market_order(self, ticker: str, side: OrderSide, qty: float) -> Optional[Order]:
        """POST /api/orders — Place market order
        
        Args:
            ticker: Asset code (e.g., "AAPL")
            side: BUY or SELL
            qty: Number of shares
            
        Returns:
            Order object or None if failed
        """
        try:
            payload = {
                "asset_code": ticker,
                "side": side.value,
                "order_quantity": qty if side == OrderSide.BUY else None,
                "order_amount": qty if side == OrderSide.SELL else None
            }
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            resp = requests.post(
                f"{self.base_url}/orders",
                headers=self._get_api_headers(),
                json=payload,
                timeout=10
            )
            resp.raise_for_status()
            result = resp.json()
            
            return Order(
                order_id=result.get('order_id', ''),
                ticker=ticker,
                side=side,
                order_type=OrderType.MARKET,
                qty=qty,
                status=OrderStatus.FILLED if result.get('filled') else OrderStatus.PENDING,
                fill_price=result.get('fill_price', 0),
                created_at=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Market order failed for {ticker}: {e}")
            return None
    
    def place_conditional_order(self, ticker: str, side: OrderSide, qty: float,
                                 limit_price: float, trigger_price: float = 0,
                                 order_type: OrderType = OrderType.LIMIT) -> Optional[Order]:
        """POST /api/conditional-orders — Place LIMIT or STOP-LIMIT order
        
        Args:
            ticker: Asset code
            side: BUY or SELL
            qty: Number of shares
            limit_price: Price to execute at (for LIMIT/STOP_LIMIT)
            trigger_price: Price to trigger at (for STOP_LIMIT)
            order_type: LIMIT or STOP_LIMIT
            
        Returns:
            Order object or None if failed
        """
        try:
            payload = {
                "ticker": ticker,
                "side": side.value,
                "order_quantity": qty,
                "order_type": order_type.value,
                "limit_price": limit_price
            }
            
            # Add trigger_price for STOP_LIMIT
            if order_type == OrderType.STOP_LIMIT and trigger_price > 0:
                payload["trigger_price"] = trigger_price
            
            resp = requests.post(
                f"{self.base_url}/conditional-orders",
                headers=self._get_api_headers(),
                json=payload,
                timeout=10
            )
            resp.raise_for_status()
            result = resp.json()
            
            return Order(
                order_id=result.get('order_id', result.get('id', '')),
                ticker=ticker,
                side=side,
                order_type=order_type,
                qty=qty,
                price=limit_price,
                trigger_price=trigger_price,
                status=OrderStatus.PENDING,
                created_at=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Conditional order failed for {ticker}: {e}")
            return None
    
    def place_stop_limit_for_tp_sl(self, ticker: str, qty: float,
                                   take_profit: float, stop_loss: float,
                                   current_price: float, side: OrderSide = OrderSide.SELL) -> List[Order]:
        """Place both TP and SL stop-limit orders after a buy
        
        Creates two conditional orders:
        1. TAKE_PROFIT: Sell when price >= take_profit
        2. STOP_LOSS: Sell when price <= stop_loss
        
        Args:
            ticker: Asset code
            qty: Shares to sell on trigger
            take_profit: Target exit price
            stop_loss: Maximum loss price
            current_price: Current market price
            side: SELL (for long positions)
            
        Returns:
            List of created orders [tp_order, sl_order]
        """
        orders = []
        
        # Take Profit order: trigger when price rises to TP
        if take_profit > 0 and take_profit > current_price:
            tp_order = self.place_conditional_order(
                ticker=ticker,
                side=side,
                qty=qty,
                limit_price=take_profit,
                trigger_price=take_profit,
                order_type=OrderType.STOP_LIMIT
            )
            if tp_order:
                tp_order.take_profit = take_profit
                orders.append(tp_order)
        
        # Stop Loss order: trigger when price drops to SL
        if stop_loss > 0 and stop_loss < current_price:
            sl_order = self.place_conditional_order(
                ticker=ticker,
                side=side,
                qty=qty,
                limit_price=stop_loss,
                trigger_price=stop_loss,
                order_type=OrderType.STOP_LIMIT
            )
            if sl_order:
                sl_order.stop_loss = stop_loss
                orders.append(sl_order)
        
        return orders
    
    def cancel_conditional_order(self, order_id: str) -> bool:
        """POST /api/conditional-orders/<id>/cancel — Cancel a conditional order"""
        try:
            resp = requests.post(
                f"{self.base_url}/conditional-orders/{order_id}/cancel",
                headers=self._get_api_headers(),
                timeout=10
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Cancel order {order_id} failed: {e}")
            return False
    
    def verify_fill_price(self, order: Order, expected_price: float, 
                          max_slippage_pct: float = 2.0) -> Tuple[bool, str]:
        """Verify fill price is within acceptable slippage
        
        Returns:
            (is_acceptable, message)
        """
        if order.status != OrderStatus.FILLED:
            return False, f"Order not filled: {order.status}"
        
        if order.avg_fill_price == 0:
            return False, "No fill price recorded"
        
        slippage_pct = abs(order.avg_fill_price - expected_price) / expected_price * 100
        
        if slippage_pct > max_slippage_pct:
            return False, f"Slippage {slippage_pct:.2f}% exceeds max {max_slippage_pct}%"
        
        return True, f"Fill OK: ${order.avg_fill_price:.2f} (slippage {slippage_pct:.2f}%)"

    def get_account(self) -> AccountInfo:
        """Get account info from KR Broker"""
        try:
            resp = requests.get(
                f"{self.base_url}/api/account",
                headers=self._get_headers(),
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            return AccountInfo(
                cash=float(data.get('cash', 0)),
                total_equity=float(data.get('equity', 0)),
                unrealized_pnl=float(data.get('unrealized_pnl', 0)),
                buying_power=float(data.get('buying_power', 0))
            )
        except Exception as e:
            logger.warning(f"Failed to get account from API: {e}")
            # Fallback to local DB
            return self._get_account_from_db()

    def _get_account_from_db(self) -> AccountInfo:
        """Fallback: compute account from local DB"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT equity, unrealized_pnl, cash FROM equity_log ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            return AccountInfo(
                cash=row[2],
                total_equity=row[0],
                unrealized_pnl=row[1],
                buying_power=row[2]
            )
        return AccountInfo(cash=0, total_equity=0, unrealized_pnl=0, buying_power=0)

    def get_positions(self) -> List[Position]:
        """Get current positions"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Get positions from broker DB (or compute from trades)
        cursor.execute("""
            SELECT ticker, SUM(CASE WHEN side = 'BUY' THEN qty ELSE -qty END) as net_qty
            FROM trade_log 
            WHERE status = 'FILLED'
            GROUP BY ticker
            HAVING net_qty > 0
        """)
        
        positions = []
        for row in cursor.fetchall():
            ticker, qty = row
            # Get avg cost
            cursor.execute("""
                SELECT SUM(qty * price) / SUM(qty) 
                FROM trade_log 
                WHERE ticker = ? AND side = 'BUY' AND status = 'FILLED'
            """, (ticker,))
            cost_row = cursor.fetchone()
            avg_cost = cost_row[0] if cost_row and cost_row[0] else 0
            
            positions.append(Position(
                ticker=ticker,
                qty=qty,
                avg_cost=avg_cost
            ))
        
        conn.close()
        return positions
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for specific ticker"""
        positions = self.get_positions()
        for p in positions:
            if p.ticker == ticker:
                return p
        return None
    
    # === Order Execution ===
    
    def place_market_order(self, ticker: str, side: OrderSide, qty: float) -> Order:
        """Place market order"""
        return self._place_order(ticker, side, OrderType.MARKET, qty, price=0)
    
    def place_limit_order(self, ticker: str, side: OrderSide, qty: float, 
                          limit_price: float) -> Order:
        """Place limit order"""
        return self._place_order(ticker, side, OrderType.LIMIT, qty, price=limit_price)
    
    def place_stop_limit_order(self, ticker: str, side: OrderSide, qty: float,
                                stop_price: float, limit_price: float) -> Order:
        """Place stop-limit order"""
        order = self._place_order(ticker, side, OrderType.STOP_LIMIT, qty, 
                                   price=limit_price, stop_price=stop_price)
        # Record in conditional_orders table
        self._record_conditional_order(order)
        return order
    
    def _place_order(self, ticker: str, side: OrderSide, order_type: OrderType,
                     qty: float, price: float = 0, stop_price: float = 0) -> Order:
        """Internal order placement via Agora API"""
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{ticker}"
        
        payload = {
            "order_id": order_id,
            "ticker": ticker,
            "side": side.value,
            "type": order_type.value,
            "qty": qty,
            "price": price,
            "stop_price": stop_price
        }
        
        try:
            response = self.session.post(
                f"{self.api_url}/orders",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            return Order(
                order_id=data.get("order_id", order_id),
                ticker=ticker,
                side=side,
                order_type=order_type,
                qty=qty,
                price=price,
                stop_price=stop_price,
                status=OrderStatus(data.get("status", "pending")),
                created_at=datetime.now().isoformat()
            )
        except requests.exceptions.RequestException as e:
            # Fall back to local simulation if API unavailable
            return self._simulate_order(order_id, ticker, side, order_type, qty, price)
    
    def _simulate_order(self, order_id: str, ticker: str, side: OrderSide,
                        order_type: OrderType, qty: float, price: float) -> Order:
        """Simulate order locally (for testing/development)"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO trade_log (order_id, ticker, side, qty, price, type, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (order_id, ticker, side.value, qty, price, order_type.value, "pending", now))
        
        conn.commit()
        conn.close()
        
        return Order(
            order_id=order_id,
            ticker=ticker,
            side=side,
            order_type=order_type,
            qty=qty,
            price=price,
            status=OrderStatus.PENDING,
            created_at=now
        )
    
    def _record_conditional_order(self, order: Order):
        """Record conditional/stop order in DB"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conditional_orders 
            (order_id, ticker, side, trigger_price, limit_price, qty, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (order.order_id, order.ticker, order.side.value, 
              order.stop_price, order.price, order.qty, "active", order.created_at))
        
        conn.commit()
        conn.close()
    
    # === Order Management ===
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT order_id, ticker, side, qty, price, type, status, timestamp
            FROM trade_log
            WHERE status = 'pending'
        """)
        
        orders = []
        for row in cursor.fetchall():
            orders.append(Order(
                order_id=row[0],
                ticker=row[1],
                side=OrderSide(row[2]),
                order_type=OrderType(row[6]) if len(row) > 6 else OrderType.MARKET,
                qty=row[3],
                price=row[4],
                status=OrderStatus(row[6]),
                created_at=row[7]
            ))
        
        conn.close()
        return orders
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE trade_log SET status = 'cancelled' WHERE order_id = ? AND status = 'pending'
        """, (order_id,))
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Check order status"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT status FROM trade_log WHERE order_id = ?
        """, (order_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return OrderStatus(row[0])
        return None
    
    # === Price Fetching ===
    
    def get_current_price(self, ticker: str) -> float:
        """Get current price for ticker"""
        try:
            response = self.session.get(
                f"{self.api_url}/prices/{ticker}",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            return data.get("price", 0)
        except:
            # Fallback: estimate from positions or return 0
            return 0
    
    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices for multiple tickers"""
        prices = {}
        for ticker in tickers:
            prices[ticker] = self.get_current_price(ticker)
        return prices
    
    # === Execution Helpers ===
    
    def wait_for_fill(self, order_id: str, timeout_seconds: int = 60) -> OrderStatus:
        """Wait for order to fill"""
        start = time.time()
        while time.time() - start < timeout_seconds:
            status = self.get_order_status(order_id)
            if status in [OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED]:
                return status
            time.sleep(2)
        return OrderStatus.PENDING
    
    def fill_limit_order(self, order_id: str) -> bool:
        """Simulate filling a limit order (for local testing)"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE trade_log SET status = 'filled', filled_qty = qty, 
            avg_fill_price = price WHERE order_id = ?
        """, (order_id,))
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0
