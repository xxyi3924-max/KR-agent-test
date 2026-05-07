"""Abstract broker contract used by news_quant.executor and daemon.

Mirrors the surface area that `kr_broker.KRBrokerConnector` already provides,
so the Kaigora adapter is a thin wrapper. Alpaca adapter implements the same
contract against the alpaca-py SDK.

Reuses `Order`, `Position`, `AccountInfo`, `OrderType`, `OrderSide`, `OrderStatus`
from `kr_broker` to avoid duplicating dataclasses across adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kr_broker import (  # noqa: E402  re-exported types
    AccountInfo,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

__all__ = [
    "BrokerConnector",
    "AccountInfo",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
]


class BrokerConnector(ABC):
    """Common interface every news_quant broker adapter implements.

    Adapters never raise on transient errors — they return sentinel values
    (`{}`, `None`, `OrderStatus.REJECTED`) and log. Fatal/auth errors should
    raise so the daemon halts loudly.
    """

    name: str = "abstract"
    is_paper: bool = False
    is_dry_run: bool = False

    @abstractmethod
    def get_account_info(self) -> AccountInfo: ...

    @abstractmethod
    def get_position(self, ticker: str) -> Optional[Position]: ...

    @abstractmethod
    def get_current_price(self, ticker: str) -> float: ...

    @abstractmethod
    def place_order(self, order: Order) -> dict:
        """Submit a market or conditional order. Returns broker response dict
        with at least an ``order_id`` key on success, or an ``error`` key on
        failure. Adapters set ``order.order_id`` before returning.
        """

    @abstractmethod
    def place_tp_sl_pair(
        self,
        ticker: str,
        qty: float,
        tp_price: float,
        sl_price: float,
        side: OrderSide = OrderSide.SELL,
    ) -> Tuple[Optional[str], Optional[str]]:
        """After an entry fill, attach a take-profit + stop-loss exit pair.
        Returns (tp_order_id, sl_order_id); either may be None on failure.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def cancel_conditional_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def wait_for_fill(self, order_id: str, timeout_seconds: int = 60) -> OrderStatus: ...

    @abstractmethod
    def close_position(self, ticker: str) -> bool:
        """Liquidate any open position on ``ticker`` and cancel any open
        bracket/child orders that would block the close. Returns True if the
        position is gone (or never existed), False on failure.
        """

    # Optional helpers — default no-op implementations adapters may override.
    def get_tradable_tickers(self) -> set:
        return set()

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: self.get_current_price(t) for t in tickers}
