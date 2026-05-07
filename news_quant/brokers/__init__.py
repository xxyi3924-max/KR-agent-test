"""Broker abstraction for news_quant.

`BrokerConnector` is the surface the executor and daemon code against.
Concrete implementations live in `alpaca.py` (paper-trading default) and
`kaigora.py` (wraps the existing kr_broker.KRBrokerConnector).

Use `factory.create_broker(cfg)` to instantiate the right connector based
on the news_quant config. Root-level decision_engine and etf_monitor are
unaffected — they keep talking to KRBrokerConnector directly.
"""

from news_quant.brokers.base import BrokerConnector
from news_quant.brokers.factory import create_broker

__all__ = ["BrokerConnector", "create_broker"]
