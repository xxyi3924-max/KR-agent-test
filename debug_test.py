from decision_engine import DecisionEngine
from tdash_connector import Signal
from kr_broker import OrderType

# Create engine without __init__
e = DecisionEngine.__new__(DecisionEngine)
e.broker = None
e.trading_config = {}
e.max_slippage = 2.0

# Create signal
s = Signal(
    ticker="AAPL", 
    action="BUY", 
    shares=100, 
    entry_price=175.0, 
    take_profit=190.0, 
    stop_loss=165.0, 
    priority=1, 
    risk_flags=[]
)

print(f"Signal action: {s.action}")
print(f"Required cash: {s.shares * s.entry_price}")
print(f"Available (95%): {50000 * 0.95}")

# Test
p = e._create_plan(s, 173.0, 50000)
print(f"Plan: {p}")
if p:
    print(f"Order type: {p.order_type}")
    print(f"Price: {p.price}")
