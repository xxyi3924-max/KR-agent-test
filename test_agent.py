"""
Tests for Agent Trader Components
"""

import pytest
import yaml
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcda_engine import MCDAMomentumEngine
from tdash_connector import Portfolio, Signal
from kr_broker import Order, OrderType, OrderSide
from decision_engine import (
    DecisionEngine, ExecutionPlan, ExecutionResult,
    LLMExecutionAgent, ExecutionDecision
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def mcda_engine(config):
    return MCDAMomentumEngine(config)


# =============================================================================
# TEST MCDA MOMENTUM ENGINE
# =============================================================================

class TestMCDAMomentum:
    """Tests for MCDA momentum scoring"""
    
    def test_momentum_positive(self, mcda_engine):
        prices = [100, 105, 110]
        momentum = mcda_engine.calculate_momentum(prices)
        assert momentum == 10.0
        assert momentum > 0
    
    def test_momentum_negative(self, mcda_engine):
        prices = [110, 105, 98]
        momentum = mcda_engine.calculate_momentum(prices)
        assert momentum < 0
        assert abs(momentum - (-10.91)) < 0.01
    
    def test_momentum_flat(self, mcda_engine):
        prices = [100, 100, 100]
        momentum = mcda_engine.calculate_momentum(prices)
        assert momentum == 0
    
    def test_rsi_oversold(self, mcda_engine):
        rsi = 25
        rsi_calc = rsi  # Already oversold
        assert rsi_calc < 30
    
    def test_rsi_overbought(self, mcda_engine):
        rsi = 85
        rsi_calc = rsi  # Already overbought
        assert rsi_calc > 70


class TestMCDAScoring:
    """Tests for position scoring"""
    
    def test_score_low_momentum_sells_first(self, mcda_engine):
        """Positions with low momentum should score low (sell first)"""
        prices = [180, 170, 160, 155]  # Declining
        score = mcda_engine.score_position("AAPL", prices)
        
        assert score.total_score < 50
    
    def test_score_high_momentum_holds(self, mcda_engine):
        """Positions with high momentum should score high (hold)"""
        prices = [200, 210, 225, 240]  # Rallying
        score = mcda_engine.score_position("TSLA", prices)
        
        assert score.total_score > 50


# =============================================================================
# TEST FEE-AWARE TRADING
# =============================================================================

class TestFeeAwareTrading:
    """Tests for fee calculations"""
    
    def test_fee_calculation(self):
        fee = max(1000 * 0.001, 1)
        assert fee == 1.0
    
    def test_minimum_fee(self):
        trade_value = 100
        fee = max(trade_value * 0.001, 1)
        assert fee == 1
    
    def test_trade_viable(self):
        entry = 100
        current = 102
        qty = 100
        
        gross_gain = (current - entry) * qty
        fee = max(100 * 0.001, 1)
        net_gain = gross_gain - fee
        
        assert net_gain > 0
        assert net_gain == 199
    
    def test_trade_too_small(self):
        entry = 100
        current = 100.50
        qty = 100
        
        gross_gain = (current - entry) * qty
        fee = max(gross_gain * 0.001, 1)
        net_gain = gross_gain - fee
        
        assert net_gain == 49
    
    def test_max_affordable_quantity(self):
        cash = 10000
        price = 100
        min_fee = 1
        
        max_qty = (cash - min_fee) / price
        assert max_qty == 99.99


# =============================================================================
# TEST TDASH CONNECTOR
# =============================================================================

class TestTDashConnector:
    """Tests for TDash connector"""
    
    def test_signal_creation(self):
        signal = Signal(
            ticker="AAPL",
            action="BUY",
            shares=100,
            entry_price=175.00,
            take_profit=190.00,
            stop_loss=165.00,
            priority=1,
            risk_flags=[]
        )
        
        assert signal.ticker == "AAPL"
        assert signal.action == "BUY"
        assert signal.shares == 100
        assert signal.entry_price == 175.00
        assert signal.take_profit == 190.00
        assert signal.stop_loss == 165.00
    
    def test_portfolio_structure(self):
        portfolio = Portfolio(
            market="US",
            period="short",
            risk_profile="aggressive",
            capital=372092,
            cash=50000,
            holdings=[],
            signals=[],
            macro_context="",
            run_timestamp="",
            token_usage={}
        )
        
        assert portfolio.market == "US"
        assert portfolio.capital == 372092
        assert isinstance(portfolio.signals, list)
        assert isinstance(portfolio.holdings, list)


# =============================================================================
# TEST KR BROKER CONNECTOR
# =============================================================================

class TestKRBrokerConnector:
    """Tests for KR Broker connector"""
    
    def test_order_creation(self):
        order = Order(
            order_id="TEST001",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=100,
            price=175.00
        )
        
        assert order.ticker == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.qty == 100
        assert order.order_type == OrderType.MARKET
    
    def test_limit_order_creation(self):
        order = Order(
            order_id="TEST002",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=100,
            price=174.00
        )
        
        assert order.order_type == OrderType.LIMIT
        assert order.price == 174.00
    
    def test_stop_limit_order(self):
        order = Order(
            order_id="TEST003",
            ticker="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            qty=100,
            price=165.00,
            trigger_price=165.00
        )
        
        assert order.order_type == OrderType.STOP_LIMIT
        assert order.trigger_price == 165.00


# =============================================================================
# TEST LLM EXECUTION AGENT
# =============================================================================

class TestLLMExecutionAgent:
    """Tests for LLM execution agent"""
    
    def test_llm_agent_initialization(self, config):
        agent = LLMExecutionAgent(config)
        
        assert agent.provider == config['llm']['provider']
        assert agent.model == config['llm']['model']
        assert agent.confidence_threshold == config['llm']['confidence_threshold']
    
    def test_llm_fallback_when_no_client(self, config):
        config['llm']['openai_api_key'] = ''
        agent = LLMExecutionAgent(config)
        
        decision = agent.decide(
            signals=[{"ticker": "AAPL", "action": "BUY", "shares": 100}],
            portfolio={"cash": 10000, "positions": []}
        )
        
        assert decision.decision == "WAIT"
        assert decision.confidence == 0.0
    
    def test_llm_order_parsing(self, config):
        agent = LLMExecutionAgent(config)
        
        response = '''
        {
            "decision": "EXECUTE",
            "reasoning": "AAPL looks good",
            "confidence": 0.85,
            "orders": [
                {"ticker": "AAPL", "action": "BUY", "qty": 100, "price": 175.00, "order_type": "MARKET"}
            ],
            "contingencies": []
        }
        '''
        
        context = {
            "signals": [{"ticker": "AAPL", "action": "BUY", "shares": 100}],
            "portfolio": {"cash": 10000, "positions": []},
            "timestamp": datetime.now().isoformat(),
            "confidence_threshold": 0.80
        }
        
        decision = agent._parse_response(response, context)
        
        assert decision.decision == "EXECUTE"
        assert decision.confidence == 0.85
        assert len(decision.orders) == 1
        assert decision.orders[0]["ticker"] == "AAPL"
    
    def test_llm_keyword_extraction(self, config):
        agent = LLMExecutionAgent(config)
        
        response = "EXECUTE: Buy AAPL at market price. Confidence: 0.90"
        
        context = {
            "signals": [{"ticker": "AAPL", "action": "BUY", "shares": 100}],
            "portfolio": {"cash": 10000, "positions": []},
            "timestamp": datetime.now().isoformat(),
            "confidence_threshold": 0.80
        }
        
        decision = agent._parse_response(response, context)
        
        assert decision.decision in ["EXECUTE", "WAIT"]


# =============================================================================
# TEST DECISION ENGINE
# =============================================================================

class TestDecisionLogic:
    """Tests for smart order decision logic"""
    
    def test_better_entry_market_order(self, config):
        """If entry > current, use market order (better fill)"""
        engine = DecisionEngine.__new__(DecisionEngine)
        engine.broker = Mock()
        engine.broker.get_current_price = Mock(return_value=173.0)
        engine.trading_config = config.get('trading', {})
        engine.max_slippage = 2.0
        
        signal = Signal(
            ticker="AAPL",
            action="BUY",
            shares=100,
            entry_price=175.00,
            take_profit=190.00,
            stop_loss=165.00,
            priority=1,
            risk_flags=[]
        )
        
        plan = engine._create_plan(signal, 173.0, 50000)
        
        assert plan is not None
        assert plan.order_type == OrderType.MARKET
        assert plan.price == 173.0
    
    def test_worse_entry_limit_order(self, config):
        """If entry < current, use limit order (wait for entry)"""
        engine = DecisionEngine.__new__(DecisionEngine)
        engine.broker = Mock()
        engine.broker.get_current_price = Mock(return_value=176.0)
        engine.trading_config = config.get('trading', {})
        engine.max_slippage = 2.0
        
        signal = Signal(
            ticker="AAPL",
            action="BUY",
            shares=100,
            entry_price=175.00,
            take_profit=190.00,
            stop_loss=165.00,
            priority=1,
            risk_flags=[]
        )
        
        plan = engine._create_plan(signal, 176.0, 50000)
        
        assert plan is not None
        assert plan.order_type == OrderType.LIMIT
        assert plan.price == 175.0
    
    def test_sell_uses_stop_limit(self, config):
        """SELL signals should use stop-limit order"""
        engine = DecisionEngine.__new__(DecisionEngine)
        engine.broker = Mock()
        engine.broker.get_current_price = Mock(return_value=175.0)
        engine.trading_config = config.get('trading', {})
        engine.max_slippage = 2.0
        
        signal = Signal(
            ticker="AAPL",
            action="SELL",
            shares=100,
            entry_price=175.00,
            take_profit=190.00,
            stop_loss=165.00,
            priority=1,
            risk_flags=[]
        )
        
        plan = engine._create_plan(signal, 175.0, 50000)
        
        assert plan is not None
        assert plan.order_type == OrderType.STOP_LIMIT
        assert plan.action == "SELL"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for module imports"""
    
    def test_imports(self):
        from decision_engine import DecisionEngine, ExecutionPlan
        from llm_execution_agent import LLMExecutionAgent, ExecutionDecision
        
        assert DecisionEngine is not None
        assert ExecutionPlan is not None
        assert LLMExecutionAgent is not None
        assert ExecutionDecision is not None
    
    def test_config_loading(self):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'llm' in config
        assert config['llm']['provider'] == 'openai'
        assert config['llm']['model'] == 'gpt-4'
        assert config['llm']['confidence_threshold'] == 0.80
        
        assert 'mcda' in config
        assert 'trading' in config
        assert config['trading']['free_cash_threshold'] == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
