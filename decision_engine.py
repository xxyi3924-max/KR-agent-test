"""
Decision Engine - Brain of the Agent
Uses LLM for signal decisions, MCDA for cash management only
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import yaml
import random

from tdash_connector import TDashConnector, Portfolio, Signal, Holding
from kr_broker import KRBrokerConnector, Order, OrderType, OrderSide
from mcda_engine import MCDAMomentumEngine

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Plan for executing a single order"""
    ticker: str
    action: str  # 'BUY', 'SELL', 'TRIM'
    qty: float
    order_type: OrderType
    price: float
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    contingency: Optional[Dict] = None


@dataclass
class ExecutionResult:
    """Result of executing a plan"""
    ticker: str
    action: str
    qty: float
    fill_price: float
    success: bool
    message: str = ""
    order_id: Optional[str] = None


# =============================================================================
# LLM EXECUTION AGENT (Integrated)
# =============================================================================

@dataclass
class ExecutionDecision:
    """LLM's decision on what to execute"""
    decision: str  # 'EXECUTE', 'WAIT', 'SKIP'
    reasoning: str
    confidence: float
    orders: List[Dict] = field(default_factory=list)  # [{ticker, action, qty, order_type, price}]
    contingencies: List[Dict] = field(default_factory=list)
    execution_time_ms: int = 0


class LLMExecutionAgent:
    """
    LLM-powered execution agent using OpenAI GPT-4
    Decides which signals to execute and handles contingencies
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.llm_config = config.get('llm', {})
        
        # LLM settings
        self.provider = self.llm_config.get('provider', 'openai')
        self.model = self.llm_config.get('model', 'gpt-4')
        self.confidence_threshold = self.llm_config.get('confidence_threshold', 0.80)
        self.max_iterations = self.llm_config.get('max_iterations', 3)
        self.timeout = self.llm_config.get('timeout_seconds', 10)
        self.fallback = self.llm_config.get('fallback_to_hardcoded', True)
        self.max_cost = self.llm_config.get('max_cost_per_run', 1.00)
        self.log_decisions = self.llm_config.get('log_all_decisions', True)
        
        # Initialize OpenAI client
        self.client = None
        if self.provider == 'openai':
            try:
                import openai
                api_key = os.environ.get('OPENAI_API_KEY') or self.llm_config.get('openai_api_key', '')
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    logger.info(f"LLM Agent initialized with {self.model}")
                else:
                    logger.warning("OpenAI API key not set, LLM agent disabled")
            except ImportError:
                logger.warning("OpenAI package not installed, LLM agent disabled")
    
    def decide(
        self,
        signals: List[Dict],
        portfolio: Dict,
        market_context: Optional[Dict] = None,
        execution_history: Optional[List[Dict]] = None
    ) -> ExecutionDecision:
        """
        Main entry point: Get LLM decision for execution.
        
        Args:
            signals: TDash signals [{ticker, action, qty, entry, tp, sl, ...}]
            portfolio: Current portfolio state
            market_context: Optional market data (VIX, sector rotation, etc.)
            execution_history: Recent execution results for context
        
        Returns:
            ExecutionDecision with orders, contingencies, and reasoning
        """
        start_time = time.time()
        
        # Build context for LLM
        context = self._build_context(signals, portfolio, market_context, execution_history)
        
        # Check if LLM is available
        if not self.client:
            logger.warning("LLM not available, returning skip decision")
            return ExecutionDecision(
                decision="WAIT",
                reasoning="LLM client not initialized",
                confidence=0.0,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Build prompt
        prompt = self._build_prompt(context)
        
        # Call LLM
        try:
            response = self._call_llm(prompt)
            decision = self._parse_response(response, context)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            decision = ExecutionDecision(
                decision="WAIT",
                reasoning=f"LLM error: {str(e)}",
                confidence=0.0,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Log decision
        if self.log_decisions:
            self._log_decision(context, decision)
        
        decision.execution_time_ms = int((time.time() - start_time) * 1000)
        return decision
    
    def _build_context(
        self,
        signals: List[Dict],
        portfolio: Dict,
        market_context: Optional[Dict] = None,
        execution_history: Optional[List[Dict]] = None
    ) -> Dict:
        """Build context dict for LLM"""
        return {
            "signals": signals,
            "portfolio": portfolio,
            "market_context": market_context or {},
            "execution_history": execution_history or [],
            "timestamp": datetime.now().isoformat(),
            "confidence_threshold": self.confidence_threshold,
            "max_iterations": self.max_iterations
        }
    
    def _build_prompt(self, context: Dict) -> str:
        """Build prompt for GPT-4"""
        signals_str = self._format_signals(context['signals'])
        portfolio_str = self._format_portfolio(context['portfolio'])
        market_str = self._format_market(context['market_context'])
        history_str = self._format_history(context['execution_history'])
        
        return f"""You are an execution agent for a quantitative trading system.

GOALS:
1. Maximize fill quality (better entry = market order)
2. Minimize transaction costs (avoid overtrading)
3. Preserve capital (don't over-leverage)
4. Handle contingencies gracefully
5. Be decisive - only WAIT if truly uncertain

RULES:
- Free cash < $10,000 → you MUST liquidate (MCDA will handle, but flag it)
- Free cash > $10,000 → auto-invest in QQQ (will happen automatically)
- If entry > current price → market order (better fill!)
- If entry < current price → limit order at entry
- Only execute if gain > 0.5% after fees
- Confidence < {context['confidence_threshold']} → WAIT or SKIP

{signals_str}

{portfolio_str}

{market_str}

{history_str}

DECIDE: Execute? Wait? Skip? Provide JSON response.
"""
    
    def _format_signals(self, signals: List[Dict]) -> str:
        if not signals:
            return "SIGNALS: None"
        lines = ["SIGNALS:"]
        for s in signals:
            lines.append(f"  - {s.get('action')}: {s.get('ticker')} | qty={s.get('qty')} | entry=${s.get('entry_price', 0):.2f} | TP=${s.get('tp_price', 0):.2f} | SL=${s.get('sl_price', 0):.2f} | current=${s.get('current_price', 0):.2f}")
        return "\n".join(lines)
    
    def _format_portfolio(self, portfolio: Dict) -> str:
        cash = portfolio.get('cash', 0)
        equity = portfolio.get('equity', 0)
        positions = portfolio.get('positions', [])
        
        lines = [f"PORTFOLIO: Cash=${cash:.2f} | Equity=${equity:.2f}"]
        if positions:
            lines.append("Positions:")
            for p in positions[:10]:  # Top 10
                lines.append(f"  - {p.get('ticker')}: {p.get('qty')} shares @ ${p.get('avg_cost', 0):.2f} (unrealized: ${p.get('unrealized_pnl', 0):.2f})")
        return "\n".join(lines)
    
    def _format_market(self, market: Dict) -> str:
        if not market:
            return "MARKET: No data"
        lines = ["MARKET:"]
        if 'vix' in market:
            lines.append(f"  - VIX: {market['vix']:.2f}")
        if 'direction' in market:
            lines.append(f"  - Direction: {market['direction']}")
        if 'reason' in market:
            lines.append(f"  - Reason: {market['reason']}")
        return "\n".join(lines)
    
    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "HISTORY: No recent trades"
        lines = ["HISTORY:"]
        for h in history[-5:]:  # Last 5
            lines.append(f"  - {h.get('ticker')}: {h.get('action')} {h.get('qty')} @ ${h.get('price', 0):.2f} - {'SUCCESS' if h.get('success') else 'FAILED'}")
        return "\n".join(lines)
    
    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a quantitative trading execution agent. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temp for consistent decisions
            max_tokens=1000,
            timeout=self.timeout
        )
        return response.choices[0].message.content
    
    def _parse_response(self, response: str, context: Dict) -> ExecutionDecision:
        """Parse LLM JSON response"""
        import json
        
        # Try to extract JSON from response
        try:
            # Find JSON block
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            data = json.loads(json_str.strip())
            
            # Validate required fields
            decision = data.get('decision', 'WAIT').upper()
            if decision not in ['EXECUTE', 'WAIT', 'SKIP']:
                decision = 'WAIT'
            
            return ExecutionDecision(
                decision=decision,
                reasoning=data.get('reasoning', 'No reasoning provided'),
                confidence=data.get('confidence', 0.5),
                orders=data.get('orders', []),
                contingencies=data.get('contingencies', [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Try simple keyword extraction
            response_lower = response.lower()
            if 'execute' in response_lower and 'skip' not in response_lower:
                return ExecutionDecision(
                    decision="EXECUTE",
                    reasoning="LLM response suggests execution",
                    confidence=0.6,
                    orders=[],
                    contingencies=[]
                )
            return ExecutionDecision(
                decision="WAIT",
                reasoning=f"Failed to parse LLM response: {str(e)}",
                confidence=0.0,
                orders=[],
                contingencies=[]
            )
    
    def _log_decision(self, context: Dict, decision: ExecutionDecision):
        """Log decision for audit"""
        log_entry = {
            "timestamp": context['timestamp'],
            "signals_count": len(context['signals']),
            "decision": decision.decision,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "orders_count": len(decision.orders),
            "execution_time_ms": decision.execution_time_ms
        }
        logger.info(f"LLM Decision: {log_entry}")


# =============================================================================
# DECISION ENGINE
# =============================================================================

class DecisionEngine:
    """
    Core decision engine that:
    1. Uses LLM for signal decisions (EXECUTE, WAIT, SKIP)
    2. Uses MCDA only for free cash management
    3. Manages TP/SL orders via KR Broker
    4. Handles contingencies
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize all connectors"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize connectors
        tdash_config = self.config.get('tdash', {})
        self.tdash = TDashConnector(
            data_dir=os.path.expanduser(tdash_config.get('data_dir')),
            user=tdash_config.get('user', 'admin')
        )
        
        broker_config = self.config.get('kr_broker', {})
        self.broker = KRBrokerConnector(
            api_url=broker_config.get('api_url'),
            api_key=broker_config.get('api_key')
        )
        
        # Initialize MCDA (only for free cash management)
        mcda_config = self.config.get('mcda', {})
        self.mcda = MCDAMomentumEngine(
            momentum_weight=mcda_config.get('momentum_weight', 0.40),
            rsi_weight=mcda_config.get('rsi_weight', 0.25),
            volume_weight=mcda_config.get('volume_weight', 0.20),
            volatility_weight=mcda_config.get('volatility_weight', 0.15)
        )
        
        # Initialize LLM Execution Agent
        self.llm_agent = LLMExecutionAgent(self.config)
        
        # Trading settings
        self.trading_config = self.config.get('trading', {})
        self.free_cash_threshold = self.trading_config.get('free_cash_threshold', 10000)
        self.etf_preference = self.trading_config.get('etf_preference', 'QQQ')
        self.max_slippage = self.trading_config.get('max_slippage_pct', 2.0)
        
        logger.info("Decision Engine initialized with LLM agent")
    
    def run_cycle(self) -> Dict:
        """
        Run a full decision cycle using LLM for signal decisions
        
        Returns summary of actions taken
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "llm_decision": None,
            "plans_created": 0,
            "executions": [],
            "auto_invest": [],
            "cash_liquidations": [],
            "errors": []
        }
        
        # 1. Get portfolio from KR Broker
        try:
            account = self.broker.get_account_info()
            portfolio_state = {
                "cash": account.cash,
                "equity": account.equity,
                "positions": [
                    {
                        "ticker": p.ticker,
                        "qty": p.qty,
                        "avg_cost": p.avg_cost,
                        "current_price": self.broker.get_current_price(p.ticker),
                        "unrealized_pnl": account.unrealized_pnl.get(p.ticker, 0)
                    }
                    for p in account.positions
                ]
            }
            summary["portfolio"] = portfolio_state
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            summary["errors"].append(f"Portfolio load: {e}")
            return summary
        
        # 2. Get TDash signals
        try:
            tdash_portfolio = self.tdash.get_portfolio()
            signals = tdash_portfolio.signals
            summary["signals"] = {
                "count": len(signals),
                "actions": list(set(s.action for s in signals))
            }
        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            summary["errors"].append(f"Signals load: {e}")
            return summary
        
        # 3. Check market conditions
        is_major_move, reason = self.check_market_movement()
        summary["market_check"] = {"is_major_move": is_major_move, "reason": reason}
        
        # 4. Use LLM to decide execution
        signals_dict = [
            {
                "ticker": s.ticker,
                "action": s.action,
                "qty": s.qty,
                "entry_price": s.entry_price,
                "tp_price": s.tp_price,
                "sl_price": s.sl_price,
                "current_price": self.broker.get_current_price(s.ticker) if s.ticker else 0
            }
            for s in signals
        ]
        
        llm_decision = self.llm_agent.decide(
            signals=signals_dict,
            portfolio=portfolio_state,
            market_context={"vix": 15, "direction": "unknown", "reason": reason}
        )
        summary["llm_decision"] = {
            "decision": llm_decision.decision,
            "confidence": llm_decision.confidence,
            "reasoning": llm_decision.reasoning
        }
        
        # 5. Execute based on LLM decision
        if llm_decision.decision == "EXECUTE" and llm_decision.confidence >= self.llm_agent.confidence_threshold:
            logger.info(f"LLM approved execution (confidence: {llm_decision.confidence:.2f})")
            
            # Convert LLM orders to execution plans
            plans = self._llm_orders_to_plans(llm_decision.orders)
            summary["plans_created"] = len(plans)
            
            # Check if we have enough cash
            required_cash = sum(
                p.qty * p.price for p in plans 
                if p.action in ['BUY', 'ADD']
            )
            
            if portfolio_state["cash"] < required_cash:
                # Need to liquidate - use MCDA
                logger.info(f"Cash shortage: ${portfolio_state['cash']:.2f} < ${required_cash:.2f}")
                shortfall = required_cash - portfolio_state["cash"]
                liquidations = self.liquidate_for_cash(shortfall)
                summary["cash_liquidations"] = [
                    {"ticker": r.ticker, "qty": r.qty, "raised": r.fill_price * r.qty}
                    for r in liquidations
                ]
            
            # Execute plans
            if plans:
                results = self.execute_plans(plans)
                summary["executions"] = [
                    {"ticker": r.ticker, "action": r.action, "success": r.success, "message": r.message}
                    for r in results
                ]
        
        elif llm_decision.decision == "WAIT":
            logger.info(f"LLM said WAIT: {llm_decision.reasoning}")
        
        else:  # SKIP or low confidence
            logger.info(f"LLM said SKIP (confidence: {llm_decision.confidence:.2f}): {llm_decision.reasoning}")
        
        # 6. Auto-invest free cash in QQQ (MCDA only for cash management)
        invest_results = self.manage_free_cash()
        summary["auto_invest"] = [
            {"ticker": r.ticker, "qty": r.qty, "price": r.fill_price}
            for r in invest_results
        ]
        
        # 7. Update TDash with new state (simple)
        try:
            account = self.broker.get_account_info()
            self.tdash.update_cash(account.cash)
        except Exception as e:
            logger.warning(f"TDash update failed: {e}")
        
        return summary
    
    def _llm_orders_to_plans(self, orders: List[Dict]) -> List[ExecutionPlan]:
        """Convert LLM orders to execution plans"""
        plans = []
        for order in orders:
            ticker = order.get('ticker', '')
            action = order.get('action', '').upper()
            qty = order.get('qty', 0)
            order_type_str = order.get('order_type', 'MARKET').upper()
            
            # Map order type
            if order_type_str == 'LIMIT':
                order_type = OrderType.LIMIT
            elif order_type_str == 'STOP_LIMIT':
                order_type = OrderType.STOP_LIMIT
            else:
                order_type = OrderType.MARKET
            
            # Get current price
            current_price = self.broker.get_current_price(ticker)
            entry_price = order.get('price', current_price)
            
            # Smart order: market if entry > current, limit if entry < current
            if action in ['BUY', 'ADD'] and entry_price > current_price:
                price = current_price  # Market order
                order_type = OrderType.MARKET
            else:
                price = entry_price
            
            plan = ExecutionPlan(
                ticker=ticker,
                action=action,
                qty=qty,
                order_type=order_type,
                price=price,
                tp_price=order.get('tp_price'),
                sl_price=order.get('sl_price'),
                contingency=order.get('contingency')
            )
            plans.append(plan)
        
        return plans
    
    def analyze_signals(self, portfolio: Portfolio) -> List[ExecutionPlan]:
        """
        Analyze signals and create execution plans (fallback if LLM fails)
        """
        plans = []
        
        for signal in portfolio.signals:
            if signal.action in ['NEW', 'ADD']:
                current_price = self.broker.get_current_price(signal.ticker)
                if current_price > 0:
                    plan = self._create_plan(signal, current_price, portfolio.cash)
                    if plan:
                        plans.append(plan)
            elif signal.action in ['SELL', 'TRIM']:
                # Find position in portfolio
                position = next((h for h in portfolio.holdings if h.ticker == signal.ticker), None)
                if position:
                    current_price = self.broker.get_current_price(signal.ticker)
                    plan = self._create_plan(signal, current_price, portfolio.cash)
                    if plan:
                        plans.append(plan)
        
        return plans
    
    def _create_plan(self, signal: Signal, current_price: float, available_cash: float) -> Optional[ExecutionPlan]:
        """
        Create execution plan using smart order logic
        """
        if signal.action in ['NEW', 'ADD', 'BUY']:
            required = signal.shares * signal.entry_price
            if required > available_cash * 0.95:  # Leave 5% buffer
                logger.warning(f"Insufficient cash for {signal.ticker}: ${required:.2f} > ${available_cash:.2f}")
                return None
            
            # Calculate price gap
            if signal.entry_price > 0:
                gap_pct = ((current_price - signal.entry_price) / signal.entry_price) * 100
            else:
                gap_pct = 0
            
            # Alert on large gap
            if abs(gap_pct) > self.max_slippage:
                logger.warning(f"Large price gap for {signal.ticker}: {gap_pct:.2f}%")
            
            # Smart order: use market order if current price is better than entry
            if current_price < signal.entry_price:
                # Better entry! Use market order
                order_type = OrderType.MARKET
                order_price = current_price
            else:
                # Worse entry, use limit order at entry price
                order_type = OrderType.LIMIT
                order_price = signal.entry_price
            
            return ExecutionPlan(
                ticker=signal.ticker,
                action='BUY',
                qty=signal.shares,
                order_type=order_type,
                price=order_price,
                tp_price=signal.take_profit,
                sl_price=signal.stop_loss
            )
        
        elif signal.action in ['SELL', 'TRIM', 'SELL_ONLY']:
            return ExecutionPlan(
                ticker=signal.ticker,
                action='SELL',
                qty=signal.shares,
                order_type=OrderType.STOP_LIMIT,
                price=signal.stop_loss or current_price,
                tp_price=signal.take_profit,
                sl_price=signal.stop_loss
            )
        
        return None
    
    def execute_plans(self, plans: List[ExecutionPlan]) -> List[ExecutionResult]:
        """Execute all plans with TP/SL handling"""
        results = []
        
        for plan in plans:
            result = self._execute_single(plan)
            results.append(result)
            
            # After successful buy, set TP/SL
            if result.success and plan.action in ['BUY', 'ADD']:
                if plan.tp_price or plan.sl_price:
                    self._set_tp_sl_orders(plan)
        
        return results
    
    def _execute_single(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute a single plan"""
        try:
            # Place order
            order = Order(
                ticker=plan.ticker,
                side=OrderSide.BUY if plan.action in ['BUY', 'ADD'] else OrderSide.SELL,
                qty=plan.qty,
                order_type=plan.order_type,
                price=plan.price,
                stop_price=plan.sl_price if plan.order_type == OrderType.STOP_LIMIT else None
            )
            
            order_result = self.broker.place_order(order)
            
            return ExecutionResult(
                ticker=plan.ticker,
                action=plan.action,
                qty=plan.qty,
                fill_price=order_result.get('fill_price', plan.price),
                success=True,
                message="Order placed",
                order_id=order_result.get('order_id')
            )
            
        except Exception as e:
            logger.error(f"Failed to execute {plan.ticker}: {e}")
            return ExecutionResult(
                ticker=plan.ticker,
                action=plan.action,
                qty=plan.qty,
                fill_price=plan.price,
                success=False,
                message=str(e)
            )
    
    def _set_tp_sl_orders(self, plan: ExecutionPlan):
        """Set take-profit and stop-loss orders after buy"""
        account = self.broker.get_account_info()
        position = next((p for p in account.positions if p.ticker == plan.ticker), None)
        
        if not position:
            return
        
        # Stop-loss
        if plan.sl_price:
            sl_order = Order(
                ticker=plan.ticker,
                side=OrderSide.SELL,
                qty=position.qty,
                order_type=OrderType.STOP_LIMIT,
                price=plan.sl_price,
                stop_price=plan.sl_price
            )
            self.broker.place_order(sl_order)
            logger.info(f"Set SL for {plan.ticker} @ ${plan.sl_price:.2f}")
        
        # Take-profit
        if plan.tp_price:
            tp_order = Order(
                ticker=plan.ticker,
                side=OrderSide.SELL,
                qty=position.qty,
                order_type=OrderType.LIMIT,
                price=plan.tp_price
            )
            self.broker.place_order(tp_order)
            logger.info(f"Set TP for {plan.ticker} @ ${plan.tp_price:.2f}")
    
    def manage_free_cash(self) -> List[ExecutionResult]:
        """
        Auto-invest free cash > $10K into QQQ (MCDA only for cash management)
        """
        results = []
        
        try:
            account = self.broker.get_account_info()
            free_cash = account.cash
            
            if free_cash <= self.free_cash_threshold:
                logger.info(f"Free cash ${free_cash:.2f} below threshold ${self.free_cash_threshold}")
                return results
            
            # Invest excess in QQQ
            invest_amount = free_cash - self.free_cash_threshold
            etf_price = self.broker.get_current_price(self.etf_preference)
            
            if etf_price <= 0:
                etf_price = self.broker.get_current_price('SPY')
                if etf_price <= 0:
                    logger.warning("Could not get ETF price")
                    return results
            
            # Fee-aware: only invest if viable
            fee = max(invest_amount * 0.001, 1)  # 0.1% or $1 min
            if invest_amount - fee < 100:  # Min $100 investment
                logger.info(f"Investment too small after fees: ${invest_amount - fee:.2f}")
                return results
            
            qty = (invest_amount - fee) / etf_price
            
            # Execute market order
            order = Order(
                ticker=self.etf_preference,
                side=OrderSide.BUY,
                qty=qty,
                order_type=OrderType.MARKET,
                price=etf_price
            )
            
            result = self.broker.place_order(order)
            results.append(ExecutionResult(
                ticker=self.etf_preference,
                action='BUY',
                qty=qty,
                fill_price=etf_price,
                success=True,
                message=f"Auto-invested ${invest_amount:.2f}"
            ))
            
            logger.info(f"Auto-invested ${invest_amount:.2f} in {self.etf_preference}")
            
        except Exception as e:
            logger.error(f"Auto-invest failed: {e}")
        
        return results
    
    def liquidate_for_cash(self, amount_needed: float) -> List[ExecutionResult]:
        """
        Liquidate positions to raise cash (MCDA ranks by momentum, lowest first)
        """
        results = []
        
        try:
            account = self.broker.get_account_info()
            positions = account.positions
            
            if not positions:
                logger.warning("No positions to liquidate")
                return results
            
            # Get current prices
            prices = {}
            for pos in positions:
                prices[pos.ticker] = self.broker.get_current_price(pos.ticker)
            
            # Score positions with MCDA
            scored = []
            for pos in positions:
                # Skip ETF positions
                if pos.ticker in [self.etf_preference, 'SPY', 'QQQ']:
                    continue
                
                score = self.mcda.score_position(pos, prices.get(pos.ticker, pos.avg_cost))
                scored.append((score, pos))
            
            # Sort by score (lowest = sell first)
            scored.sort(key=lambda x: x[0])
            
            # Liquidate starting from lowest score
            raised = 0
            for score, pos in scored:
                if raised >= amount_needed:
                    break
                
                price = prices.get(pos.ticker, pos.avg_cost)
                order = Order(
                    ticker=pos.ticker,
                    side=OrderSide.SELL,
                    qty=pos.qty,
                    order_type=OrderType.MARKET,
                    price=price
                )
                
                try:
                    result = self.broker.place_order(order)
                    raised += pos.qty * price
                    results.append(ExecutionResult(
                        ticker=pos.ticker,
                        action='SELL',
                        qty=pos.qty,
                        fill_price=price,
                        success=True,
                        message=f"MCDA liquidation (score: {score:.2f})"
                    ))
                    logger.info(f"MCDA liquidating {pos.ticker} (score: {score:.2f})")
                except Exception as e:
                    logger.error(f"Failed to liquidate {pos.ticker}: {e}")
            
            logger.info(f"MCDA raised ${raised:.2f} from liquidations")
            
        except Exception as e:
            logger.error(f"MCDA liquidation failed: {e}")
        
        return results
    
    def check_market_movement(self) -> Tuple[bool, str]:
        """
        Check if major market movement detected
        Returns (is_major_move, reason)
        """
        try:
            # Simple VIX check
            vix = self.broker.get_current_price('VIX') or 15  # Default to 15
            
            if vix > 25:
                return True, f"High VIX: {vix:.2f}"
            elif vix < 12:
                return True, f"Low VIX (calm market): {vix:.2f}"
            else:
                return True, f"Normal VIX: {vix:.2f}"  # Always allow execution for now
            
        except Exception as e:
            logger.warning(f"Market check failed: {e}")
            return True, "Market check unavailable"
