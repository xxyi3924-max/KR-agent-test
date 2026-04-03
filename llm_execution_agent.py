"""
LLM Execution Agent
Uses GPT-4 to make intelligent execution decisions with contingencies.
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    SKIP = "skip"


class Decision(str, Enum):
    EXECUTE = "execute"
    WAIT = "wait"
    PARTIAL = "partial"  # Execute with reduced size


@dataclass
class ExecutionOrder:
    ticker: str
    action: str  # BUY, SELL, TRIM
    quantity: int
    order_type: OrderType
    price: float  # Target price for limit/stop, or expected price for market
    stop_price: Optional[float] = None  # For stop-loss
    limit_price: Optional[float] = None  # For take-profit
    reasoning: str = ""
    confidence: float = 0.0  # 0.0 - 1.0
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "price": self.price,
            "stop_price": self.stop_price,
            "limit_price": self.limit_price,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }


@dataclass
class Contingency:
    condition: str  # e.g., "order_rejected", "partial_fill"
    action: str  # e.g., "retry_higher", "skip"
    trigger_price_adjustment: float = 0.0  # e.g., 0.01 = +1%


@dataclass
class ExecutionDecision:
    decision: Decision
    orders: List[ExecutionOrder] = field(default_factory=list)
    contingencies: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    llm_tokens_used: int = 0
    execution_time_ms: int = 0
    
    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "orders": [o.to_dict() for o in self.orders],
            "contingencies": self.contingencies,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "llm_tokens_used": self.llm_tokens_used,
            "execution_time_ms": self.execution_time_ms
        }


class LLMExecutionAgent:
    """
    LLM-powered execution agent that decides HOW to execute TDash signals.
    
    Behavior:
    - Receives signals, portfolio, and market context
    - Uses GPT-4 to decide: execute, wait, or partial
    - Provides smart order type selection (market vs limit)
    - Handles contingencies
    - Falls back to hardcoded logic if confidence < threshold
    """
    
    def __init__(
        self,
        config: dict,
        model: str = "gpt-4",
        confidence_threshold: float = 0.80,
        max_iterations: int = 3,
        timeout_seconds: int = 10
    ):
        self.config = config
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        
        # Initialize OpenAI client
        self.client = None
        if OPENAI_AVAILABLE:
            api_key = config.get('llm', {}).get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not found in config or environment")
        else:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
    
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
                decision=Decision.WAIT,
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
                decision=Decision.WAIT,
                reasoning=f"LLM error: {str(e)}",
                confidence=0.0,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Update execution time
        decision.execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Log decision
        self._log_decision(decision)
        
        return decision
    
    def _build_context(
        self,
        signals: List[Dict],
        portfolio: Dict,
        market_context: Optional[Dict],
        execution_history: Optional[List[Dict]]
    ) -> Dict:
        """Build structured context for LLM"""
        
        # Extract free cash
        cash = portfolio.get('cash', 0)
        buying_power = portfolio.get('buying_power', cash)
        
        # Calculate positions info
        positions = portfolio.get('positions', [])
        total_unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
        
        return {
            "signals": signals,
            "portfolio": {
                "cash": cash,
                "buying_power": buying_power,
                "total_equity": portfolio.get('total_equity', cash),
                "positions_count": len(positions),
                "total_unrealized_pnl": total_unrealized_pnl
            },
            "market_context": market_context or {},
            "execution_history": execution_history or [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _build_prompt(self, context: Dict) -> str:
        """Build LLM prompt from context"""
        
        system_prompt = """You are an execution agent for a quantitative trading system.

GOALS:
1. Maximize fill quality (better entry = market order)
2. Minimize transaction costs (avoid overtrading)
3. Preserve capital (don't over-leverage)
4. Handle contingencies gracefully

RULES:
- Free cash < $10,000 → signal need for liquidation
- Free cash > $10,000 → auto-invest in QQQ (handled separately)
- If signal entry price > current price → market order (better entry!)
- If signal entry price < current price → limit order at entry price
- Only execute if potential gain > 0.5% after fees
- VIX > 25 → reduce risk, consider smaller positions or wait
- Gap > 2% at open → evaluate market order or wait

OUTPUT FORMAT:
Respond with valid JSON only:
{
  "decision": "execute" | "wait" | "partial",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "orders": [
    {
      "ticker": "AAPL",
      "action": "BUY" | "SELL" | "TRIM",
      "quantity": 100,
      "order_type": "market" | "limit" | "stop_limit",
      "price": 175.00,
      "stop_price": 170.00,
      "limit_price": 180.00,
      "reasoning": "why this order type",
      "confidence": 0.85
    }
  ],
  "contingencies": [
    {
      "condition": "order_rejected" | "partial_fill" | "vix_spike" | "gap_large",
      "action": "retry" | "skip" | "reduce_50pct" | "wait",
      "price_adjustment": 0.0
    }
  ]
}

IMPORTANT:
- confidence < 0.80 → use "wait" or "partial" with reduced size
- Always provide contingencies for edge cases
- Only include orders that pass confidence threshold
- JSON only, no markdown or explanation outside JSON"""

        user_prompt = f"""SIGNALS FROM TDASH v9:
{json.dumps(context['signals'], indent=2)}

PORTFOLIO STATE:
- Cash: ${context['portfolio']['cash']:,.2f}
- Buying Power: ${context['portfolio']['buying_power']:,.2f}
- Total Equity: ${context['portfolio']['total_equity']:,.2f}
- Positions: {context['portfolio']['positions_count']}
- Unrealized PnL: ${context['portfolio']['total_unrealized_pnl']:,.2f}

MARKET CONTEXT:
{json.dumps(context['market_context'], indent=2)}

RECENT EXECUTION HISTORY:
{json.dumps(context['execution_history'][-5:], indent=2)}

TIMESTAMP: {context['timestamp']}

DECIDE: Execute? Wait? Partial? Provide plan in JSON format."""

        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _call_llm(self, prompt: Dict) -> str:
        """Call OpenAI API"""
        import time
        start = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ],
            temperature=0.3,  # Low temp for consistent decisions
            max_tokens=2000,
            timeout=self.timeout_seconds
        )
        
        # Track tokens
        total_tokens = response.usage.total_tokens if response.usage else 0
        
        return {
            "content": response.choices[0].message.content,
            "tokens": total_tokens,
            "time_ms": int((time.time() - start) * 1000)
        }
    
    def _parse_response(self, response: Dict, context: Dict) -> ExecutionDecision:
        """Parse LLM JSON response into ExecutionDecision"""
        
        try:
            content = response["content"]
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content.strip())
            
            # Parse orders
            orders = []
            for order_data in data.get("orders", []):
                order = ExecutionOrder(
                    ticker=order_data["ticker"],
                    action=order_data["action"],
                    quantity=order_data["quantity"],
                    order_type=OrderType(order_data.get("order_type", "market")),
                    price=order_data.get("price", 0),
                    stop_price=order_data.get("stop_price"),
                    limit_price=order_data.get("limit_price"),
                    reasoning=order_data.get("reasoning", ""),
                    confidence=order_data.get("confidence", 0.5)
                )
                orders.append(order)
            
            # Parse decision
            decision = Decision(data.get("decision", "wait"))
            
            # Check confidence threshold
            confidence = data.get("confidence", 0.0)
            if confidence < self.confidence_threshold:
                if decision == Decision.EXECUTE:
                    decision = Decision.PARTIAL
                    logger.warning(f"Confidence {confidence:.2f} < {self.confidence_threshold}, reducing to PARTIAL")
            
            return ExecutionDecision(
                decision=decision,
                orders=orders,
                contingencies=data.get("contingencies", []),
                reasoning=data.get("reasoning", ""),
                confidence=confidence,
                llm_tokens_used=response.get("tokens", 0),
                execution_time_ms=response.get("time_ms", 0)
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response content: {response.get('content', '')}")
            
            return ExecutionDecision(
                decision=Decision.WAIT,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                confidence=0.0,
                llm_tokens_used=response.get("tokens", 0)
            )
    
    def _log_decision(self, decision: ExecutionDecision):
        """Log decision for audit trail"""
        logger.info(
            f"LLM Decision: {decision.decision.value} "
            f"(confidence: {decision.confidence:.2f}, "
            f"orders: {len(decision.orders)}, "
            f"tokens: {decision.llm_tokens_used}, "
            f"time: {decision.execution_time_ms}ms)"
        )
        
        # Log individual orders
        for order in decision.orders:
            logger.info(
                f"  → {order.action} {order.quantity} {order.ticker} "
                f"@ {order.order_type.value} "
                f"(price: ${order.price:.2f}, confidence: {order.confidence:.2f})"
            )
        
        # Log contingencies
        if decision.contingencies:
            logger.info(f"  Contingencies: {len(decision.contingencies)}")


def create_llm_agent(config: dict) -> LLMExecutionAgent:
    """Factory function to create LLM agent from config"""
    llm_config = config.get('llm', {})
    
    return LLMExecutionAgent(
        config=config,
        model=llm_config.get('model', 'gpt-4'),
        confidence_threshold=llm_config.get('confidence_threshold', 0.80),
        max_iterations=llm_config.get('max_iterations', 3),
        timeout_seconds=llm_config.get('timeout_seconds', 10)
    )
