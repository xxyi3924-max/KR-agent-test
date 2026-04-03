"""
MCDA Engine — Momentum-based Multi-Criteria Decision Analysis
Fee-aware liquidation prioritization
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta


@dataclass
class MCDAScore:
    """MCDA scoring result for a position"""
    ticker: str
    total_score: float
    momentum_score: float      # Higher = stronger momentum
    rsi_score: float           # Higher = more oversold (better buy)
    volume_score: float        # Higher = more liquid
    volatility_score: float    # Higher = safer (lower vol)
    decision: str              # "HOLD", "SELL", "BUY"
    reasoning: str


class MCDAMomentumEngine:
    """
    Momentum-based MCDA for:
    1. Ranking which assets to liquidate first (lowest score = sell first)
    2. Identifying assets with strongest momentum for new positions
    """
    
    def __init__(self, config: dict):
        self.config = config
        weights = config.get('mcda', {})
        
        self.momentum_weight = weights.get('momentum_weight', 0.40)
        self.rsi_weight = weights.get('rsi_weight', 0.25)
        self.volume_weight = weights.get('volume_weight', 0.20)
        self.volatility_weight = weights.get('volatility_weight', 0.15)
        self.min_threshold = weights.get('min_score_threshold', 30)
        
    def calculate_momentum(self, prices: List[float]) -> float:
        """
        Calculate 20-day price momentum
        Returns % change from 20 days ago to now
        """
        if len(prices) < 2:
            return 0
        
        # Simple momentum: (current - oldest) / oldest * 100
        oldest = prices[0]
        current = prices[-1]
        
        if oldest == 0:
            return 0
        
        return ((current - oldest) / oldest) * 100
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index)
        Returns 0-100: <30 oversold, >70 overbought
        """
        if len(prices) < period + 1:
            return 50  # Neutral
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_volume_trend(self, volumes: List[float]) -> float:
        """
        Calculate volume trend (normalized)
        Higher = more bullish volume pattern
        """
        if len(volumes) < 5:
            return 50  # Neutral
        
        # Compare recent avg volume to older avg
        recent_avg = np.mean(volumes[-5:])
        older_avg = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else np.mean(volumes[:5])
        
        if older_avg == 0:
            return 50
        
        ratio = recent_avg / older_avg
        # Normalize to 0-100 (50 = no change)
        return min(100, max(0, 50 + (ratio - 1) * 25))
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """
        Calculate volatility score
        Lower volatility = safer hold = higher score
        """
        if len(prices) < 5:
            return 50  # Neutral
        
        returns = np.diff(prices) / prices[:-1]
        std_dev = np.std(returns)
        
        # Annualize and normalize
        # Lower std dev = higher score
        annual_vol = std_dev * np.sqrt(252) * 100
        
        # Map to 0-100 (lower vol = higher score)
        # 0% vol = 100 score, 50%+ vol = 0 score
        score = max(0, min(100, 100 - annual_vol * 2))
        return score
    
    def score_position(self, ticker: str, prices: List[float], 
                       volumes: Optional[List[float]] = None,
                       current_rsi: Optional[float] = None) -> MCDAScore:
        """
        Calculate MCDA score for a single position
        """
        # Calculate individual metrics
        momentum = self.calculate_momentum(prices)
        rsi = current_rsi if current_rsi else self.calculate_rsi(prices)
        vol_score = self.calculate_volatility(prices)
        volume_score = self.calculate_volume_trend(volumes) if volumes else 50
        
        # Normalize momentum to 0-100 scale
        # Typical range: -30% to +30%
        momentum_norm = max(0, min(100, 50 + momentum * 2))
        
        # For selling: prefer positions with weak momentum
        # For holding: don't change if momentum is neutral/strong
        
        # Total weighted score
        total = (
            momentum_norm * self.momentum_weight +
            rsi * self.rsi_weight +
            volume_score * self.volume_weight +
            vol_score * self.volatility_weight
        )
        
        # Decision logic
        if total < self.min_threshold:
            decision = "SELL"
            reasoning = f"Weak momentum ({momentum:.1f}%), RSI {rsi:.0f}, score {total:.1f}"
        elif total > 70:
            decision = "HOLD"
            reasoning = f"Strong momentum ({momentum:.1f}%), RSI {rsi:.0f}, score {total:.1f}"
        else:
            decision = "HOLD"
            reasoning = f"Neutral conditions, score {total:.1f}"
        
        return MCDAScore(
            ticker=ticker,
            total_score=total,
            momentum_score=momentum,
            rsi_score=rsi,
            volume_score=volume_score,
            volatility_score=vol_score,
            decision=decision,
            reasoning=reasoning
        )
    
    def rank_for_liquidation(self, positions_data: Dict[str, Dict]) -> List[Tuple[str, float, str]]:
        """
        Rank positions by liquidation priority (lowest score = sell first)
        
        positions_data: {ticker: {"prices": [...], "volumes": [...], "rsi": float}}
        """
        rankings = []
        
        for ticker, data in positions_data.items():
            score = self.score_position(
                ticker,
                data.get("prices", []),
                data.get("volumes"),
                data.get("rsi")
            )
            rankings.append((ticker, score.total_score, score.reasoning))
        
        # Sort by score (lowest = sell first)
        rankings.sort(key=lambda x: x[1])
        return rankings
    
    def is_profitable_to_sell(self, entry_price: float, current_price: float,
                              fee_pct: float = 0.1) -> Tuple[bool, float]:
        """
        Check if selling is profitable after fees
        
        Returns: (is_profitable, net_pnl_pct)
        """
        if entry_price == 0:
            return False, 0
        
        gross_pnl = (current_price - entry_price) / entry_price * 100
        net_pnl = gross_pnl - fee_pct  # Subtract fee
        
        return net_pnl > 0, net_pnl
    
    def calculate_sell_priority(self, ticker: str, entry_price: float,
                                 current_price: float, volumes: List[float],
                                 fee_pct: float = 0.1) -> Tuple[float, str]:
        """
        Calculate sell priority for a position
        
        Lower score = higher sell priority
        Considers: MCDA score + profitability
        """
        # Get MCDA score
        mcda = self.score_position(ticker, [entry_price, current_price], volumes)
        
        # Check profitability
        profitable, net_pnl = self.is_profitable_to_sell(entry_price, current_price, fee_pct)
        
        if not profitable:
            # Don't sell at a loss unless MCDA strongly suggests it
            priority = mcda.total_score - 50  # Penalize unprofitable
            reason = f"UNPROFITABLE: net loss {net_pnl:.2f}%"
        else:
            # Higher profit = lower priority (don't rush taking profits)
            profit_factor = min(net_pnl / 10, 1) * 20  # Max 20 point reduction
            priority = mcda.total_score - profit_factor
            reason = f"PROFITABLE: +{net_pnl:.2f}% after fees"
        
        return priority, reason


# === Fee-Aware Trading Logic ===

class FeeAwareTrader:
    """Handles fee calculations and minimum trade viability"""
    
    def __init__(self, fee_pct: float = 0.1, min_trade: float = 5.0):
        """
        fee_pct: Commission as % of trade value
        min_trade: Minimum trade value to be worth it ($)
        """
        self.fee_pct = fee_pct
        self.min_trade = min_trade
    
    def is_trade_viable(self, price: float, qty: float, side: str = "BUY") -> Tuple[bool, float]:
        """
        Check if a trade is worth executing after fees
        
        Returns: (is_viable, fee_amount)
        """
        trade_value = price * qty
        
        # Minimum trade value check
        if trade_value < self.min_trade:
            return False, 0
        
        # Fee calculation
        fee = trade_value * (self.fee_pct / 100)
        
        # For buys: just need to exceed min trade
        # For sells: need profit > fee
        if side == "SELL":
            return trade_value > self.min_trade, fee
        
        return True, fee
    
    def estimate_net_proceeds(self, price: float, qty: float, 
                              entry_price: float = 0) -> Dict[str, float]:
        """Estimate net proceeds from a sale"""
        gross = price * qty
        fee = gross * (self.fee_pct / 100)
        net = gross - fee
        
        pnl = 0
        pnl_pct = 0
        if entry_price > 0:
            cost = entry_price * qty
            pnl = net - cost
            pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
        
        return {
            "gross": gross,
            "fee": fee,
            "net": net,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        }
    
    def calculate_max_affordable_qty(self, cash: float, price: float) -> float:
        """Calculate max shares affordable after fees"""
        # cash = price * qty * (1 + fee/100)
        # qty = cash / (price * (1 + fee/100))
        affordable = cash / (price * (1 + self.fee_pct / 100))
        
        # Round down to whole shares
        return int(affordable)
    
    def estimate_break_even(self, entry_price: float) -> float:
        """Calculate price needed to break even after fees"""
        # price * (1 - fee_pct/100) = entry_price
        # price = entry_price / (1 - fee_pct/100)
        return entry_price / (1 - self.fee_pct / 100)
