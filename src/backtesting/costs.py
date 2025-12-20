import numpy as np
from loguru import logger


class TransactionCostModel:
    
    def __init__(self, commission_rate=0.001, slippage_rate=0.001, bid_ask_spread=0.0005,
                 min_commission=0.0, max_commission=None, market_impact_factor=0.0):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.bid_ask_spread = bid_ask_spread
        self.min_commission = min_commission
        self.max_commission = max_commission
        self.market_impact_factor = market_impact_factor
        logger.info(f"TransactionCostModel: commission={commission_rate:.2%}, slippage={slippage_rate:.2%}, spread={bid_ask_spread:.2%}")
    
    def calculate_commission(self, trade_value):
        commission = abs(trade_value) * self.commission_rate
        commission = max(commission, self.min_commission)
        if self.max_commission is not None:
            commission = min(commission, self.max_commission)
        return commission
    
    def calculate_slippage(self, trade_value, volatility=None):
        base = abs(trade_value) * self.slippage_rate
        spread_cost = abs(trade_value) * (self.bid_ask_spread / 2)
        
        if volatility is not None and volatility > 0:
            base += abs(trade_value) * volatility * 0.01
        
        return base + spread_cost
    
    def calculate_total_cost(self, trade_value, volatility=None):
        return self.calculate_commission(trade_value) + self.calculate_slippage(trade_value, volatility)
    
    @classmethod
    def zero_cost(cls):
        return cls(commission_rate=0.0, slippage_rate=0.0, bid_ask_spread=0.0)
    
    @classmethod
    def retail(cls):
        return cls(commission_rate=0.0, slippage_rate=0.002, bid_ask_spread=0.001)
    
    @classmethod
    def institutional(cls):
        return cls(commission_rate=0.0005, slippage_rate=0.0005, bid_ask_spread=0.0002, market_impact_factor=0.1)
