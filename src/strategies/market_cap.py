import pandas as pd
from typing import Dict, List
from loguru import logger
from .base import BaseStrategy


class MarketCapWeightedStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='quarterly', max_position_weight=0.10, min_position_weight=0.001):
        super().__init__(
            name="Market Cap Weighted",
            rebalance_frequency=rebalance_frequency,
            max_position_weight=max_position_weight,
            min_position_weight=min_position_weight
        )
    
    def calculate_weights(self, universe, date, data):
        market_caps = data.get('market_caps')
        if market_caps is None:
            return pd.Series(dtype=float)
        
        try:
            valid_dates = market_caps.index[market_caps.index <= date]
            if len(valid_dates) == 0:
                return pd.Series(dtype=float)
            
            current_mcaps = market_caps.loc[valid_dates[-1]]
            available = [t for t in universe if t in current_mcaps.index]
            mcaps = current_mcaps[available].dropna()
            
            if mcaps.empty:
                return pd.Series(dtype=float)
            
            weights = mcaps / mcaps.sum()
            return self.apply_constraints(weights)
            
        except Exception as e:
            logger.error(f"Error getting market caps: {e}")
            return pd.Series(dtype=float)


class LargeCapStrategy(MarketCapWeightedStrategy):
    
    def __init__(self, top_n=50, rebalance_frequency='quarterly', max_position_weight=0.10):
        super().__init__(rebalance_frequency=rebalance_frequency, max_position_weight=max_position_weight)
        self.name = f"Large Cap Top {top_n}"
        self.top_n = top_n
        self.max_positions = top_n
    
    def calculate_weights(self, universe, date, data):
        market_caps = data.get('market_caps')
        if market_caps is None:
            return pd.Series(dtype=float)
        
        try:
            valid_dates = market_caps.index[market_caps.index <= date]
            if len(valid_dates) == 0:
                return pd.Series(dtype=float)
            
            current_mcaps = market_caps.loc[valid_dates[-1]]
            available = [t for t in universe if t in current_mcaps.index]
            mcaps = current_mcaps[available].dropna()
            top_mcaps = mcaps.nlargest(self.top_n)
            
            weights = top_mcaps / top_mcaps.sum()
            return self.apply_constraints(weights)
            
        except:
            return pd.Series(dtype=float)
