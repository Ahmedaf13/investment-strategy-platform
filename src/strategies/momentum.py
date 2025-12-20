import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger
from .base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='monthly', lookback_period=252, skip_period=21,
                 top_percentile=0.20, weighting='equal'):
        super().__init__(name="Momentum Strategy", rebalance_frequency=rebalance_frequency, max_position_weight=0.10)
        self.lookback_period = lookback_period
        self.skip_period = skip_period
        self.top_percentile = top_percentile
        self.weighting = weighting
    
    def calculate_weights(self, universe, date, data):
        prices = data.get('prices')
        if prices is None:
            return pd.Series(dtype=float)
        
        if isinstance(prices.columns, pd.MultiIndex):
            close_prices = prices.xs('Close', axis=1, level=1)
        else:
            close_prices = prices
        
        momentum = self._calculate_momentum(close_prices, date)
        if momentum.empty:
            return pd.Series(dtype=float)
        
        available = [t for t in universe if t in momentum.index]
        momentum = momentum[available].dropna()
        
        if momentum.empty:
            return pd.Series(dtype=float)
        
        n_select = max(1, int(len(momentum) * self.top_percentile))
        top_momentum = momentum.nlargest(n_select)
        
        if self.weighting == 'equal':
            weights = pd.Series(1.0 / len(top_momentum), index=top_momentum.index)
        else:
            shifted = top_momentum - top_momentum.min() + 0.001
            weights = shifted / shifted.sum()
        
        return self.apply_constraints(weights)
    
    def _calculate_momentum(self, prices, date):
        try:
            historical = prices.loc[:date]
            if len(historical) < self.lookback_period:
                return pd.Series(dtype=float)
            
            if self.skip_period > 0:
                current = historical.iloc[-self.skip_period - 1]
            else:
                current = historical.iloc[-1]
            
            past = historical.iloc[-self.lookback_period]
            momentum = current / past - 1
            
            return momentum.replace([np.inf, -np.inf], np.nan).dropna()
        except:
            return pd.Series(dtype=float)


class DualMomentumStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='monthly', lookback_period=252, 
                 top_percentile=0.20, absolute_threshold=0.0):
        super().__init__(name="Dual Momentum Strategy", rebalance_frequency=rebalance_frequency, max_position_weight=0.10)
        self.lookback_period = lookback_period
        self.top_percentile = top_percentile
        self.absolute_threshold = absolute_threshold
    
    def calculate_weights(self, universe, date, data):
        prices = data.get('prices')
        if prices is None:
            return pd.Series(dtype=float)
        
        if isinstance(prices.columns, pd.MultiIndex):
            close_prices = prices.xs('Close', axis=1, level=1)
        else:
            close_prices = prices
        
        try:
            historical = close_prices.loc[:date]
            if len(historical) < self.lookback_period:
                return pd.Series(dtype=float)
            
            current = historical.iloc[-1]
            past = historical.iloc[-self.lookback_period]
            momentum = current / past - 1
            
            available = [t for t in universe if t in momentum.index]
            momentum = momentum[available].dropna()
            
            positive_mom = momentum[momentum > self.absolute_threshold]
            if positive_mom.empty:
                return pd.Series(dtype=float)
            
            n_select = max(1, int(len(positive_mom) * self.top_percentile))
            top_momentum = positive_mom.nlargest(n_select)
            
            weights = pd.Series(1.0 / len(top_momentum), index=top_momentum.index)
            return self.apply_constraints(weights)
            
        except:
            return pd.Series(dtype=float)
