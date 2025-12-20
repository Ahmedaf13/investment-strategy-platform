import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class FeatureEngineer:
    
    def __init__(self, trading_days_per_year=252):
        self.trading_days = trading_days_per_year
    
    def calculate_momentum(self, prices, lookback=252, skip=21):
        if skip > 0:
            shifted = prices.shift(skip)
            return shifted / shifted.shift(lookback - skip) - 1
        return prices / prices.shift(lookback) - 1
    
    def calculate_momentum_6m(self, prices):
        return self.calculate_momentum(prices, lookback=126, skip=21)
    
    def calculate_momentum_12m(self, prices):
        return self.calculate_momentum(prices, lookback=252, skip=21)
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_volatility(self, returns, window=63, annualize=True):
        vol = returns.rolling(window=window, min_periods=window//2).std()
        if annualize:
            vol = vol * np.sqrt(self.trading_days)
        return vol
    
    def calculate_beta(self, returns, market_returns, window=252):
        betas = pd.DataFrame(index=returns.index, columns=returns.columns)
        for col in returns.columns:
            cov = returns[col].rolling(window).cov(market_returns)
            var = market_returns.rolling(window).var()
            betas[col] = cov / var
        return betas
    
    def calculate_drawdown(self, prices):
        rolling_max = prices.expanding().max()
        return prices / rolling_max - 1
    
    def calculate_value_composite(self, fundamentals, weights=None):
        if weights is None:
            weights = {'pe_ratio': -0.3, 'pb_ratio': -0.3, 'dividend_yield': 0.4}
        
        standardized = pd.DataFrame(index=fundamentals.index)
        for factor, weight in weights.items():
            if factor in fundamentals.columns:
                values = fundamentals[factor].replace([np.inf, -np.inf], np.nan)
                z = (values - values.mean()) / values.std()
                standardized[factor] = z * weight
        
        return standardized.sum(axis=1)
    
    def calculate_quality_composite(self, fundamentals, weights=None):
        if weights is None:
            weights = {'roe': 0.4, 'profit_margin': 0.3, 'debt_to_equity': -0.3}
        
        standardized = pd.DataFrame(index=fundamentals.index)
        for factor, weight in weights.items():
            if factor in fundamentals.columns:
                values = fundamentals[factor].replace([np.inf, -np.inf], np.nan)
                z = (values - values.mean()) / values.std()
                standardized[factor] = z * weight
        
        return standardized.sum(axis=1)
