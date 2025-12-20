import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.optimize import minimize
from loguru import logger
from .base import BaseStrategy


class RiskParityStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='monthly', lookback_period=63,
                 target_volatility=None, use_correlation=True):
        super().__init__(name="Risk Parity", rebalance_frequency=rebalance_frequency, max_position_weight=0.25)
        self.lookback_period = lookback_period
        self.target_volatility = target_volatility
        self.use_correlation = use_correlation
    
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
            if len(historical) < self.lookback_period + 10:
                return pd.Series(dtype=float)
            
            returns = historical.iloc[-self.lookback_period:].pct_change().dropna()
            available = [t for t in universe if t in returns.columns]
            returns = returns[available].dropna(axis=1, how='all')
            
            if returns.empty or len(returns.columns) < 2:
                return pd.Series(dtype=float)
            
            cov_matrix = returns.cov() * 252
            
            if self.use_correlation:
                w = self._optimize_risk_parity(cov_matrix)
            else:
                w = self._inverse_volatility(returns)
            
            weights = pd.Series(w, index=returns.columns)
            weights = self.apply_constraints(weights)
            
            if self.target_volatility:
                weights = self._scale_to_target(weights, cov_matrix)
            
            return weights
            
        except Exception as e:
            logger.error(f"Risk parity error: {e}")
            return pd.Series(dtype=float)
    
    def _inverse_volatility(self, returns):
        vol = returns.std() * np.sqrt(252)
        inv_vol = 1 / vol
        return (inv_vol / inv_vol.sum()).values
    
    def _optimize_risk_parity(self, cov_matrix):
        n = len(cov_matrix)
        cov = cov_matrix.values
        vol = np.sqrt(np.diag(cov))
        x0 = (1 / vol) / (1 / vol).sum()
        
        def risk_parity_obj(w):
            port_vol = np.sqrt(w @ cov @ w)
            marginal = cov @ w
            rc = w * marginal / port_vol
            target = 1 / n
            return np.sum((rc - target) ** 2)
        
        bounds = [(0.001, 0.5) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        try:
            result = minimize(risk_parity_obj, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            return result.x if result.success else x0
        except:
            return x0
    
    def _scale_to_target(self, weights, cov_matrix):
        aligned = cov_matrix.loc[weights.index, weights.index]
        port_vol = np.sqrt(weights @ aligned @ weights)
        if port_vol > 0:
            leverage = min(self.target_volatility / port_vol, 2.0)
            scaled = weights * leverage
            return scaled / scaled.sum()
        return weights


class InverseVolatilityStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='monthly', lookback_period=63, max_positions=None):
        super().__init__(name="Inverse Volatility", rebalance_frequency=rebalance_frequency,
                        max_position_weight=0.20, max_positions=max_positions)
        self.lookback_period = lookback_period
    
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
            returns = historical.iloc[-self.lookback_period:].pct_change().dropna()
            
            available = [t for t in universe if t in returns.columns]
            returns = returns[available].dropna(axis=1, how='all')
            
            if returns.empty:
                return pd.Series(dtype=float)
            
            volatility = returns.std() * np.sqrt(252)
            volatility = volatility[volatility > 0.01]
            
            if volatility.empty:
                return pd.Series(dtype=float)
            
            inv_vol = 1 / volatility
            weights = inv_vol / inv_vol.sum()
            return self.apply_constraints(weights)
            
        except:
            return pd.Series(dtype=float)


class MinimumVarianceStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='monthly', lookback_period=126, shrinkage=0.5):
        super().__init__(name="Minimum Variance", rebalance_frequency=rebalance_frequency, max_position_weight=0.15)
        self.lookback_period = lookback_period
        self.shrinkage = shrinkage
    
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
            returns = historical.iloc[-self.lookback_period:].pct_change().dropna()
            
            available = [t for t in universe if t in returns.columns]
            returns = returns[available].dropna(axis=1, how='all')
            
            if returns.empty or len(returns.columns) < 2:
                return pd.Series(dtype=float)
            
            cov = returns.cov()
            n = len(cov)
            avg_var = np.diag(cov).mean()
            target = np.eye(n) * avg_var
            shrunk = (1 - self.shrinkage) * cov.values + self.shrinkage * target
            
            x0 = np.ones(n) / n
            bounds = [(0, 0.15) for _ in range(n)]
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            result = minimize(lambda w: w @ shrunk @ w, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            w = result.x if result.success else x0
            
            weights = pd.Series(w, index=returns.columns)
            return self.apply_constraints(weights)
            
        except:
            return pd.Series(dtype=float)
