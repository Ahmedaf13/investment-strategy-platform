import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from .base import BaseStrategy


class ValueStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='quarterly', top_percentile=0.20, 
                 value_metrics=None, metric_weights=None, weighting='equal'):
        super().__init__(name="Value Strategy", rebalance_frequency=rebalance_frequency, max_position_weight=0.10)
        
        self.top_percentile = top_percentile
        self.weighting = weighting
        self.value_metrics = value_metrics or ['pe_ratio', 'pb_ratio', 'dividend_yield']
        self.metric_weights = metric_weights or {
            'pe_ratio': -0.35, 'pb_ratio': -0.35, 'dividend_yield': 0.30
        }
    
    def calculate_weights(self, universe, date, data):
        fundamentals = data.get('fundamentals')
        if fundamentals is None:
            return pd.Series(dtype=float)
        
        available = [t for t in universe if t in fundamentals.index]
        if not available:
            return pd.Series(dtype=float)
        
        value_scores = self._calculate_value_score(fundamentals.loc[available])
        if value_scores.empty:
            return pd.Series(dtype=float)
        
        n_select = max(1, int(len(value_scores) * self.top_percentile))
        top_value = value_scores.nlargest(n_select)
        
        if self.weighting == 'equal':
            weights = pd.Series(1.0 / len(top_value), index=top_value.index)
        else:
            shifted = top_value - top_value.min() + 1
            weights = shifted / shifted.sum()
        
        return self.apply_constraints(weights)
    
    def _calculate_value_score(self, fundamentals):
        scores = pd.DataFrame(index=fundamentals.index)
        
        for metric in self.value_metrics:
            if metric not in fundamentals.columns:
                continue
            
            values = fundamentals[metric].replace([np.inf, -np.inf], np.nan)
            lower, upper = values.quantile(0.01), values.quantile(0.99)
            values = values.clip(lower=lower, upper=upper)
            
            std = values.std()
            if std > 0:
                z_scores = (values - values.mean()) / std
            else:
                z_scores = values * 0
            
            scores[metric] = z_scores * self.metric_weights.get(metric, 1.0)
        
        return scores.sum(axis=1).dropna()


class DeepValueStrategy(ValueStrategy):
    
    def __init__(self, rebalance_frequency='quarterly', top_percentile=0.10,
                 min_dividend_yield=0.02, max_debt_equity=1.5):
        super().__init__(rebalance_frequency=rebalance_frequency, top_percentile=top_percentile)
        self.name = "Deep Value Strategy"
        self.min_dividend_yield = min_dividend_yield
        self.max_debt_equity = max_debt_equity
    
    def calculate_weights(self, universe, date, data):
        fundamentals = data.get('fundamentals')
        if fundamentals is None:
            return pd.Series(dtype=float)
        
        filtered = []
        for ticker in universe:
            if ticker not in fundamentals.index:
                continue
            row = fundamentals.loc[ticker]
            
            div_yield = row.get('dividend_yield', 0)
            if pd.notna(div_yield) and div_yield < self.min_dividend_yield:
                continue
            
            debt_eq = row.get('debt_to_equity', 0)
            if pd.notna(debt_eq) and debt_eq > self.max_debt_equity:
                continue
            
            filtered.append(ticker)
        
        if not filtered:
            return pd.Series(dtype=float)
        
        return super().calculate_weights(filtered, date, data)
