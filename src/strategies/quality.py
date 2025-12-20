import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from .base import BaseStrategy


class QualityStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='quarterly', top_percentile=0.20,
                 quality_metrics=None, metric_weights=None, weighting='equal'):
        super().__init__(name="Quality Strategy", rebalance_frequency=rebalance_frequency, max_position_weight=0.10)
        
        self.top_percentile = top_percentile
        self.weighting = weighting
        self.quality_metrics = quality_metrics or ['roe', 'profit_margin', 'operating_margin', 'debt_to_equity', 'current_ratio']
        self.metric_weights = metric_weights or {
            'roe': 0.30, 'profit_margin': 0.25, 'operating_margin': 0.20,
            'debt_to_equity': -0.15, 'current_ratio': 0.10
        }
    
    def calculate_weights(self, universe, date, data):
        fundamentals = data.get('fundamentals')
        if fundamentals is None:
            return pd.Series(dtype=float)
        
        available = [t for t in universe if t in fundamentals.index]
        if not available:
            return pd.Series(dtype=float)
        
        quality_scores = self._calculate_quality_score(fundamentals.loc[available])
        if quality_scores.empty:
            return pd.Series(dtype=float)
        
        n_select = max(1, int(len(quality_scores) * self.top_percentile))
        top_quality = quality_scores.nlargest(n_select)
        
        if self.weighting == 'equal':
            weights = pd.Series(1.0 / len(top_quality), index=top_quality.index)
        else:
            shifted = top_quality - top_quality.min() + 1
            weights = shifted / shifted.sum()
        
        return self.apply_constraints(weights)
    
    def _calculate_quality_score(self, fundamentals):
        scores = pd.DataFrame(index=fundamentals.index)
        
        for metric in self.quality_metrics:
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


class QualityAtReasonablePrice(BaseStrategy):
    
    def __init__(self, rebalance_frequency='quarterly', top_percentile=0.20,
                 quality_weight=0.6, value_weight=0.4):
        super().__init__(name="Quality at Reasonable Price", rebalance_frequency=rebalance_frequency, max_position_weight=0.10)
        self.top_percentile = top_percentile
        self.quality_weight = quality_weight
        self.value_weight = value_weight
    
    def calculate_weights(self, universe, date, data):
        fundamentals = data.get('fundamentals')
        if fundamentals is None:
            return pd.Series(dtype=float)
        
        available = [t for t in universe if t in fundamentals.index]
        if not available:
            return pd.Series(dtype=float)
        
        fund_data = fundamentals.loc[available]
        
        quality = self._calc_quality(fund_data)
        value = self._calc_value(fund_data)
        
        common = quality.index.intersection(value.index)
        combined = quality.loc[common] * self.quality_weight + value.loc[common] * self.value_weight
        
        if combined.empty:
            return pd.Series(dtype=float)
        
        n_select = max(1, int(len(combined) * self.top_percentile))
        top_stocks = combined.nlargest(n_select)
        
        weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
        return self.apply_constraints(weights)
    
    def _calc_quality(self, fundamentals):
        scores = pd.DataFrame(index=fundamentals.index)
        for m in ['roe', 'profit_margin', 'operating_margin']:
            if m in fundamentals.columns:
                v = fundamentals[m].replace([np.inf, -np.inf], np.nan)
                scores[m] = (v - v.mean()) / (v.std() + 1e-10)
        return scores.mean(axis=1).dropna()
    
    def _calc_value(self, fundamentals):
        scores = pd.DataFrame(index=fundamentals.index)
        for m in ['pe_ratio', 'pb_ratio']:
            if m in fundamentals.columns:
                v = fundamentals[m].replace([np.inf, -np.inf], np.nan)
                scores[m] = -(v - v.mean()) / (v.std() + 1e-10)
        return scores.mean(axis=1).dropna()
