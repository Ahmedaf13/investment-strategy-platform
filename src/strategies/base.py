import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from loguru import logger


class BaseStrategy(ABC):
    
    def __init__(self, name, rebalance_frequency='monthly', max_position_weight=0.10,
                 min_position_weight=0.001, max_positions=None):
        self.name = name
        self.rebalance_frequency = rebalance_frequency
        self.max_position_weight = max_position_weight
        self.min_position_weight = min_position_weight
        self.max_positions = max_positions
        logger.info(f"Strategy '{name}' initialized: rebal={rebalance_frequency}, max_pos={max_position_weight:.1%}")
    
    @abstractmethod
    def calculate_weights(self, universe, date, data):
        pass
    
    def apply_constraints(self, weights):
        if weights.empty:
            return weights
        
        weights = weights.clip(lower=0, upper=self.max_position_weight)
        
        total = weights.sum()
        if total > 0:
            weights = weights / total
        
        weights = weights[weights >= self.min_position_weight]
        
        if self.max_positions and len(weights) > self.max_positions:
            weights = weights.nlargest(self.max_positions)
        
        total = weights.sum()
        if total > 0:
            weights = weights / total
        
        return weights
    
    def get_rebalance_dates(self, start_date, end_date):
        freq_map = {'daily': 'D', 'weekly': 'W-FRI', 'monthly': 'ME', 'quarterly': 'QE'}
        freq = freq_map.get(self.rebalance_frequency, 'ME')
        return pd.date_range(start_date, end_date, freq=freq)
    
    def calculate_turnover(self, old_weights, new_weights):
        all_tickers = old_weights.index.union(new_weights.index)
        old_aligned = old_weights.reindex(all_tickers, fill_value=0)
        new_aligned = new_weights.reindex(all_tickers, fill_value=0)
        return (new_aligned - old_aligned).abs().sum() / 2
