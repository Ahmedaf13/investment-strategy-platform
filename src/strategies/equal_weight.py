import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from .base import BaseStrategy


class EqualWeightedStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='monthly', max_positions=None, selection_method='all'):
        super().__init__(
            name="Equal Weighted",
            rebalance_frequency=rebalance_frequency,
            max_position_weight=1.0,
            min_position_weight=0.0,
            max_positions=max_positions
        )
        self.selection_method = selection_method
    
    def calculate_weights(self, universe, date, data):
        if not universe:
            return pd.Series(dtype=float)
        
        if self.max_positions and len(universe) > self.max_positions:
            selected = self._select_stocks(universe, date, data)
        else:
            selected = universe
        
        if not selected:
            return pd.Series(dtype=float)
        
        weight = 1.0 / len(selected)
        return pd.Series(weight, index=selected)
    
    def _select_stocks(self, universe, date, data):
        if self.selection_method == 'random':
            np.random.seed(int(date.timestamp()) % 2**32)
            return list(np.random.choice(universe, self.max_positions, replace=False))
        
        if self.selection_method == 'top_mcap':
            market_caps = data.get('market_caps')
            if market_caps is not None:
                try:
                    valid_dates = market_caps.index[market_caps.index <= date]
                    if len(valid_dates) > 0:
                        mcaps = market_caps.loc[valid_dates[-1]]
                        available = [t for t in universe if t in mcaps.index]
                        return mcaps[available].dropna().nlargest(self.max_positions).index.tolist()
                except:
                    pass
        
        return universe[:self.max_positions]


class EqualSectorWeightedStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='monthly', min_stocks_per_sector=3):
        super().__init__(name="Equal Sector Weighted", rebalance_frequency=rebalance_frequency, max_position_weight=0.20)
        self.min_stocks_per_sector = min_stocks_per_sector
    
    def calculate_weights(self, universe, date, data):
        fundamentals = data.get('fundamentals')
        
        if fundamentals is None or 'sector' not in fundamentals.columns:
            n = len(universe)
            return pd.Series(1.0/n if n > 0 else 0, index=universe)
        
        available = [t for t in universe if t in fundamentals.index]
        sectors = fundamentals.loc[available, 'sector'].dropna()
        sector_groups = sectors.groupby(sectors).apply(lambda x: x.index.tolist())
        
        valid_sectors = {s: stocks for s, stocks in sector_groups.items() 
                        if len(stocks) >= self.min_stocks_per_sector}
        
        if not valid_sectors:
            return pd.Series(dtype=float)
        
        sector_weight = 1.0 / len(valid_sectors)
        weights = {}
        for sector, stocks in valid_sectors.items():
            stock_weight = sector_weight / len(stocks)
            for stock in stocks:
                weights[stock] = stock_weight
        
        return self.apply_constraints(pd.Series(weights))
