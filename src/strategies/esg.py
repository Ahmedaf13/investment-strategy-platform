import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from .base import BaseStrategy


class ESGStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='quarterly', min_esg_score=5.0,
                 weighting='market_cap', exclude_sectors=None, tilt_by_esg=False):
        super().__init__(name="ESG Integration", rebalance_frequency=rebalance_frequency, max_position_weight=0.10)
        self.min_esg_score = min_esg_score
        self.weighting = weighting
        self.exclude_sectors = exclude_sectors or []
        self.tilt_by_esg = tilt_by_esg
    
    def calculate_weights(self, universe, date, data):
        fundamentals = data.get('fundamentals')
        esg_scores = data.get('esg_scores')
        market_caps = data.get('market_caps')
        
        if fundamentals is not None and 'sector' in fundamentals.columns:
            filtered = [t for t in universe 
                       if t in fundamentals.index and 
                       fundamentals.loc[t, 'sector'] not in self.exclude_sectors]
        else:
            filtered = universe
        
        if esg_scores is not None:
            filtered = self._filter_by_esg(filtered, esg_scores)
        else:
            logger.warning("No ESG data available, using sector-filtered universe")
        
        if not filtered:
            return pd.Series(dtype=float)
        
        if self.weighting == 'market_cap':
            weights = self._market_cap_weights(filtered, market_caps, date)
        elif self.weighting == 'esg_score' and esg_scores is not None:
            weights = self._esg_score_weights(filtered, esg_scores)
        else:
            weights = pd.Series(1.0 / len(filtered), index=filtered)
        
        if self.tilt_by_esg and esg_scores is not None:
            weights = self._apply_esg_tilt(weights, esg_scores)
        
        return self.apply_constraints(weights)
    
    def _filter_by_esg(self, universe, esg_scores):
        score_col = None
        for col in ['composite_score', 'esg_score']:
            if col in esg_scores.columns:
                score_col = col
                break
        
        if not score_col:
            return universe
        
        return [t for t in universe 
                if t in esg_scores.index and 
                pd.notna(esg_scores.loc[t, score_col]) and 
                esg_scores.loc[t, score_col] >= self.min_esg_score]
    
    def _market_cap_weights(self, tickers, market_caps, date):
        if market_caps is None:
            return pd.Series(1.0 / len(tickers), index=tickers)
        
        try:
            valid_dates = market_caps.index[market_caps.index <= date]
            if len(valid_dates) == 0:
                return pd.Series(1.0 / len(tickers), index=tickers)
            
            mcaps = market_caps.loc[valid_dates[-1]]
            available = [t for t in tickers if t in mcaps.index]
            mcaps = mcaps[available].dropna()
            
            if mcaps.empty:
                return pd.Series(1.0 / len(tickers), index=tickers)
            
            return mcaps / mcaps.sum()
        except:
            return pd.Series(1.0 / len(tickers), index=tickers)
    
    def _esg_score_weights(self, tickers, esg_scores):
        score_col = 'composite_score' if 'composite_score' in esg_scores.columns else 'esg_score'
        if score_col not in esg_scores.columns:
            return pd.Series(1.0 / len(tickers), index=tickers)
        
        scores = esg_scores.loc[tickers, score_col].dropna()
        if scores.empty:
            return pd.Series(1.0 / len(tickers), index=tickers)
        
        return scores / scores.sum()
    
    def _apply_esg_tilt(self, weights, esg_scores):
        score_col = 'composite_score' if 'composite_score' in esg_scores.columns else 'esg_score'
        if score_col not in esg_scores.columns:
            return weights
        
        available = [t for t in weights.index if t in esg_scores.index]
        if not available:
            return weights
        
        scores = esg_scores.loc[available, score_col]
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        tilted = weights.loc[available] * (1 + norm_scores * 0.5)
        return tilted / tilted.sum()


class NegativeScreeningStrategy(BaseStrategy):
    
    def __init__(self, rebalance_frequency='quarterly', exclude_sectors=None, base_weighting='market_cap'):
        super().__init__(name="ESG Negative Screening", rebalance_frequency=rebalance_frequency, max_position_weight=0.10)
        self.exclude_sectors = exclude_sectors or ['Tobacco', 'Gambling', 'Weapons']
        self.base_weighting = base_weighting
    
    def calculate_weights(self, universe, date, data):
        fundamentals = data.get('fundamentals')
        market_caps = data.get('market_caps')
        
        if fundamentals is not None and 'sector' in fundamentals.columns:
            screened = [t for t in universe
                       if t not in fundamentals.index or 
                       fundamentals.loc[t, 'sector'] not in self.exclude_sectors]
        else:
            screened = universe
        
        if not screened:
            return pd.Series(dtype=float)
        
        if self.base_weighting == 'market_cap' and market_caps is not None:
            try:
                valid_dates = market_caps.index[market_caps.index <= date]
                if len(valid_dates) > 0:
                    mcaps = market_caps.loc[valid_dates[-1]]
                    available = [t for t in screened if t in mcaps.index]
                    mcaps = mcaps[available].dropna()
                    weights = mcaps / mcaps.sum()
                else:
                    weights = pd.Series(1.0 / len(screened), index=screened)
            except:
                weights = pd.Series(1.0 / len(screened), index=screened)
        else:
            weights = pd.Series(1.0 / len(screened), index=screened)
        
        return self.apply_constraints(weights)
