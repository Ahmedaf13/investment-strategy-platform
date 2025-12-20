import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger


class DataCleaner:
    
    def __init__(self, min_history_days=252, max_missing_pct=0.10, outlier_std=5.0):
        self.min_history_days = min_history_days
        self.max_missing_pct = max_missing_pct
        self.outlier_std = outlier_std
        logger.info(f"DataCleaner initialized: min_history={min_history_days}, max_missing={max_missing_pct:.1%}")
    
    def clean_price_data(self, prices):
        logger.info(f"Cleaning price data: {prices.shape}")
        
        if prices.index.tz is not None:
            prices = prices.copy()
            prices.index = prices.index.tz_localize(None)
        
        tickers = prices.columns.get_level_values(0).unique()
        cleaned_tickers = []
        removed = []
        
        for ticker in tickers:
            try:
                ticker_data = prices[ticker]
                valid_days = ticker_data['Close'].notna().sum()
                
                if valid_days < self.min_history_days:
                    removed.append(ticker)
                    continue
                
                missing_pct = ticker_data['Close'].isna().mean()
                if missing_pct > self.max_missing_pct:
                    removed.append(ticker)
                    continue
                
                cleaned_tickers.append(ticker)
            except:
                removed.append(ticker)
        
        logger.info(f"Kept {len(cleaned_tickers)} tickers, removed {len(removed)}")
        
        cleaned = prices[[t for t in cleaned_tickers if t in prices.columns.get_level_values(0)]]
        cleaned = cleaned.ffill(limit=5).bfill(limit=5)
        return cleaned
    
    def create_returns(self, prices, method='simple'):
        if isinstance(prices.columns, pd.MultiIndex):
            close_prices = prices.xs('Close', axis=1, level=1)
        else:
            close_prices = prices
        
        if method == 'simple':
            returns = close_prices.pct_change()
        else:
            returns = np.log(close_prices / close_prices.shift(1))
        
        return returns.iloc[1:]
    
    def get_close_prices(self, prices):
        if isinstance(prices.columns, pd.MultiIndex):
            return prices.xs('Close', axis=1, level=1)
        return prices
