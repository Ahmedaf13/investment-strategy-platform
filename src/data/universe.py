import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from loguru import logger


class UniverseManager:
    
    def __init__(self, min_market_cap=1e9, min_price=5.0, min_volume=1e6):
        self.min_market_cap = min_market_cap
        self.min_price = min_price
        self.min_volume = min_volume
        logger.info(f"UniverseManager initialized: min_mcap=${min_market_cap/1e9:.1f}B, min_price=${min_price}, min_vol={min_volume/1e6:.1f}M")
    
    def filter_universe(self, tickers, prices, market_caps, date):
        if hasattr(date, 'tz') and date.tz is not None:
            date = date.tz_localize(None)
        
        if prices.index.tz is not None:
            prices = prices.copy()
            prices.index = prices.index.tz_localize(None)
        
        if market_caps.index.tz is not None:
            market_caps = market_caps.copy()
            market_caps.index = market_caps.index.tz_localize(None)
        
        filtered = []
        for ticker in tickers:
            try:
                if ticker in market_caps.columns:
                    mcap = market_caps.loc[:date, ticker].iloc[-1]
                    if pd.isna(mcap) or mcap < self.min_market_cap:
                        continue
                
                if isinstance(prices.columns, pd.MultiIndex):
                    if ticker in prices.columns.get_level_values(0):
                        price = prices.loc[:date, (ticker, 'Close')].iloc[-1]
                        volume = prices.loc[:date, (ticker, 'Volume')].iloc[-20:].mean()
                    else:
                        continue
                else:
                    if ticker in prices.columns:
                        price = prices.loc[:date, ticker].iloc[-1]
                        volume = 1e7
                    else:
                        continue
                
                if pd.isna(price) or price < self.min_price:
                    continue
                if volume < self.min_volume:
                    continue
                
                filtered.append(ticker)
            except:
                continue
        
        logger.info(f"Filtered universe: {len(filtered)}/{len(tickers)} tickers on {date.date()}")
        return filtered
    
    def generate_rebalance_dates(self, start_date, end_date, frequency='monthly', calendar_type='end_of_period'):
        freq_map = {
            'daily': 'D', 'weekly': 'W-FRI', 'monthly': 'ME', 'quarterly': 'QE'
        }
        freq = freq_map.get(frequency, 'ME')
        
        if calendar_type == 'start_of_period':
            if frequency == 'monthly':
                freq = 'MS'
            elif frequency == 'quarterly':
                freq = 'QS'
        
        dates = pd.date_range(start_date, end_date, freq=freq)
        logger.info(f"Generated {len(dates)} rebalance dates ({frequency})")
        return dates
    
    def get_tradeable_dates(self, prices, rebalance_dates):
        trading_days = prices.index
        tradeable = []
        
        for date in rebalance_dates:
            valid_days = trading_days[trading_days >= date]
            if len(valid_days) > 0:
                tradeable.append(valid_days[0])
        
        result = pd.DatetimeIndex(tradeable).unique()
        logger.info(f"Aligned to {len(result)} tradeable dates")
        return result
