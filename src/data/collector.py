import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import pickle
from tqdm import tqdm
from loguru import logger
import urllib.request


class DataCollector:
    
    def __init__(self, start_date="2014-01-01", end_date="2024-12-31", cache_dir="data/cache"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataCollector initialized: {start_date} to {end_date}")
    
    def get_sp500_tickers(self, use_cache=True):
        cache_file = self.cache_dir / "sp500_tickers.pkl"
        
        if use_cache and cache_file.exists():
            with open(cache_file, 'rb') as f:
                tickers = pickle.load(f)
            logger.info(f"Loaded {len(tickers)} S&P 500 tickers from cache")
            return tickers
        
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read()
            
            tables = pd.read_html(html)
            tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
            
            with open(cache_file, 'wb') as f:
                pickle.dump(tickers, f)
            
            logger.info(f"Downloaded {len(tickers)} S&P 500 tickers")
            return tickers
            
        except Exception as e:
            logger.warning(f"Could not fetch S&P 500 from Wikipedia: {e}")
            return self._get_fallback_tickers()
    
    def _get_fallback_tickers(self):
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 
            'BRK-B', 'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA',
            'ABBV', 'MRK', 'PFE', 'COST', 'AVGO', 'KO', 'PEP', 'TMO', 'WMT',
            'CSCO', 'MCD', 'ACN', 'ABT', 'DHR', 'VZ', 'CMCSA', 'LIN', 'NEE',
            'ADBE', 'TXN', 'PM', 'WFC', 'BMY', 'NKE', 'INTC', 'RTX', 'QCOM',
            'UPS', 'COP', 'HON', 'LOW', 'SPGI', 'ORCL', 'AMD', 'IBM', 'GE',
            'CAT', 'INTU', 'AMAT', 'BA', 'SBUX', 'ISRG', 'GS', 'BLK', 'NOW',
            'AXP', 'BKNG', 'MDLZ', 'ADI', 'GILD', 'DE', 'LMT', 'ADP', 'TJX',
            'SYK', 'VRTX', 'MMC', 'REGN', 'CI', 'CB', 'CVS', 'LRCX', 'ETN',
            'MO', 'SCHW', 'AMT', 'ZTS', 'PGR', 'SO', 'BDX', 'C', 'DUK', 'BSX',
            'CME', 'TMUS', 'EQIX', 'FI', 'ITW', 'SLB', 'EOG', 'AON', 'SNPS',
            'PNC', 'MU', 'CL', 'KLAC', 'NOC', 'USB', 'ICE', 'WM', 'CDNS'
        ]
    
    def download_price_data(self, tickers, use_cache=True, show_progress=True):
        cache_file = self.cache_dir / f"prices_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.pkl"
        
        if use_cache and cache_file.exists():
            df = pd.read_pickle(cache_file)
            logger.info(f"Loaded price data from cache: {df.shape}")
            return df
        
        logger.info(f"Downloading price data for {len(tickers)} tickers...")
        
        all_data = {}
        failed = []
        
        iterator = tqdm(tickers, desc="Downloading") if show_progress else tickers
        
        for ticker in iterator:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date, auto_adjust=True)
                if len(hist) > 0:
                    all_data[ticker] = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
                else:
                    failed.append(ticker)
            except Exception as e:
                failed.append(ticker)
        
        if failed:
            logger.warning(f"Failed to download {len(failed)} tickers: {failed[:10]}...")
        
        if all_data:
            df = pd.concat(all_data, axis=1)
            df.columns = pd.MultiIndex.from_tuples([
                (ticker, field) for ticker, sub_df in all_data.items() 
                for field in sub_df.columns
            ])
            
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            df.to_pickle(cache_file)
            logger.info(f"Downloaded and cached price data: {df.shape}")
            return df
        
        return pd.DataFrame()
    
    def download_fundamental_data(self, tickers, use_cache=True):
        cache_file = self.cache_dir / "fundamentals.pkl"
        
        if use_cache and cache_file.exists():
            df = pd.read_pickle(cache_file)
            logger.info(f"Loaded fundamental data from cache: {df.shape}")
            return df
        
        logger.info(f"Downloading fundamental data for {len(tickers)} tickers...")
        
        fundamentals = []
        for ticker in tqdm(tickers, desc="Downloading fundamentals"):
            try:
                info = yf.Ticker(ticker).info
                fundamentals.append({
                    'ticker': ticker,
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'ps_ratio': info.get('priceToSalesTrailing12Months'),
                    'dividend_yield': info.get('dividendYield'),
                    'payout_ratio': info.get('payoutRatio'),
                    'roe': info.get('returnOnEquity'),
                    'roa': info.get('returnOnAssets'),
                    'profit_margin': info.get('profitMargins'),
                    'operating_margin': info.get('operatingMargins'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio'),
                    'quick_ratio': info.get('quickRatio'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth'),
                    'beta': info.get('beta'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry')
                })
            except:
                pass
        
        df = pd.DataFrame(fundamentals).set_index('ticker')
        df.to_pickle(cache_file)
        logger.info(f"Downloaded and cached fundamental data: {df.shape}")
        return df
    
    def download_market_cap_history(self, tickers, prices, use_cache=True):
        cache_file = self.cache_dir / "market_caps.pkl"
        
        if use_cache and cache_file.exists():
            df = pd.read_pickle(cache_file)
            logger.info(f"Loaded market cap data from cache: {df.shape}")
            return df
        
        logger.info("Calculating historical market caps...")
        
        market_caps = {}
        for ticker in tqdm(tickers, desc="Getting shares outstanding"):
            try:
                shares = yf.Ticker(ticker).info.get('sharesOutstanding')
                if shares and ticker in prices.columns.get_level_values(0):
                    market_caps[ticker] = prices[ticker]['Close'] * shares
            except:
                pass
        
        df = pd.DataFrame(market_caps)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df.to_pickle(cache_file)
        logger.info(f"Calculated and cached market cap data: {df.shape}")
        return df
    
    def download_benchmark_data(self, benchmarks=['^GSPC', '^W5000', '^IXIC'], use_cache=True):
        cache_file = self.cache_dir / "benchmarks.pkl"
        
        if use_cache and cache_file.exists():
            df = pd.read_pickle(cache_file)
            logger.info(f"Loaded benchmark data from cache: {df.shape}")
            return df
        
        logger.info(f"Downloading benchmark data: {benchmarks}")
        
        all_data = {}
        for benchmark in benchmarks:
            try:
                data = yf.Ticker(benchmark).history(
                    start=self.start_date, end=self.end_date, auto_adjust=True
                )
                if data is not None and not data.empty and 'Close' in data.columns:
                    close_data = data['Close']
                    if isinstance(close_data, pd.Series) and len(close_data) > 0:
                        all_data[benchmark] = close_data
                        logger.info(f"Downloaded {benchmark}: {len(close_data)} days")
            except Exception as e:
                logger.warning(f"Failed to download {benchmark}: {e}")
        
        if not all_data:
            logger.warning("Trying SPY as S&P 500 proxy...")
            try:
                data = yf.Ticker('SPY').history(
                    start=self.start_date, end=self.end_date, auto_adjust=True
                )
                if data is not None and not data.empty:
                    all_data['^GSPC'] = data['Close']
            except:
                pass
        
        if not all_data:
            logger.warning("No benchmark data available")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df.to_pickle(cache_file)
        logger.info(f"Downloaded and cached benchmark data: {df.shape}")
        return df
