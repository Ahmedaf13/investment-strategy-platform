import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from loguru import logger

from .portfolio import Portfolio
from .costs import TransactionCostModel
from ..strategies.base import BaseStrategy
from ..data.universe import UniverseManager


class BacktestEngine:
    
    def __init__(self, initial_capital=1_000_000, cost_model=None, universe_manager=None):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.universe_manager = universe_manager or UniverseManager()
        self.results = {}
        logger.info(f"BacktestEngine initialized with ${initial_capital:,.2f}")
    
    def run(self, strategy, data, start_date=None, end_date=None, show_progress=True):
        logger.info(f"Running backtest for {strategy.name}")
        
        prices = data.get('prices')
        if prices is None:
            raise ValueError("Price data is required")
        
        if isinstance(prices.columns, pd.MultiIndex):
            close_prices = prices.xs('Close', axis=1, level=1)
        else:
            close_prices = prices
        
        if start_date is None:
            start_date = close_prices.index[0]
        if end_date is None:
            end_date = close_prices.index[-1]
        
        if close_prices.index.tz is not None:
            close_prices = close_prices.copy()
            close_prices.index = close_prices.index.tz_localize(None)
        
        if hasattr(start_date, 'tz') and start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if hasattr(end_date, 'tz') and end_date.tz is not None:
            end_date = end_date.tz_localize(None)
        
        mask = (close_prices.index >= start_date) & (close_prices.index <= end_date)
        trading_dates = close_prices.index[mask]
        
        rebalance_dates = strategy.get_rebalance_dates(start_date, end_date)
        rebalance_dates = self.universe_manager.get_tradeable_dates(close_prices, rebalance_dates)
        
        portfolio = Portfolio(self.initial_capital)
        tickers = data.get('tickers', close_prices.columns.tolist())
        
        iterator = tqdm(trading_dates, desc=strategy.name) if show_progress else trading_dates
        
        for date in iterator:
            current_prices = close_prices.loc[date].to_dict()
            
            if date in rebalance_dates:
                if 'market_caps' in data:
                    universe = self.universe_manager.filter_universe(tickers, prices, data['market_caps'], date)
                else:
                    universe = [t for t in tickers if t in current_prices and 
                               pd.notna(current_prices[t]) and current_prices[t] > 0]
                
                if universe:
                    target_weights = strategy.calculate_weights(universe, date, data)
                    if not target_weights.empty:
                        portfolio.execute_trades(target_weights, current_prices, date, self.cost_model)
            
            portfolio.record_state(date, current_prices)
        
        results = self._compile_results(strategy, portfolio, data)
        self.results[strategy.name] = results
        
        logger.info(f"Backtest complete: Return={results['metrics']['total_return']:.2%}, Sharpe={results['metrics']['sharpe_ratio']:.2f}")
        return results
    
    def run_multiple(self, strategies, data, start_date=None, end_date=None, show_progress=True):
        logger.info(f"Running backtests for {len(strategies)} strategies")
        
        all_results = {}
        for strategy in strategies:
            try:
                results = self.run(strategy, data, start_date, end_date, show_progress)
                all_results[strategy.name] = results
            except Exception as e:
                logger.error(f"Error running {strategy.name}: {e}")
                all_results[strategy.name] = {'error': str(e)}
        
        return all_results
    
    def _compile_results(self, strategy, portfolio, data):
        values = portfolio.get_value_series()
        returns = portfolio.get_returns()
        
        benchmarks = data.get('benchmarks')
        benchmark_returns = None
        if benchmarks is not None and '^GSPC' in benchmarks.columns:
            benchmark = benchmarks['^GSPC'].reindex(values.index).ffill()
            benchmark_returns = benchmark.pct_change().dropna()
        
        metrics = self._calculate_metrics(returns, benchmark_returns)
        
        return {
            'strategy_name': strategy.name,
            'values': values,
            'returns': returns,
            'cumulative_returns': portfolio.get_cumulative_returns(),
            'metrics': metrics,
            'trade_summary': portfolio.get_trade_summary(),
            'final_value': values.iloc[-1] if len(values) > 0 else 0,
            'weight_history': pd.DataFrame(portfolio.weight_history)
        }
    
    def _calculate_metrics(self, returns, benchmark_returns=None, risk_free_rate=0.02):
        if returns.empty:
            return {}
        
        ann_factor = 252
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ann_factor)
        
        sharpe = (ann_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        downside = returns[returns < 0].std() * np.sqrt(ann_factor)
        sortino = (ann_return - risk_free_rate) / downside if downside > 0 else 0
        
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        
        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def compare_strategies(self):
        if not self.results:
            logger.warning("No backtest results available")
            return pd.DataFrame()
        
        rows = []
        for name, result in self.results.items():
            if 'error' in result:
                continue
            
            m = result.get('metrics', {})
            t = result.get('trade_summary', {})
            
            rows.append({
                'Strategy': name,
                'Total Return': m.get('total_return', 0),
                'Ann. Return': m.get('annualized_return', 0),
                'Volatility': m.get('volatility', 0),
                'Sharpe': m.get('sharpe_ratio', 0),
                'Sortino': m.get('sortino_ratio', 0),
                'Max DD': m.get('max_drawdown', 0),
                'Calmar': m.get('calmar_ratio', 0),
                'VaR 95%': m.get('var_95', 0),
                'Win Rate': m.get('win_rate', 0),
                'Total Trades': t.get('total_trades', 0),
                'Turnover': t.get('total_turnover', 0),
                'Total Costs': t.get('total_costs', 0)
            })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index('Strategy').sort_values('Sharpe', ascending=False)
        return df
