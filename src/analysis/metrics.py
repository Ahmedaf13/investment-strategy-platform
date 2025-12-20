import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from loguru import logger


class PerformanceMetrics:
    
    def __init__(self, risk_free_rate=0.02, trading_days=252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
    
    def calculate_all(self, returns, benchmark_returns=None):
        metrics = {}
        metrics.update(self.return_metrics(returns))
        metrics.update(self.risk_metrics(returns))
        metrics.update(self.drawdown_metrics(returns))
        metrics.update(self.distribution_metrics(returns))
        metrics.update(self.risk_adjusted_metrics(returns))
        
        if benchmark_returns is not None and not benchmark_returns.empty:
            try:
                metrics.update(self.benchmark_metrics(returns, benchmark_returns))
            except Exception as e:
                logger.warning(f"Could not calculate benchmark metrics: {e}")
        
        return metrics
    
    def return_metrics(self, returns):
        if returns.empty:
            return {}
        
        cum = (1 + returns).cumprod()
        total = cum.iloc[-1] - 1
        n_years = len(returns) / self.trading_days
        ann = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) if hasattr(returns.index, 'to_period') else pd.Series()
        yearly = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1) if hasattr(returns.index, 'to_period') else pd.Series()
        
        return {
            'total_return': total,
            'annualized_return': ann,
            'cagr': ann,
            'monthly_avg_return': monthly.mean() if len(monthly) > 0 else np.nan,
            'best_month': monthly.max() if len(monthly) > 0 else np.nan,
            'worst_month': monthly.min() if len(monthly) > 0 else np.nan,
            'best_year': yearly.max() if len(yearly) > 0 else np.nan,
            'worst_year': yearly.min() if len(yearly) > 0 else np.nan
        }
    
    def risk_metrics(self, returns):
        if returns.empty:
            return {}
        
        daily_vol = returns.std()
        ann_vol = daily_vol * np.sqrt(self.trading_days)
        downside = returns[returns < 0].std() * np.sqrt(self.trading_days)
        
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        return {
            'daily_volatility': daily_vol,
            'annual_volatility': ann_vol,
            'downside_deviation': downside,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }
    
    def drawdown_metrics(self, returns):
        if returns.empty:
            return {}
        
        cum = (1 + returns).cumprod()
        rolling_max = cum.expanding().max()
        dd = cum / rolling_max - 1
        
        max_dd = dd.min()
        max_dd_end = dd.idxmin()
        max_dd_start = cum[:max_dd_end].idxmax()
        
        in_dd = dd < 0
        ulcer = np.sqrt((dd ** 2).mean())
        
        return {
            'max_drawdown': max_dd,
            'max_dd_start': max_dd_start,
            'max_dd_end': max_dd_end,
            'avg_drawdown': dd.mean(),
            'ulcer_index': ulcer,
            'time_in_drawdown': in_dd.mean()
        }
    
    def distribution_metrics(self, returns):
        if returns.empty:
            return {}
        
        wins = returns > 0
        losses = returns < 0
        
        win_rate = wins.mean()
        avg_win = returns[wins].mean() if wins.any() else 0
        avg_loss = returns[losses].mean() if losses.any() else 0
        
        total_wins = returns[wins].sum()
        total_losses = abs(returns[losses].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        return {
            'mean_return': returns.mean(),
            'median_return': returns.median(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': returns.max(),
            'max_loss': returns.min(),
            'profit_factor': profit_factor
        }
    
    def risk_adjusted_metrics(self, returns):
        if returns.empty:
            return {}
        
        ret = self.return_metrics(returns)
        risk = self.risk_metrics(returns)
        dd = self.drawdown_metrics(returns)
        
        ann_ret = ret['annualized_return']
        ann_vol = risk['annual_volatility']
        downside = risk['downside_deviation']
        max_dd = dd['max_drawdown']
        ulcer = dd['ulcer_index']
        
        sharpe = (ann_ret - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0
        sortino = (ann_ret - self.risk_free_rate) / downside if downside > 0 else 0
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        burke = ann_ret / ulcer if ulcer > 0 else 0
        
        rf_daily = self.risk_free_rate / self.trading_days
        excess = returns - rf_daily
        gains = excess[excess > 0].sum()
        losses_val = abs(excess[excess < 0].sum())
        omega = gains / losses_val if losses_val > 0 else np.inf
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'burke_ratio': burke,
            'omega_ratio': omega
        }
    
    def benchmark_metrics(self, returns, benchmark_returns):
        aligned, bench = returns.align(benchmark_returns, join='inner')
        if aligned.empty:
            return {}
        
        cov = aligned.cov(bench)
        var = bench.var()
        beta = cov / var if var > 0 else 0
        
        rf_daily = self.risk_free_rate / self.trading_days
        strat_excess = aligned.mean() - rf_daily
        bench_excess = bench.mean() - rf_daily
        alpha = (strat_excess - beta * bench_excess) * self.trading_days
        
        active = aligned - bench
        tracking_error = active.std() * np.sqrt(self.trading_days)
        info_ratio = active.mean() * self.trading_days / tracking_error if tracking_error > 0 else 0
        
        up = bench > 0
        down = bench < 0
        up_capture = aligned[up].mean() / bench[up].mean() if up.any() else np.nan
        down_capture = aligned[down].mean() / bench[down].mean() if down.any() else np.nan
        
        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': info_ratio,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'correlation': aligned.corr(bench),
            'r_squared': aligned.corr(bench) ** 2
        }
