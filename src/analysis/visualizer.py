import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
from loguru import logger

plt.style.use('seaborn-v0_8-whitegrid')


class Visualizer:
    
    def __init__(self, figsize=(12, 6), color_palette=None, save_dpi=150):
        self.figsize = figsize
        self.save_dpi = save_dpi
        self.colors = color_palette or [
            '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
            '#95C623', '#5D5D5D', '#7B68EE', '#FF6B6B', '#4ECDC4'
        ]
    
    def plot_cumulative_returns(self, results, benchmark=None, title="Cumulative Returns", save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (name, result) in enumerate(results.items()):
            if 'cumulative_returns' in result:
                cum = result['cumulative_returns']
                ax.plot(cum.index, cum * 100, label=name, color=self.colors[i % len(self.colors)], linewidth=1.5)
        
        if benchmark is not None:
            cum_bench = (1 + benchmark).cumprod() - 1
            ax.plot(cum_bench.index, cum_bench * 100, label='Benchmark', color='black', linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        return fig
    
    def plot_drawdowns(self, returns, title="Drawdown Analysis", save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5),
                                        gridspec_kw={'height_ratios': [2, 1]})
        
        cum = (1 + returns).cumprod()
        rolling_max = cum.expanding().max()
        dd = (cum / rolling_max - 1) * 100
        
        ax1.plot(cum.index, cum, color=self.colors[0], linewidth=1.5)
        ax1.plot(rolling_max.index, rolling_max, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax1.fill_between(cum.index, cum, rolling_max, alpha=0.3, color='red')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title(title, fontweight='bold')
        ax1.legend(['Portfolio', 'High Water Mark'])
        
        ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.5)
        ax2.plot(dd.index, dd, color='darkred', linewidth=0.5)
        ax2.set_ylabel('Drawdown (%)')
        
        max_dd_idx = dd.idxmin()
        ax2.scatter([max_dd_idx], [dd[max_dd_idx]], color='black', s=100, zorder=5, marker='v')
        ax2.annotate(f'Max DD: {dd[max_dd_idx]:.1f}%', xy=(max_dd_idx, dd[max_dd_idx]),
                    xytext=(10, -20), textcoords='offset points', fontweight='bold')
        
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
    
    def plot_rolling_metrics(self, returns, windows=[63, 252], metrics=['return', 'volatility', 'sharpe'], save_path=None):
        n = len(metrics)
        fig, axes = plt.subplots(n, 1, figsize=(self.figsize[0], 4 * n), sharex=True)
        if n == 1:
            axes = [axes]
        
        labels = {63: '3M', 126: '6M', 252: '1Y'}
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            for j, window in enumerate(windows):
                label = labels.get(window, f'{window}D')
                color = self.colors[j % len(self.colors)]
                
                if metric == 'return':
                    rolling = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1) * 100
                    ax.axhline(y=0, color='gray', linewidth=0.5)
                elif metric == 'volatility':
                    rolling = returns.rolling(window).std() * np.sqrt(252) * 100
                elif metric == 'sharpe':
                    rf = 0.02 / 252
                    rolling = returns.rolling(window).apply(
                        lambda x: (x.mean() - rf) * np.sqrt(252) / x.std() if x.std() > 0 else 0
                    )
                    ax.axhline(y=0, color='gray', linewidth=0.5)
                    ax.axhline(y=1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
                
                ax.plot(rolling.index, rolling, label=label, color=color, linewidth=1.5)
            
            ax.legend(loc='upper right')
            ax.set_title(f'Rolling {metric.title()}')
        
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.suptitle('Rolling Performance Metrics', fontweight='bold', y=1.02)
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
    
    def plot_return_distribution(self, returns, title="Return Distribution", bins=50, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0], 5))
        
        pct = returns * 100
        ax1.hist(pct, bins=bins, density=True, alpha=0.7, color=self.colors[0], edgecolor='white')
        
        from scipy.stats import norm
        mu, std = pct.mean(), pct.std()
        x = np.linspace(pct.min(), pct.max(), 100)
        ax1.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2)
        ax1.axvline(x=mu, color='black', linestyle='--', linewidth=1)
        ax1.axvline(x=pct.quantile(0.05), color='red', linestyle=':', linewidth=1.5)
        ax1.set_xlabel('Daily Return (%)')
        ax1.set_ylabel('Density')
        ax1.set_title('Return Distribution')
        
        from scipy.stats import probplot
        probplot(returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        fig.suptitle(title, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
    
    def plot_monthly_returns_heatmap(self, returns, title="Monthly Returns Heatmap", save_path=None):
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
        
        df = pd.DataFrame({'year': monthly.index.year, 'month': monthly.index.month, 'return': monthly.values})
        pivot = df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        annual = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1) * 100
        pivot['Annual'] = annual.values
        
        fig, ax = plt.subplots(figsize=(14, len(pivot) * 0.6 + 2))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        
        ax.figure.colorbar(im, ax=ax, shrink=0.8).ax.set_ylabel('Return (%)', rotation=-90, va="bottom")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    color = 'white' if abs(val) > 5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha="center", va="center", color=color, fontsize=8)
        
        ax.set_title(title, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
    
    def plot_strategy_comparison(self, comparison_df, metrics=None, save_path=None):
        if metrics is None:
            metrics = ['Ann. Return', 'Volatility', 'Sharpe', 'Max DD']
        
        available = [m for m in metrics if m in comparison_df.columns]
        n = len(available)
        
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]
        
        strategies = comparison_df.index.tolist()
        x = np.arange(len(strategies))
        
        for i, metric in enumerate(available):
            ax = axes[i]
            values = comparison_df[metric].values
            
            if metric in ['Sharpe', 'Ann. Return', 'Calmar']:
                colors = [self.colors[0] if v >= 0 else 'red' for v in values]
            elif metric in ['Max DD']:
                colors = ['red' if abs(v) > 0.2 else self.colors[0] for v in values]
            else:
                colors = [self.colors[0]] * len(values)
            
            bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
            ax.set_title(metric, fontweight='bold')
            
            for bar, val in zip(bars, values):
                if 'Return' in metric or 'DD' in metric or 'Volatility' in metric:
                    label = f'{val:.1%}'
                else:
                    label = f'{val:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), label, ha='center', va='bottom', fontsize=8)
            
            ax.axhline(y=0, color='gray', linewidth=0.5)
        
        fig.suptitle('Strategy Comparison', fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
    
    def create_tearsheet(self, returns, benchmark_returns=None, strategy_name="Strategy", save_path=None):
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.2)
        
        ax1 = fig.add_subplot(gs[0, :])
        cum = (1 + returns).cumprod()
        ax1.plot(cum.index, (cum - 1) * 100, color=self.colors[0], linewidth=1.5, label=strategy_name)
        if benchmark_returns is not None:
            cum_bench = (1 + benchmark_returns).cumprod()
            ax1.plot(cum_bench.index, (cum_bench - 1) * 100, color='gray', linewidth=1.5, linestyle='--', label='Benchmark')
        ax1.set_title('Cumulative Returns (%)', fontweight='bold')
        ax1.legend()
        ax1.axhline(y=0, color='gray', linewidth=0.5)
        
        ax2 = fig.add_subplot(gs[1, :])
        rolling_max = cum.expanding().max()
        dd = (cum / rolling_max - 1) * 100
        ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.5)
        ax2.set_title('Drawdown (%)', fontweight='bold')
        ax2.set_ylim(dd.min() * 1.1, 5)
        
        ax3 = fig.add_subplot(gs[2, 0])
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
        df = pd.DataFrame({'year': monthly.index.year, 'month': monthly.index.month, 'return': monthly.values})
        pivot = df.pivot(index='year', columns='month', values='return')
        ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-8, vmax=8)
        ax3.set_title('Monthly Returns (%)', fontweight='bold')
        
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.hist(returns * 100, bins=50, density=True, alpha=0.7, color=self.colors[0], edgecolor='white')
        ax4.axvline(x=returns.mean() * 100, color='red', linestyle='--', linewidth=1.5)
        ax4.set_title('Daily Return Distribution', fontweight='bold')
        
        ax5 = fig.add_subplot(gs[3, 0])
        rf = 0.02 / 252
        rolling_sharpe = returns.rolling(252).apply(lambda x: (x.mean() - rf) * np.sqrt(252) / x.std() if x.std() > 0 else 0)
        ax5.plot(rolling_sharpe.index, rolling_sharpe, color=self.colors[0])
        ax5.axhline(y=0, color='gray', linewidth=0.5)
        ax5.axhline(y=1, color='green', linestyle='--', linewidth=0.5, alpha=0.7)
        ax5.set_title('Rolling 1Y Sharpe', fontweight='bold')
        
        ax6 = fig.add_subplot(gs[3, 1])
        rolling_vol = returns.rolling(63).std() * np.sqrt(252) * 100
        ax6.plot(rolling_vol.index, rolling_vol, color=self.colors[0])
        ax6.set_title('Rolling 3M Volatility (%)', fontweight='bold')
        
        ax7 = fig.add_subplot(gs[4, :])
        ax7.axis('off')
        
        from .metrics import PerformanceMetrics
        m = PerformanceMetrics().calculate_all(returns, benchmark_returns)
        
        table_data = [
            ['Total Return', f"{m.get('total_return', 0):.2%}", 'Sharpe', f"{m.get('sharpe_ratio', 0):.2f}"],
            ['Ann. Return', f"{m.get('annualized_return', 0):.2%}", 'Sortino', f"{m.get('sortino_ratio', 0):.2f}"],
            ['Volatility', f"{m.get('annual_volatility', 0):.2%}", 'Max DD', f"{m.get('max_drawdown', 0):.2%}"],
            ['Win Rate', f"{m.get('win_rate', 0):.1%}", 'Calmar', f"{m.get('calmar_ratio', 0):.2f}"]
        ]
        
        table = ax7.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.15, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        fig.suptitle(f'{strategy_name} - Performance Tearsheet', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
