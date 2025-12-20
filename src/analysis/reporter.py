import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger

from .metrics import PerformanceMetrics
from .visualizer import Visualizer


class ReportGenerator:
    
    def __init__(self, output_dir="reports", include_charts=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_charts = include_charts
        self.metrics = PerformanceMetrics()
        self.visualizer = Visualizer()
        logger.info(f"ReportGenerator initialized: output_dir={output_dir}")
    
    def generate_full_report(self, results, benchmark_returns=None, report_name="backtest_report"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"{report_name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating report in {report_dir}")
        
        summary = self._generate_summary(results, benchmark_returns)
        summary.to_csv(report_dir / "summary_metrics.csv")
        logger.info(f"Saved summary to {report_dir / 'summary_metrics.csv'}")
        
        for name, result in results.items():
            if 'error' not in result:
                self._generate_strategy_report(name, result, benchmark_returns, report_dir)
        
        if self.include_charts and len(results) > 1:
            self._generate_comparison_charts(results, benchmark_returns, report_dir)
        
        self._generate_html(results, summary, report_dir)
        
        logger.info(f"Report generated: {report_dir}")
        return str(report_dir)
    
    def _generate_summary(self, results, benchmark_returns=None):
        rows = []
        for name, result in results.items():
            if 'error' in result:
                continue
            
            m = result.get('metrics', {})
            t = result.get('trade_summary', {})
            
            row = {
                'Strategy': name,
                'Total Return': m.get('total_return', 0),
                'Ann. Return': m.get('annualized_return', 0),
                'Volatility': m.get('volatility', 0),
                'Sharpe': m.get('sharpe_ratio', 0),
                'Sortino': m.get('sortino_ratio', 0),
                'Max Drawdown': m.get('max_drawdown', 0),
                'Calmar': m.get('calmar_ratio', 0),
                'Win Rate': m.get('win_rate', 0),
                'VaR 95%': m.get('var_95', 0),
                'Total Trades': t.get('total_trades', 0),
                'Turnover': t.get('total_turnover', 0),
                'Total Costs': t.get('total_costs', 0),
                'Final Value': result.get('final_value', 0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index('Strategy').sort_values('Sharpe', ascending=False)
        return df
    
    def _generate_strategy_report(self, name, result, benchmark_returns, report_dir):
        strategy_dir = report_dir / name.replace(' ', '_').lower()
        strategy_dir.mkdir(exist_ok=True)
        
        returns = result.get('returns')
        if returns is None or returns.empty:
            return
        
        bench = benchmark_returns if benchmark_returns is not None and not benchmark_returns.empty else None
        detailed = self.metrics.calculate_all(returns, bench)
        
        pd.DataFrame([detailed]).T.to_csv(strategy_dir / 'detailed_metrics.csv')
        returns.to_csv(strategy_dir / 'daily_returns.csv')
        
        if self.include_charts:
            try:
                self.visualizer.create_tearsheet(returns, bench, name, save_path=str(strategy_dir / 'tearsheet.png'))
                self.visualizer.plot_drawdowns(returns, title=f'{name} - Drawdowns', save_path=str(strategy_dir / 'drawdowns.png'))
                self.visualizer.plot_rolling_metrics(returns, save_path=str(strategy_dir / 'rolling_metrics.png'))
                self.visualizer.plot_return_distribution(returns, title=f'{name} - Returns', save_path=str(strategy_dir / 'distribution.png'))
                self.visualizer.plot_monthly_returns_heatmap(returns, title=f'{name} - Monthly', save_path=str(strategy_dir / 'monthly_heatmap.png'))
            except Exception as e:
                logger.error(f"Error generating charts for {name}: {e}")
        
        logger.info(f"Generated report for {name}")
    
    def _generate_comparison_charts(self, results, benchmark_returns, report_dir):
        comparison_dir = report_dir / 'comparison'
        comparison_dir.mkdir(exist_ok=True)
        
        try:
            self.visualizer.plot_cumulative_returns(results, benchmark=benchmark_returns,
                                                    save_path=str(comparison_dir / 'cumulative_returns.png'))
            
            summary = self._generate_summary(results, benchmark_returns)
            self.visualizer.plot_strategy_comparison(summary, save_path=str(comparison_dir / 'strategy_comparison.png'))
        except Exception as e:
            logger.error(f"Error generating comparison charts: {e}")
    
    def _generate_html(self, results, summary, report_dir):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sections = ""
        for name, result in results.items():
            if 'error' in result:
                continue
            safe_name = name.replace(' ', '_').lower()
            m = result.get('metrics', {})
            sections += f"""
            <div class="strategy-section">
                <h3>{name}</h3>
                <ul>
                    <li>Total Return: {m.get('total_return', 0):.2%}</li>
                    <li>Sharpe Ratio: {m.get('sharpe_ratio', 0):.2f}</li>
                    <li>Max Drawdown: {m.get('max_drawdown', 0):.2%}</li>
                </ul>
                <img src="{safe_name}/tearsheet.png" style="max-width:100%;">
            </div>"""
        
        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Backtest Report</title>
<style>
body {{ font-family: Arial, sans-serif; padding: 20px; max-width: 1200px; margin: auto; }}
.header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
.strategy-section {{ border: 1px solid #ddd; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
table {{ width: 100%; border-collapse: collapse; }} th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
</style></head>
<body>
<div class="header"><h1>Portfolio Backtest Report</h1><p>Generated: {timestamp}</p></div>
<h2>Summary</h2>
{summary.to_html()}
<h2>Strategy Analysis</h2>
{sections}
</body></html>"""
        
        with open(report_dir / 'report.html', 'w') as f:
            f.write(html)
        logger.info(f"Generated HTML report: {report_dir / 'report.html'}")
    
    def generate_excel_report(self, results, benchmark_returns=None, filename="backtest_report.xlsx"):
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            summary = self._generate_summary(results, benchmark_returns)
            summary.to_excel(writer, sheet_name='Summary')
            
            for name, result in results.items():
                if 'error' in result:
                    continue
                safe = name[:31].replace(' ', '_')
                
                if result.get('returns') is not None:
                    result['returns'].to_excel(writer, sheet_name=f'{safe}_Ret')
                
                if result.get('metrics'):
                    pd.DataFrame([result['metrics']]).T.to_excel(writer, sheet_name=f'{safe}_Met')
        
        logger.info(f"Generated Excel report: {filepath}")
        return str(filepath)
