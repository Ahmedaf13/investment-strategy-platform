import argparse
from loguru import logger

from src.utils.config import load_config
from src.utils.helpers import setup_logging
from src.data.collector import DataCollector
from src.data.cleaner import DataCleaner
from src.data.features import FeatureEngineer
from src.data.universe import UniverseManager
from src.strategies.market_cap import MarketCapWeightedStrategy
from src.strategies.equal_weight import EqualWeightedStrategy
from src.strategies.value import ValueStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.quality import QualityStrategy
from src.strategies.risk_parity import RiskParityStrategy, InverseVolatilityStrategy
from src.strategies.esg import ESGStrategy
from src.backtesting.engine import BacktestEngine
from src.backtesting.costs import TransactionCostModel
from src.analysis.reporter import ReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Portfolio Backtesting Framework")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--strategies', type=str, nargs='+', default=['all'], help='Strategies to run')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--quick', action='store_true', help='Quick backtest with fewer stocks')
    parser.add_argument('--no-cache', action='store_true', help='Disable data caching')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def get_strategies(strategy_names, config):
    all_strategies = {
        'market_cap': MarketCapWeightedStrategy(),
        'equal_weight': EqualWeightedStrategy(),
        'value': ValueStrategy(),
        'momentum': MomentumStrategy(),
        'quality': QualityStrategy(),
        'risk_parity': RiskParityStrategy(),
        'inverse_vol': InverseVolatilityStrategy(),
        'esg': ESGStrategy()
    }
    
    if 'all' in strategy_names:
        return list(all_strategies.values())
    
    return [all_strategies[name] for name in strategy_names if name in all_strategies]


def main():
    args = parse_args()
    config = load_config(args.config)
    setup_logging(level=args.log_level, log_file='logs/backtest.log')
    
    logger.info("=" * 60)
    logger.info("PORTFOLIO BACKTESTING FRAMEWORK")
    logger.info("=" * 60)
    
    data_config = config.get('data', {})
    start_date = args.start_date or data_config.get('start_date', '2014-01-01')
    end_date = args.end_date or data_config.get('end_date', '2024-12-31')
    
    logger.info(f"Backtest period: {start_date} to {end_date}")
    
    collector = DataCollector(start_date=start_date, end_date=end_date, cache_dir=data_config.get('cache_dir', 'data/cache'))
    
    logger.info("Collecting data...")
    tickers = collector.get_sp500_tickers()
    
    if args.quick:
        tickers = tickers[:50]
        logger.info(f"Quick mode: using {len(tickers)} stocks")
    
    use_cache = not args.no_cache
    
    prices = collector.download_price_data(tickers, use_cache=use_cache)
    fundamentals = collector.download_fundamental_data(tickers, use_cache=use_cache)
    market_caps = collector.download_market_cap_history(tickers, prices, use_cache=use_cache)
    benchmarks = collector.download_benchmark_data(use_cache=use_cache)
    
    logger.info("Cleaning data...")
    cleaner = DataCleaner()
    prices = cleaner.clean_price_data(prices)
    
    data = {
        'tickers': tickers,
        'prices': prices,
        'fundamentals': fundamentals,
        'market_caps': market_caps,
        'benchmarks': benchmarks
    }
    
    strategies = get_strategies(args.strategies, config)
    logger.info(f"Running {len(strategies)} strategies: {[s.name for s in strategies]}")
    
    portfolio_config = config.get('portfolio', {})
    costs_config = config.get('costs', {})
    
    cost_model = TransactionCostModel(
        commission_rate=costs_config.get('commission_rate', 0.001),
        slippage_rate=costs_config.get('slippage', 0.001),
        bid_ask_spread=costs_config.get('bid_ask_spread', 0.0005)
    )
    
    engine = BacktestEngine(
        initial_capital=portfolio_config.get('initial_capital', 1_000_000),
        cost_model=cost_model
    )
    
    results = engine.run_multiple(strategies, data)
    
    comparison = engine.compare_strategies()
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 60)
    logger.info("\n" + comparison.to_string())
    
    benchmark_returns = None
    if benchmarks is not None and not benchmarks.empty:
        if '^GSPC' in benchmarks.columns:
            benchmark_returns = benchmarks['^GSPC'].pct_change().dropna()
    
    reporting_config = config.get('reporting', {})
    reporter = ReportGenerator(
        output_dir=reporting_config.get('output_dir', 'reports'),
        include_charts=reporting_config.get('include_charts', True)
    )
    
    report_path = reporter.generate_full_report(results, benchmark_returns=benchmark_returns)
    logger.info(f"Report saved to: {report_path}")
    
    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
