import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def load_config(config_path="config/config.yaml"):
    path = Path(config_path)
    
    if not path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return get_default_config()
    
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()


def get_default_config():
    return {
        'data': {
            'start_date': '2014-01-01',
            'end_date': '2024-12-31',
            'universe': 'sp500',
            'cache_dir': 'data/cache'
        },
        'benchmark': {'primary': '^GSPC'},
        'portfolio': {
            'initial_capital': 1000000,
            'max_position_weight': 0.10,
            'min_position_weight': 0.001,
            'cash_buffer': 0.02
        },
        'costs': {
            'commission_rate': 0.001,
            'slippage': 0.001,
            'bid_ask_spread': 0.0005
        },
        'risk': {'risk_free_rate': 0.02},
        'rebalancing': {'default_frequency': 'monthly'},
        'strategies': {
            'momentum': {'lookback_period': 252, 'skip_period': 21, 'top_percentile': 0.2},
            'value': {'top_percentile': 0.2},
            'quality': {'top_percentile': 0.2},
            'risk_parity': {'lookback_volatility': 63},
            'esg': {'min_score': 5.0, 'weighting': 'market_cap'}
        },
        'reporting': {'output_dir': 'reports', 'include_charts': True}
    }
