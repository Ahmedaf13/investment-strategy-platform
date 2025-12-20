# Portfolio Backtesting Framework

A Python framework for portfolio construction and strategy backtesting.

## Features

- **Strategies**: Market Cap, Equal Weight, Value, Momentum, Quality, Risk Parity, ESG
- **Backtesting**: Transaction costs, slippage, position limits, rebalancing
- **Analysis**: Sharpe, Sortino, Drawdown, VaR, Alpha, Beta, and more
- **Reports**: HTML reports with charts and Excel exports

## Quick Start

```bash
pip install -r requirements.txt
python main.py --quick
```

## Usage

```bash
# Full backtest
python main.py

# Specific strategies
python main.py --strategies momentum value quality

# Custom date range
python main.py --start-date 2018-01-01 --end-date 2023-12-31
```


## Configuration

Edit `config/config.yaml` to customize:

- Date range and universe
- Portfolio parameters
- Transaction costs
- Strategy settings

## Strategies

| Strategy | Description |
|----------|-------------|
| Market Cap | Weight by market capitalization |
| Equal Weight | Equal allocation across stocks |
| Value | Rank by P/E, P/B, dividend yield |
| Momentum | Rank by 12-month returns |
| Quality | Rank by ROE, margins, debt |
| Risk Parity | Inverse volatility weighting |
| ESG | Filter by ESG scores |

