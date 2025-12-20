import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    quantity: float
    price: float
    side: str
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def gross_value(self):
        return self.quantity * self.price
    
    @property
    def total_cost(self):
        return self.commission + self.slippage


class Portfolio:
    
    def __init__(self, initial_capital=1_000_000, cash_buffer=0.02):
        self.initial_capital = initial_capital
        self.cash_buffer = cash_buffer
        self.cash = initial_capital
        self.holdings = {}
        self.value_history = []
        self.trade_history = []
        self.weight_history = []
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_turnover = 0.0
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    def get_value(self, prices):
        holdings_value = sum(qty * prices.get(ticker, 0) for ticker, qty in self.holdings.items())
        return self.cash + holdings_value
    
    def get_weights(self, prices):
        total_value = self.get_value(prices)
        if total_value == 0:
            return pd.Series(dtype=float)
        
        weights = {ticker: qty * prices.get(ticker, 0) / total_value 
                  for ticker, qty in self.holdings.items()}
        weights['_CASH'] = self.cash / total_value
        return pd.Series(weights)
    
    def execute_trades(self, target_weights, prices, date, cost_model=None):
        from .costs import TransactionCostModel
        cost_model = cost_model or TransactionCostModel()
        
        current_weights = self.get_weights(prices)
        total_value = self.get_value(prices)
        trades = []
        
        all_tickers = set(target_weights.index) | set(self.holdings.keys())
        all_tickers.discard('_CASH')
        
        for ticker in all_tickers:
            if ticker not in prices or prices[ticker] <= 0:
                continue
            
            current_weight = current_weights.get(ticker, 0)
            target_weight = target_weights.get(ticker, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) < 0.001:
                continue
            
            value_diff = weight_diff * total_value
            price = prices[ticker]
            quantity = abs(value_diff) / price
            
            if value_diff > 0:
                trade = self._execute_buy(ticker, quantity, price, date, cost_model)
            else:
                trade = self._execute_sell(ticker, quantity, price, date, cost_model)
            
            if trade:
                trades.append(trade)
                self.trade_history.append(trade)
        
        self.total_turnover += sum(t.gross_value for t in trades) / total_value / 2
        return trades
    
    def _execute_buy(self, ticker, quantity, price, date, cost_model):
        gross_value = quantity * price
        commission = cost_model.calculate_commission(gross_value)
        slippage = cost_model.calculate_slippage(gross_value)
        total_cost = gross_value + commission + slippage
        
        if total_cost > self.cash:
            available = self.cash - commission - slippage
            if available <= 0:
                return None
            quantity = available / price
            gross_value = quantity * price
            commission = cost_model.calculate_commission(gross_value)
            slippage = cost_model.calculate_slippage(gross_value)
            total_cost = gross_value + commission + slippage
        
        self.cash -= total_cost
        self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
        self.total_commission += commission
        self.total_slippage += slippage
        
        return Trade(date, ticker, quantity, price, 'BUY', commission, slippage)
    
    def _execute_sell(self, ticker, quantity, price, date, cost_model):
        current_qty = self.holdings.get(ticker, 0)
        quantity = min(quantity, current_qty)
        
        if quantity <= 0:
            return None
        
        gross_value = quantity * price
        commission = cost_model.calculate_commission(gross_value)
        slippage = cost_model.calculate_slippage(gross_value)
        
        self.cash += gross_value - commission - slippage
        self.holdings[ticker] -= quantity
        
        if self.holdings[ticker] <= 0:
            del self.holdings[ticker]
        
        self.total_commission += commission
        self.total_slippage += slippage
        
        return Trade(date, ticker, quantity, price, 'SELL', commission, slippage)
    
    def record_state(self, date, prices):
        value = self.get_value(prices)
        self.value_history.append({
            'date': date, 'total_value': value, 'cash': self.cash,
            'holdings_value': value - self.cash, 'n_positions': len(self.holdings)
        })
        
        weight_record = {'date': date}
        weight_record.update(self.get_weights(prices).to_dict())
        self.weight_history.append(weight_record)
    
    def get_value_series(self):
        if not self.value_history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.value_history)
        return df.set_index('date')['total_value']
    
    def get_returns(self):
        return self.get_value_series().pct_change().dropna()
    
    def get_cumulative_returns(self):
        values = self.get_value_series()
        return values / values.iloc[0] - 1
    
    def get_trade_summary(self):
        if not self.trade_history:
            return {'total_trades': 0, 'total_commission': 0, 'total_slippage': 0, 'total_turnover': 0}
        
        return {
            'total_trades': len(self.trade_history),
            'buy_trades': len([t for t in self.trade_history if t.side == 'BUY']),
            'sell_trades': len([t for t in self.trade_history if t.side == 'SELL']),
            'total_volume': sum(t.gross_value for t in self.trade_history),
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_costs': self.total_commission + self.total_slippage,
            'total_turnover': self.total_turnover
        }
