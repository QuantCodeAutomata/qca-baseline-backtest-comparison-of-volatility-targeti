"""
Backtesting engine for volatility targeting strategies.
Handles weight updates, return calculations, and transaction costs.

# Custom — Context7 found no library equivalent (paper Section 2.3)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Container for backtest results."""
    dates: pd.DatetimeIndex
    weights: np.ndarray
    index_returns: np.ndarray
    risky_returns: np.ndarray
    rf_returns: np.ndarray
    index_volatility: Optional[np.ndarray] = None
    risky_volatility: Optional[np.ndarray] = None
    transaction_costs: Optional[np.ndarray] = None
    cumulative_returns: Optional[np.ndarray] = None
    strategy_name: str = ""


class Backtester:
    """
    Backtesting engine for volatility targeting strategies.
    
    Implements chronological simulation with:
    - Daily weight rebalancing
    - Transaction costs (spread)
    - Proper handling of look-ahead bias
    """
    
    def __init__(
        self,
        transaction_cost_bps: float = 5.0,
        initial_weight: float = 1.0
    ):
        """
        Initialize backtester.
        
        Parameters
        ----------
        transaction_cost_bps : float, default=5.0
            Transaction cost in basis points (one-way)
        initial_weight : float, default=1.0
            Initial weight in risky asset
        """
        self.transaction_cost_bps = transaction_cost_bps
        self.transaction_cost_rate = transaction_cost_bps / 10000.0
        self.initial_weight = initial_weight
    
    def run(
        self,
        strategy,
        risky_returns: pd.Series,
        rf_returns: pd.Series
    ) -> BacktestResult:
        """
        Run backtest for a given strategy.
        
        Parameters
        ----------
        strategy : Strategy object
            Strategy instance (BuyAndHoldStrategy, OpenLoopStrategy, or FeedbackControllerStrategy)
        risky_returns : pd.Series
            Daily returns of risky asset
        rf_returns : pd.Series
            Daily risk-free returns
            
        Returns
        -------
        BacktestResult
            Backtest results including weights, returns, and costs
        """
        # Reset strategy state
        strategy.reset()
        
        # Ensure data is aligned
        assert len(risky_returns) == len(rf_returns), "Return series must have same length"
        
        n = len(risky_returns)
        dates = risky_returns.index
        
        # Initialize arrays
        weights = np.zeros(n)
        index_returns = np.zeros(n)
        transaction_costs = np.zeros(n)
        index_volatility = np.zeros(n)
        risky_volatility = np.zeros(n)
        
        # Initial weight
        prev_weight = self.initial_weight
        
        # Iterate through each day
        for k in range(n):
            risky_ret = risky_returns.iloc[k]
            rf_ret = rf_returns.iloc[k]
            
            # Calculate index return using previous weight
            # r_k^ind = r_k * w_{k-1} + r_k^rf * (1 - w_{k-1})
            index_ret = risky_ret * prev_weight + rf_ret * (1 - prev_weight)
            index_returns[k] = index_ret
            
            # Calculate new weight for next period using data up to k
            # This is the weight that will be applied to period [t_k, t_{k+1}]
            new_weight = strategy.get_weight(
                risky_return=risky_ret,
                rf_return=rf_ret
            )
            
            weights[k] = new_weight
            
            # Calculate weight after market drift (post-return)
            # w_{k-1}^post = w_{k-1} * (1 + r_k) / (1 + r_k^ind)
            if abs(index_ret + 1) > 1e-10:
                weight_post_drift = prev_weight * (1 + risky_ret) / (1 + index_ret)
            else:
                weight_post_drift = prev_weight
            
            # Calculate transaction cost based on weight change
            # Cost = |w_k - w_{k-1}^post| * transaction_cost_rate
            weight_change = abs(new_weight - weight_post_drift)
            transaction_costs[k] = weight_change * self.transaction_cost_rate
            
            # Subtract transaction cost from index return
            index_returns[k] -= transaction_costs[k]
            
            # Store volatility estimates if available
            if hasattr(strategy, 'index_vol_estimator'):
                index_volatility[k] = strategy.index_vol_estimator.volatility
            if hasattr(strategy, 'risky_vol_estimator'):
                risky_volatility[k] = strategy.risky_vol_estimator.volatility
            
            # Update previous weight for next iteration
            prev_weight = new_weight
        
        # Calculate cumulative returns
        cumulative_returns = (1 + pd.Series(index_returns, index=dates)).cumprod().values
        
        # Create result object
        result = BacktestResult(
            dates=dates,
            weights=weights,
            index_returns=index_returns,
            risky_returns=risky_returns.values,
            rf_returns=rf_returns.values,
            index_volatility=index_volatility if index_volatility.sum() > 0 else None,
            risky_volatility=risky_volatility if risky_volatility.sum() > 0 else None,
            transaction_costs=transaction_costs,
            cumulative_returns=cumulative_returns,
            strategy_name=strategy.name
        )
        
        return result
    
    def calculate_turnover(self, result: BacktestResult) -> float:
        """
        Calculate portfolio turnover.
        
        Turnover = Σ |w_k - w_{k-1}^post|
        
        Parameters
        ----------
        result : BacktestResult
            Backtest result
            
        Returns
        -------
        float
            Total turnover over the backtest period
        """
        weights = result.weights
        risky_returns = result.risky_returns
        index_returns = result.index_returns
        
        turnover = 0.0
        prev_weight = self.initial_weight
        
        for k in range(len(weights)):
            # Calculate weight after market drift
            if abs(index_returns[k] + 1) > 1e-10:
                weight_post_drift = prev_weight * (1 + risky_returns[k]) / (1 + index_returns[k])
            else:
                weight_post_drift = prev_weight
            
            # Add absolute weight change
            turnover += abs(weights[k] - weight_post_drift)
            
            prev_weight = weights[k]
        
        return turnover


def run_multiple_backtests(
    strategies: list,
    risky_returns: pd.Series,
    rf_returns: pd.Series,
    transaction_cost_bps: float = 5.0
) -> dict[str, BacktestResult]:
    """
    Run backtests for multiple strategies.
    
    Parameters
    ----------
    strategies : list
        List of strategy instances
    risky_returns : pd.Series
        Daily returns of risky asset
    rf_returns : pd.Series
        Daily risk-free returns
    transaction_cost_bps : float, default=5.0
        Transaction cost in basis points
        
    Returns
    -------
    dict[str, BacktestResult]
        Dictionary mapping strategy names to backtest results
    """
    backtester = Backtester(transaction_cost_bps=transaction_cost_bps)
    
    results = {}
    for strategy in strategies:
        result = backtester.run(strategy, risky_returns, rf_returns)
        results[strategy.name] = result
    
    return results
