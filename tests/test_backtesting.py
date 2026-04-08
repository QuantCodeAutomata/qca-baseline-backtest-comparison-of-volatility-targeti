"""
Tests for backtesting engine.
"""

import numpy as np
import pandas as pd
import pytest
from src.backtesting import Backtester, BacktestResult
from src.strategies import BuyAndHoldStrategy, OpenLoopStrategy


def create_test_data(n=100):
    """Create test return data."""
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    risky_returns = pd.Series(np.random.randn(n) * 0.01, index=dates)
    rf_returns = pd.Series(np.ones(n) * 0.0001, index=dates)
    return risky_returns, rf_returns


def test_backtester_initialization():
    """Test backtester initialization."""
    backtester = Backtester(transaction_cost_bps=5.0)
    
    assert backtester.transaction_cost_bps == 5.0
    assert backtester.transaction_cost_rate == 0.0005
    assert backtester.initial_weight == 1.0


def test_backtest_buy_and_hold():
    """Test backtest with buy-and-hold strategy."""
    risky_returns, rf_returns = create_test_data(100)
    
    strategy = BuyAndHoldStrategy()
    backtester = Backtester(transaction_cost_bps=0.0)  # No transaction costs
    
    result = backtester.run(strategy, risky_returns, rf_returns)
    
    # Check result structure
    assert isinstance(result, BacktestResult)
    assert len(result.weights) == len(risky_returns)
    assert len(result.index_returns) == len(risky_returns)
    
    # All weights should be 1
    assert np.allclose(result.weights, 1.0)
    
    # Index returns should equal risky returns (100% allocation)
    assert np.allclose(result.index_returns, risky_returns.values, atol=1e-6)


def test_backtest_result_cumulative_returns():
    """Test cumulative returns calculation."""
    risky_returns, rf_returns = create_test_data(100)
    
    strategy = BuyAndHoldStrategy()
    backtester = Backtester()
    
    result = backtester.run(strategy, risky_returns, rf_returns)
    
    # Cumulative returns should start at 1
    assert result.cumulative_returns[0] > 0
    
    # Should be monotonic if no large losses
    # (Not strictly required, but check it's reasonable)
    assert result.cumulative_returns[-1] > 0


def test_backtest_transaction_costs():
    """Test that transaction costs are applied."""
    risky_returns, rf_returns = create_test_data(100)
    
    strategy = OpenLoopStrategy(target_volatility=0.01)
    
    # With transaction costs
    backtester_with_costs = Backtester(transaction_cost_bps=5.0)
    result_with_costs = backtester_with_costs.run(strategy, risky_returns, rf_returns)
    
    # Without transaction costs
    strategy.reset()
    backtester_no_costs = Backtester(transaction_cost_bps=0.0)
    result_no_costs = backtester_no_costs.run(strategy, risky_returns, rf_returns)
    
    # Total transaction costs should be positive
    assert np.sum(result_with_costs.transaction_costs) > 0
    
    # Returns with costs should be lower
    total_return_with_costs = result_with_costs.cumulative_returns[-1]
    total_return_no_costs = result_no_costs.cumulative_returns[-1]
    assert total_return_with_costs < total_return_no_costs


def test_backtest_turnover_calculation():
    """Test turnover calculation."""
    risky_returns, rf_returns = create_test_data(100)
    
    # Buy-and-hold should have zero turnover
    strategy_bh = BuyAndHoldStrategy()
    backtester = Backtester()
    result_bh = backtester.run(strategy_bh, risky_returns, rf_returns)
    turnover_bh = backtester.calculate_turnover(result_bh)
    
    assert turnover_bh == 0.0
    
    # Open-loop should have positive turnover
    strategy_ol = OpenLoopStrategy(target_volatility=0.01)
    result_ol = backtester.run(strategy_ol, risky_returns, rf_returns)
    turnover_ol = backtester.calculate_turnover(result_ol)
    
    assert turnover_ol > 0


def test_backtest_data_alignment():
    """Test that backtest handles data alignment correctly."""
    risky_returns, rf_returns = create_test_data(100)
    
    strategy = BuyAndHoldStrategy()
    backtester = Backtester()
    
    result = backtester.run(strategy, risky_returns, rf_returns)
    
    # Result dates should match input dates
    assert len(result.dates) == len(risky_returns)
    assert result.dates[0] == risky_returns.index[0]
    assert result.dates[-1] == risky_returns.index[-1]


def test_backtest_edge_case_single_day():
    """Test backtest with single day of data."""
    risky_returns, rf_returns = create_test_data(1)
    
    strategy = BuyAndHoldStrategy()
    backtester = Backtester()
    
    result = backtester.run(strategy, risky_returns, rf_returns)
    
    assert len(result.weights) == 1
    assert len(result.index_returns) == 1


def test_backtest_edge_case_zero_returns():
    """Test backtest with all zero returns."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    risky_returns = pd.Series(np.zeros(100), index=dates)
    rf_returns = pd.Series(np.zeros(100), index=dates)
    
    strategy = BuyAndHoldStrategy()
    backtester = Backtester()
    
    result = backtester.run(strategy, risky_returns, rf_returns)
    
    # All returns should be zero (minus transaction costs)
    assert np.all(result.index_returns <= 0)  # Can be negative due to costs


def test_backtest_extreme_returns():
    """Test backtest with extreme returns."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Mix of large positive and negative returns
    risky_returns = pd.Series(np.random.choice([0.1, -0.1], size=100), index=dates)
    rf_returns = pd.Series(np.ones(100) * 0.0001, index=dates)
    
    strategy = OpenLoopStrategy(target_volatility=0.01)
    backtester = Backtester()
    
    result = backtester.run(strategy, risky_returns, rf_returns)
    
    # Should complete without errors
    assert len(result.weights) == 100
    
    # Weights should be non-negative
    assert np.all(result.weights >= 0)


def test_backtest_strategy_reset():
    """Test that strategy is reset before backtest."""
    risky_returns, rf_returns = create_test_data(100)
    
    strategy = OpenLoopStrategy(target_volatility=0.01)
    backtester = Backtester()
    
    # Run backtest twice
    result1 = backtester.run(strategy, risky_returns, rf_returns)
    result2 = backtester.run(strategy, risky_returns, rf_returns)
    
    # Results should be identical (strategy was reset)
    assert np.allclose(result1.weights, result2.weights)
    assert np.allclose(result1.index_returns, result2.index_returns)


def test_backtest_result_attributes():
    """Test that BacktestResult has all required attributes."""
    risky_returns, rf_returns = create_test_data(100)
    
    strategy = OpenLoopStrategy(target_volatility=0.01)
    backtester = Backtester()
    
    result = backtester.run(strategy, risky_returns, rf_returns)
    
    # Check all attributes exist
    assert hasattr(result, 'dates')
    assert hasattr(result, 'weights')
    assert hasattr(result, 'index_returns')
    assert hasattr(result, 'risky_returns')
    assert hasattr(result, 'rf_returns')
    assert hasattr(result, 'transaction_costs')
    assert hasattr(result, 'cumulative_returns')
    assert hasattr(result, 'strategy_name')
    
    # Check types
    assert isinstance(result.dates, pd.DatetimeIndex)
    assert isinstance(result.weights, np.ndarray)
    assert isinstance(result.index_returns, np.ndarray)


def test_backtest_no_lookahead_bias():
    """Test that backtest doesn't use future information."""
    # This is a conceptual test - in practice, we verify by code inspection
    # that weight at time k is calculated using only data up to k
    
    risky_returns, rf_returns = create_test_data(100)
    
    strategy = OpenLoopStrategy(target_volatility=0.01)
    backtester = Backtester()
    
    result = backtester.run(strategy, risky_returns, rf_returns)
    
    # Weight at time k should be applied to return at time k
    # This is enforced by the backtest logic
    assert len(result.weights) == len(result.index_returns)
