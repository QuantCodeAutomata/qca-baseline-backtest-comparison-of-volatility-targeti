"""
Tests for performance metrics.
"""

import numpy as np
import pandas as pd
import pytest
from src.metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_kalmar_ratio,
    calculate_volatility_tracking_error
)


def test_annualized_return_positive():
    """Test annualized return with positive returns."""
    # 1% daily return for 252 days
    returns = np.ones(252) * 0.01
    annual_return = calculate_annualized_return(returns)
    
    # Should be approximately (1.01)^252 - 1 ≈ 11.6
    assert 10 < annual_return < 13


def test_annualized_return_zero():
    """Test annualized return with zero returns."""
    returns = np.zeros(252)
    annual_return = calculate_annualized_return(returns)
    
    assert annual_return == 0.0


def test_annualized_return_negative():
    """Test annualized return with negative returns."""
    returns = np.ones(252) * -0.01
    annual_return = calculate_annualized_return(returns)
    
    # Should be negative
    assert annual_return < 0


def test_annualized_volatility():
    """Test annualized volatility calculation."""
    # Random returns with known std
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01
    
    annual_vol = calculate_annualized_volatility(returns)
    
    # Should be approximately std * sqrt(252)
    expected = np.std(returns, ddof=1) * np.sqrt(252)
    assert np.abs(annual_vol - expected) < 1e-10


def test_annualized_volatility_constant_returns():
    """Test volatility with constant returns."""
    returns = np.ones(252) * 0.01
    annual_vol = calculate_annualized_volatility(returns)
    
    # Volatility should be zero
    assert annual_vol == 0.0


def test_max_drawdown_no_drawdown():
    """Test max drawdown with monotonically increasing returns."""
    returns = np.ones(252) * 0.01
    max_dd = calculate_max_drawdown(returns)
    
    # Should be zero (no drawdown)
    assert max_dd == 0.0


def test_max_drawdown_with_loss():
    """Test max drawdown with losses."""
    # Start with gains, then large loss
    returns = np.concatenate([np.ones(100) * 0.01, np.ones(50) * -0.02, np.ones(100) * 0.01])
    max_dd = calculate_max_drawdown(returns)
    
    # Should be positive
    assert max_dd > 0


def test_max_drawdown_range():
    """Test that max drawdown is between 0 and 1."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.02
    max_dd = calculate_max_drawdown(returns)
    
    # Should be between 0 and 1
    assert 0 <= max_dd <= 1


def test_sharpe_ratio_positive():
    """Test Sharpe ratio with positive excess returns."""
    # Positive returns
    returns = np.random.randn(252) * 0.01 + 0.001  # Mean > 0
    rf_returns = np.ones(252) * 0.0001
    
    sharpe = calculate_sharpe_ratio(returns, rf_returns)
    
    # Should be positive
    assert sharpe > 0


def test_sharpe_ratio_zero_volatility():
    """Test Sharpe ratio with zero volatility."""
    returns = np.ones(252) * 0.01
    rf_returns = np.ones(252) * 0.0001
    
    sharpe = calculate_sharpe_ratio(returns, rf_returns)
    
    # Should return 0 (or handle gracefully)
    assert sharpe == 0.0


def test_kalmar_ratio_positive():
    """Test Kalmar ratio with positive returns and drawdown."""
    # Mix of positive and negative returns
    np.random.seed(42)
    returns = np.random.randn(252) * 0.02 + 0.001
    
    kalmar = calculate_kalmar_ratio(returns)
    
    # Should be a finite number
    assert np.isfinite(kalmar)


def test_kalmar_ratio_no_drawdown():
    """Test Kalmar ratio with no drawdown."""
    returns = np.ones(252) * 0.01
    kalmar = calculate_kalmar_ratio(returns)
    
    # Should return 0 (no drawdown)
    assert kalmar == 0.0


def test_volatility_tracking_error():
    """Test volatility tracking error calculation."""
    # Simulated index volatility around target
    target_vol = 0.01
    index_vol = np.random.randn(252) * 0.001 + target_vol
    
    tracking_error = calculate_volatility_tracking_error(index_vol, target_vol)
    
    # Should be positive
    assert tracking_error > 0
    
    # Should be small if index_vol is close to target
    assert tracking_error < 0.1


def test_volatility_tracking_error_perfect():
    """Test tracking error with perfect tracking."""
    target_vol = 0.01
    index_vol = np.ones(252) * target_vol
    
    tracking_error = calculate_volatility_tracking_error(index_vol, target_vol)
    
    # Should be zero
    assert tracking_error == 0.0


def test_volatility_tracking_error_large_deviation():
    """Test tracking error with large deviation."""
    target_vol = 0.01
    index_vol = np.ones(252) * 0.05  # 5x target
    
    tracking_error = calculate_volatility_tracking_error(index_vol, target_vol)
    
    # Should be large
    assert tracking_error > 0.5


def test_metrics_with_edge_case_single_day():
    """Test metrics with single day of data."""
    returns = np.array([0.01])
    rf_returns = np.array([0.0001])
    
    # Should handle gracefully
    annual_return = calculate_annualized_return(returns)
    assert np.isfinite(annual_return)
    
    # Volatility should be zero (only one data point)
    annual_vol = calculate_annualized_volatility(returns)
    assert annual_vol == 0.0 or np.isnan(annual_vol)


def test_metrics_with_extreme_values():
    """Test metrics with extreme return values."""
    # Very large returns
    returns = np.array([0.5, -0.3, 0.4, -0.2] * 63)  # 252 days
    rf_returns = np.ones(252) * 0.0001
    
    # Should handle without overflow
    annual_return = calculate_annualized_return(returns)
    assert np.isfinite(annual_return)
    
    sharpe = calculate_sharpe_ratio(returns, rf_returns)
    assert np.isfinite(sharpe)
    
    max_dd = calculate_max_drawdown(returns)
    assert 0 <= max_dd <= 1


def test_sharpe_ratio_negative():
    """Test Sharpe ratio with negative excess returns."""
    # Negative returns
    returns = np.random.randn(252) * 0.01 - 0.001  # Mean < 0
    rf_returns = np.ones(252) * 0.0001
    
    sharpe = calculate_sharpe_ratio(returns, rf_returns)
    
    # Should be negative
    assert sharpe < 0


def test_kalmar_ratio_negative_returns():
    """Test Kalmar ratio with negative returns."""
    # Consistently negative returns
    returns = np.ones(252) * -0.001
    
    kalmar = calculate_kalmar_ratio(returns)
    
    # Should be negative (negative return / positive drawdown)
    assert kalmar < 0


def test_annualized_return_consistency():
    """Test that annualized return is consistent with cumulative return."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01
    
    # Calculate annualized return
    annual_return = calculate_annualized_return(returns)
    
    # Calculate cumulative return
    cumulative_return = np.prod(1 + returns) - 1
    
    # Annualized should be approximately (1 + cumulative)^(1/1) - 1 for 1 year
    expected = (1 + cumulative_return) ** (252 / len(returns)) - 1
    
    assert np.abs(annual_return - expected) < 1e-10


def test_volatility_positive():
    """Test that volatility is always non-negative."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.02
    
    annual_vol = calculate_annualized_volatility(returns)
    
    assert annual_vol >= 0


def test_max_drawdown_positive():
    """Test that max drawdown is always non-negative."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.02
    
    max_dd = calculate_max_drawdown(returns)
    
    assert max_dd >= 0
