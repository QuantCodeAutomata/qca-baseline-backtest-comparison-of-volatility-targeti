"""
Tests for volatility estimation module.
"""

import numpy as np
import pytest
from src.volatility import (
    calculate_ewma_decay_factor,
    ewma_variance,
    ewma_volatility,
    EWMAVolatilityEstimator,
    annualize_volatility,
    daily_target_volatility
)


def test_ewma_decay_factor():
    """Test EWMA decay factor calculation."""
    # Half-life of 126 days
    beta = calculate_ewma_decay_factor(126)
    
    # After 126 periods, weight should be approximately 0.5
    assert 0.4 < beta ** 126 < 0.6
    
    # Beta should be between 0 and 1
    assert 0 < beta < 1


def test_ewma_variance_constant_returns():
    """Test EWMA variance with constant returns."""
    # Constant returns should give constant variance
    returns = np.ones(100) * 0.01
    var = ewma_variance(returns, half_life=126)
    
    # Variance should converge to r^2
    assert np.abs(var[-1] - 0.01**2) < 1e-6


def test_ewma_variance_zero_returns():
    """Test EWMA variance with zero returns."""
    returns = np.zeros(100)
    var = ewma_variance(returns, half_life=126)
    
    # All variances should be zero
    assert np.allclose(var, 0)


def test_ewma_variance_finite_sample_correction():
    """Test finite-sample correction in EWMA variance."""
    returns = np.random.randn(100) * 0.01
    
    # With correction
    var_corrected = ewma_variance(returns, half_life=126, finite_sample_correction=True)
    
    # Without correction
    var_uncorrected = ewma_variance(returns, half_life=126, finite_sample_correction=False)
    
    # Early values should differ more
    assert np.abs(var_corrected[0] - var_uncorrected[0]) > 1e-10
    
    # Later values should converge
    assert np.abs(var_corrected[-1] - var_uncorrected[-1]) < np.abs(var_corrected[0] - var_uncorrected[0])


def test_ewma_volatility_annualization():
    """Test volatility annualization."""
    returns = np.random.randn(252) * 0.01
    
    # Daily volatility
    vol_daily = ewma_volatility(returns, half_life=126, annualize=False)
    
    # Annualized volatility
    vol_annual = ewma_volatility(returns, half_life=126, annualize=True)
    
    # Annualized should be approximately sqrt(252) times daily
    assert np.abs(vol_annual[-1] / vol_daily[-1] - np.sqrt(252)) < 0.1


def test_ewma_estimator_stateful():
    """Test stateful EWMA estimator."""
    estimator = EWMAVolatilityEstimator(half_life=126)
    
    returns = np.random.randn(100) * 0.01
    
    # Update sequentially
    for r in returns:
        var, vol = estimator.update(r)
        
        # Variance and volatility should be non-negative
        assert var >= 0
        assert vol >= 0
    
    # Time index should match number of updates
    assert estimator.k == len(returns)


def test_ewma_estimator_reset():
    """Test EWMA estimator reset."""
    estimator = EWMAVolatilityEstimator(half_life=126)
    
    # Update with some data
    for _ in range(10):
        estimator.update(0.01)
    
    # Reset
    estimator.reset()
    
    # State should be reset
    assert estimator.k == 0
    assert estimator.weighted_sum == 0.0
    assert estimator.variance == 0.0
    assert estimator.volatility == 0.0


def test_ewma_estimator_consistency():
    """Test consistency between batch and online EWMA calculation."""
    returns = np.random.randn(100) * 0.01
    
    # Batch calculation
    var_batch = ewma_variance(returns, half_life=126)
    
    # Online calculation
    estimator = EWMAVolatilityEstimator(half_life=126)
    var_online = []
    for r in returns:
        var, _ = estimator.update(r)
        var_online.append(var)
    var_online = np.array(var_online)
    
    # Should be very close
    assert np.allclose(var_batch, var_online, rtol=1e-10)


def test_annualize_volatility():
    """Test volatility annualization function."""
    daily_vol = 0.01
    annual_vol = annualize_volatility(daily_vol, trading_days_per_year=252)
    
    # Should be approximately sqrt(252) times daily
    assert np.abs(annual_vol - daily_vol * np.sqrt(252)) < 1e-10


def test_daily_target_volatility():
    """Test conversion of annual to daily target volatility."""
    annual_target = 0.15
    daily_target = daily_target_volatility(annual_target, trading_days_per_year=252)
    
    # Should be approximately annual / sqrt(252)
    assert np.abs(daily_target - annual_target / np.sqrt(252)) < 1e-10


def test_ewma_variance_positive():
    """Test that EWMA variance is always non-negative."""
    # Random returns including negative values
    returns = np.random.randn(100) * 0.02
    var = ewma_variance(returns, half_life=126)
    
    # All variances should be non-negative
    assert np.all(var >= 0)


def test_ewma_variance_increasing_with_volatility():
    """Test that EWMA variance increases with return volatility."""
    # Low volatility returns
    returns_low = np.random.randn(100) * 0.001
    var_low = ewma_variance(returns_low, half_life=126)
    
    # High volatility returns
    returns_high = np.random.randn(100) * 0.05
    var_high = ewma_variance(returns_high, half_life=126)
    
    # High volatility should give higher variance on average
    assert np.mean(var_high) > np.mean(var_low)


def test_ewma_edge_case_single_return():
    """Test EWMA with single return."""
    returns = np.array([0.01])
    var = ewma_variance(returns, half_life=126)
    
    # Should equal r^2
    assert np.abs(var[0] - 0.01**2) < 1e-10


def test_ewma_edge_case_two_returns():
    """Test EWMA with two returns."""
    returns = np.array([0.01, 0.02])
    var = ewma_variance(returns, half_life=126)
    
    # Should have two variance estimates
    assert len(var) == 2
    assert var[0] > 0
    assert var[1] > 0
