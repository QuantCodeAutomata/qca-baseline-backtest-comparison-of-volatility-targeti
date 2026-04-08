"""
Tests for strategy implementations.
"""

import numpy as np
import pytest
from src.strategies import BuyAndHoldStrategy, OpenLoopStrategy, FeedbackControllerStrategy


def test_buy_and_hold_always_one():
    """Test that buy-and-hold always returns weight of 1."""
    strategy = BuyAndHoldStrategy()
    
    # Should always return 1 regardless of inputs
    assert strategy.get_weight() == 1.0
    assert strategy.get_weight(risky_return=0.05) == 1.0
    assert strategy.get_weight(risky_return=-0.05) == 1.0


def test_buy_and_hold_reset():
    """Test buy-and-hold reset (no-op)."""
    strategy = BuyAndHoldStrategy()
    strategy.reset()  # Should not raise error
    assert strategy.get_weight() == 1.0


def test_open_loop_weight_calculation():
    """Test open-loop weight calculation."""
    target_vol = 0.01
    strategy = OpenLoopStrategy(target_volatility=target_vol, leverage_limit=1.5)
    
    # Feed some returns to build up volatility estimate
    returns = np.random.randn(50) * 0.02
    
    for r in returns:
        weight = strategy.get_weight(risky_return=r)
        
        # Weight should be non-negative
        assert weight >= 0
        
        # Weight should respect leverage limit
        assert weight <= 1.5


def test_open_loop_leverage_limit():
    """Test that open-loop respects leverage limit."""
    target_vol = 0.05  # High target
    strategy = OpenLoopStrategy(target_volatility=target_vol, leverage_limit=1.5)
    
    # Feed low volatility returns
    returns = np.random.randn(50) * 0.001
    
    for r in returns:
        weight = strategy.get_weight(risky_return=r)
        
        # Should hit leverage limit
        assert weight <= 1.5


def test_open_loop_zero_volatility():
    """Test open-loop with zero volatility (edge case)."""
    target_vol = 0.01
    strategy = OpenLoopStrategy(target_volatility=target_vol)
    
    # Feed zero returns
    for _ in range(10):
        weight = strategy.get_weight(risky_return=0.0)
        
        # Should return 0 to avoid division by zero
        assert weight == 0.0


def test_open_loop_reset():
    """Test open-loop reset."""
    strategy = OpenLoopStrategy(target_volatility=0.01)
    
    # Feed some returns
    for _ in range(10):
        strategy.get_weight(risky_return=0.01)
    
    # Reset
    strategy.reset()
    
    # Estimator should be reset
    assert strategy.vol_estimator.k == 0


def test_feedback_controller_warmup():
    """Test feedback controller warmup period."""
    target_vol = 0.01
    strategy = FeedbackControllerStrategy(
        target_volatility=target_vol,
        warmup_days=10
    )
    
    # During warmup, should behave like open-loop
    returns = np.random.randn(15) * 0.02
    rf_returns = np.ones(15) * 0.0001
    
    for i, (r, rf) in enumerate(zip(returns, rf_returns)):
        weight = strategy.get_weight(risky_return=r, rf_return=rf)
        
        # Weight should be non-negative
        assert weight >= 0
        
        # Day count should increment
        assert strategy.day_count == i + 1


def test_feedback_controller_kappa_clipping():
    """Test that feedback controller clips kappa parameter."""
    target_vol = 0.01
    strategy = FeedbackControllerStrategy(
        target_volatility=target_vol,
        gain=100.0,  # High gain to test clipping
        kappa_min=-1.0,
        kappa_max=1.0,
        warmup_days=5
    )
    
    # Feed returns to get past warmup
    returns = np.random.randn(20) * 0.02
    rf_returns = np.ones(20) * 0.0001
    
    for r, rf in zip(returns, rf_returns):
        strategy.get_weight(risky_return=r, rf_return=rf)
    
    # Kappa should be within bounds
    assert -1.0 <= strategy.kappa <= 1.0


def test_feedback_controller_leverage_limit():
    """Test that feedback controller respects leverage limit."""
    target_vol = 0.01
    strategy = FeedbackControllerStrategy(
        target_volatility=target_vol,
        leverage_limit=1.5,
        warmup_days=5
    )
    
    returns = np.random.randn(50) * 0.02
    rf_returns = np.ones(50) * 0.0001
    
    for r, rf in zip(returns, rf_returns):
        weight = strategy.get_weight(risky_return=r, rf_return=rf)
        
        # Should respect leverage limit
        assert weight <= 1.5


def test_feedback_controller_reset():
    """Test feedback controller reset."""
    strategy = FeedbackControllerStrategy(target_volatility=0.01)
    
    # Feed some returns
    for _ in range(20):
        strategy.get_weight(risky_return=0.01, rf_return=0.0001)
    
    # Reset
    strategy.reset()
    
    # State should be reset
    assert strategy.kappa == 0.0
    assert strategy.day_count == 0
    assert strategy.prev_weight == 1.0
    assert strategy.risky_vol_estimator.k == 0
    assert strategy.index_vol_estimator.k == 0


def test_feedback_controller_smoothing():
    """Test feedback controller smoothing parameter."""
    target_vol = 0.01
    
    # No smoothing
    strategy_no_smooth = FeedbackControllerStrategy(
        target_volatility=target_vol,
        smoothing=0.0,
        warmup_days=5
    )
    
    # Full smoothing
    strategy_full_smooth = FeedbackControllerStrategy(
        target_volatility=target_vol,
        smoothing=0.9,
        warmup_days=5
    )
    
    # Feed same returns to both
    np.random.seed(42)
    returns = np.random.randn(30) * 0.02
    rf_returns = np.ones(30) * 0.0001
    
    for r, rf in zip(returns, rf_returns):
        strategy_no_smooth.get_weight(risky_return=r, rf_return=rf)
        strategy_full_smooth.get_weight(risky_return=r, rf_return=rf)
    
    # Kappa should be more stable with smoothing
    # (This is a qualitative test - in practice, smoothing reduces kappa volatility)
    assert strategy_no_smooth.kappa != strategy_full_smooth.kappa


def test_feedback_controller_gain_effect():
    """Test effect of gain parameter on controller."""
    target_vol = 0.01
    
    # Low gain
    strategy_low_gain = FeedbackControllerStrategy(
        target_volatility=target_vol,
        gain=1.0,
        warmup_days=5
    )
    
    # High gain
    strategy_high_gain = FeedbackControllerStrategy(
        target_volatility=target_vol,
        gain=100.0,
        warmup_days=5
    )
    
    # Feed same returns
    np.random.seed(42)
    returns = np.random.randn(30) * 0.02
    rf_returns = np.ones(30) * 0.0001
    
    for r, rf in zip(returns, rf_returns):
        strategy_low_gain.get_weight(risky_return=r, rf_return=rf)
        strategy_high_gain.get_weight(risky_return=r, rf_return=rf)
    
    # Different gains should produce different kappa values
    assert strategy_low_gain.kappa != strategy_high_gain.kappa


def test_feedback_controller_index_volatility():
    """Test that feedback controller tracks index volatility."""
    target_vol = 0.01
    strategy = FeedbackControllerStrategy(target_volatility=target_vol)
    
    returns = np.random.randn(50) * 0.02
    rf_returns = np.ones(50) * 0.0001
    
    for r, rf in zip(returns, rf_returns):
        strategy.get_weight(risky_return=r, rf_return=rf)
    
    # Should have index volatility estimate
    index_vol = strategy.get_index_volatility()
    assert index_vol >= 0


def test_strategy_weight_non_negative():
    """Test that all strategies return non-negative weights."""
    target_vol = 0.01
    
    strategies = [
        BuyAndHoldStrategy(),
        OpenLoopStrategy(target_volatility=target_vol),
        FeedbackControllerStrategy(target_volatility=target_vol)
    ]
    
    returns = np.random.randn(50) * 0.02
    rf_returns = np.ones(50) * 0.0001
    
    for strategy in strategies:
        for r, rf in zip(returns, rf_returns):
            if isinstance(strategy, BuyAndHoldStrategy):
                weight = strategy.get_weight()
            else:
                weight = strategy.get_weight(risky_return=r, rf_return=rf)
            
            assert weight >= 0, f"{strategy.name} returned negative weight"


def test_strategy_names():
    """Test that strategies have correct names."""
    assert BuyAndHoldStrategy().name == "Buy-and-Hold"
    assert OpenLoopStrategy(target_volatility=0.01).name == "Open-Loop"
    assert FeedbackControllerStrategy(target_volatility=0.01).name == "Feedback-Controller"
