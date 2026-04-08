"""
Volatility targeting strategies implementation.

Three strategies:
1. Buy-and-Hold (Underlying Asset)
2. Open-Loop Volatility Targeting
3. Feedback Controller (Closed-Loop)

# Custom — Context7 found no library equivalent (paper Section 2)
"""

import numpy as np
from typing import Optional
from src.volatility import EWMAVolatilityEstimator


class BuyAndHoldStrategy:
    """
    Strategy A: Buy and hold the underlying risky asset.
    Weight in risky asset is always 1.
    """
    
    def __init__(self):
        """Initialize buy-and-hold strategy."""
        self.name = "Buy-and-Hold"
    
    def get_weight(self, *args, **kwargs) -> float:
        """
        Get portfolio weight for risky asset.
        
        Returns
        -------
        float
            Always returns 1.0 (100% in risky asset)
        """
        return 1.0
    
    def reset(self):
        """Reset strategy state (no-op for buy-and-hold)."""
        pass


class OpenLoopStrategy:
    """
    Strategy B: Open-loop volatility targeting.
    
    Formula:
    w_k = min(σ_tar / σ̂_k, L)
    
    where:
    - σ_tar: target volatility (daily)
    - σ̂_k: EWMA volatility estimate of risky asset
    - L: leverage limit
    """
    
    def __init__(
        self,
        target_volatility: float,
        leverage_limit: float = 1.5,
        half_life: int = 126
    ):
        """
        Initialize open-loop strategy.
        
        Parameters
        ----------
        target_volatility : float
            Target volatility (daily)
        leverage_limit : float, default=1.5
            Maximum leverage allowed
        half_life : int, default=126
            EWMA half-life in trading days
        """
        self.name = "Open-Loop"
        self.target_volatility = target_volatility
        self.leverage_limit = leverage_limit
        self.half_life = half_life
        
        # EWMA estimator for risky asset volatility
        self.vol_estimator = EWMAVolatilityEstimator(half_life=half_life)
    
    def get_weight(
        self,
        risky_return: float,
        **kwargs
    ) -> float:
        """
        Calculate portfolio weight for next period.
        
        Parameters
        ----------
        risky_return : float
            Most recent return of risky asset
            
        Returns
        -------
        float
            Portfolio weight for risky asset
        """
        # Update volatility estimate
        _, vol_estimate = self.vol_estimator.update(risky_return)
        
        # Avoid division by zero
        if vol_estimate < 1e-10:
            return 0.0
        
        # Calculate weight: w_k = min(σ_tar / σ̂_k, L)
        weight = self.target_volatility / vol_estimate
        weight = min(weight, self.leverage_limit)
        
        return weight
    
    def reset(self):
        """Reset strategy state."""
        self.vol_estimator.reset()


class FeedbackControllerStrategy:
    """
    Strategy C: Feedback-based volatility controller.
    
    Formulas:
    e_k = log(σ̂_k^ind / σ_tar)                                    [log-volatility error]
    κ_k = (1-θ) * clip(-g * e_k, [κ_min, κ_max]) + θ * κ_{k-1}  [control parameter update]
    w_k = min(exp(κ_k) * σ_tar / σ̂_k, L)                         [portfolio weight]
    
    where:
    - g: controller gain
    - θ: smoothing parameter
    - κ_min, κ_max: clip range for control parameter
    - σ̂_k^ind: EWMA volatility of portfolio (index) returns
    - σ̂_k: EWMA volatility of risky asset returns
    """
    
    def __init__(
        self,
        target_volatility: float,
        gain: float = 55.0,
        smoothing: float = 0.6,
        leverage_limit: float = 1.5,
        kappa_min: float = -1.0,
        kappa_max: float = 1.0,
        half_life: int = 126,
        warmup_days: int = 10
    ):
        """
        Initialize feedback controller strategy.
        
        Parameters
        ----------
        target_volatility : float
            Target volatility (daily)
        gain : float, default=55.0
            Controller gain parameter
        smoothing : float, default=0.6
            Smoothing parameter (0 = no smoothing, 1 = full smoothing)
        leverage_limit : float, default=1.5
            Maximum leverage allowed
        kappa_min : float, default=-1.0
            Minimum value for control parameter
        kappa_max : float, default=1.0
            Maximum value for control parameter
        half_life : int, default=126
            EWMA half-life in trading days
        warmup_days : int, default=10
            Number of days to run in open-loop mode before engaging controller
        """
        self.name = "Feedback-Controller"
        self.target_volatility = target_volatility
        self.gain = gain
        self.smoothing = smoothing
        self.leverage_limit = leverage_limit
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.half_life = half_life
        self.warmup_days = warmup_days
        
        # EWMA estimators
        self.risky_vol_estimator = EWMAVolatilityEstimator(half_life=half_life)
        self.index_vol_estimator = EWMAVolatilityEstimator(half_life=half_life)
        
        # Controller state
        self.kappa = 0.0  # Control parameter
        self.day_count = 0  # Day counter
        self.prev_weight = 1.0  # Previous weight (for calculating index return)
    
    def get_weight(
        self,
        risky_return: float,
        rf_return: float,
        **kwargs
    ) -> float:
        """
        Calculate portfolio weight for next period.
        
        Parameters
        ----------
        risky_return : float
            Most recent return of risky asset
        rf_return : float
            Most recent risk-free return
            
        Returns
        -------
        float
            Portfolio weight for risky asset
        """
        # Update risky asset volatility estimate
        _, risky_vol = self.risky_vol_estimator.update(risky_return)
        
        # Calculate index return using previous weight
        index_return = risky_return * self.prev_weight + rf_return * (1 - self.prev_weight)
        
        # Update index volatility estimate
        _, index_vol = self.index_vol_estimator.update(index_return)
        
        # Increment day counter
        self.day_count += 1
        
        # Warmup period: run in open-loop mode
        if self.day_count <= self.warmup_days:
            # Open-loop: w_k = min(σ_tar / σ̂_k, L)
            if risky_vol < 1e-10:
                weight = 0.0
            else:
                weight = self.target_volatility / risky_vol
                weight = min(weight, self.leverage_limit)
        else:
            # Feedback controller mode
            
            # Calculate log-volatility tracking error
            if index_vol < 1e-10 or self.target_volatility < 1e-10:
                error = 0.0
            else:
                error = np.log(index_vol / self.target_volatility)
            
            # Update control parameter with clipping and smoothing
            kappa_raw = -self.gain * error
            kappa_clipped = np.clip(kappa_raw, self.kappa_min, self.kappa_max)
            self.kappa = (1 - self.smoothing) * kappa_clipped + self.smoothing * self.kappa
            
            # Calculate weight: w_k = min(exp(κ_k) * σ_tar / σ̂_k, L)
            if risky_vol < 1e-10:
                weight = 0.0
            else:
                weight = np.exp(self.kappa) * self.target_volatility / risky_vol
                weight = min(weight, self.leverage_limit)
        
        # Store weight for next iteration
        self.prev_weight = weight
        
        return weight
    
    def reset(self):
        """Reset strategy state."""
        self.risky_vol_estimator.reset()
        self.index_vol_estimator.reset()
        self.kappa = 0.0
        self.day_count = 0
        self.prev_weight = 1.0
    
    def get_index_volatility(self) -> float:
        """
        Get current index volatility estimate.
        
        Returns
        -------
        float
            Current index volatility
        """
        return self.index_vol_estimator.volatility
