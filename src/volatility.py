"""
Volatility estimation using EWMA (Exponentially Weighted Moving Average).
Custom implementation with finite-sample correction as per paper methodology.

# Custom — Context7 found no library equivalent (paper Eq. 2.1-2.2)
"""

import numpy as np
import pandas as pd
from typing import Union


def calculate_ewma_decay_factor(half_life: int) -> float:
    """
    Calculate EWMA decay factor from half-life.
    
    Parameters
    ----------
    half_life : int
        Half-life in number of periods (e.g., 126 trading days)
        
    Returns
    -------
    float
        Decay factor β = exp(-log(2) / h)
    """
    return np.exp(-np.log(2) / half_life)


def ewma_variance(
    returns: Union[np.ndarray, pd.Series],
    half_life: int = 126,
    finite_sample_correction: bool = True
) -> np.ndarray:
    """
    Calculate EWMA variance with finite-sample correction.
    
    Formula (from paper):
    (σ̂_k)^2 = ((1-β)/(1-β^k)) * Σ_{j=1 to k} β^(k-j) * r_j^2
    
    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Return series
    half_life : int, default=126
        Half-life for EWMA (126 trading days = 6 months)
    finite_sample_correction : bool, default=True
        Apply finite-sample correction factor (1-β^k)
        
    Returns
    -------
    np.ndarray
        EWMA variance estimates at each time k
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    n = len(returns)
    beta = calculate_ewma_decay_factor(half_life)
    
    # Initialize variance array
    var = np.zeros(n)
    
    # Calculate EWMA variance for each time k
    for k in range(n):
        # Squared returns up to time k
        r_squared = returns[:k+1] ** 2
        
        # Exponential weights: β^(k-j) for j=0 to k
        weights = beta ** np.arange(k, -1, -1)
        
        # Weighted sum
        weighted_sum = np.sum(weights * r_squared)
        
        # Finite-sample correction
        if finite_sample_correction:
            correction = (1 - beta) / (1 - beta ** (k + 1))
        else:
            correction = (1 - beta)
        
        var[k] = correction * weighted_sum
    
    return var


def ewma_volatility(
    returns: Union[np.ndarray, pd.Series],
    half_life: int = 126,
    finite_sample_correction: bool = True,
    annualize: bool = False,
    trading_days_per_year: int = 252
) -> np.ndarray:
    """
    Calculate EWMA volatility (standard deviation).
    
    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Return series
    half_life : int, default=126
        Half-life for EWMA
    finite_sample_correction : bool, default=True
        Apply finite-sample correction
    annualize : bool, default=False
        If True, annualize the volatility
    trading_days_per_year : int, default=252
        Number of trading days per year for annualization
        
    Returns
    -------
    np.ndarray
        EWMA volatility estimates
    """
    var = ewma_variance(returns, half_life, finite_sample_correction)
    vol = np.sqrt(var)
    
    if annualize:
        vol = vol * np.sqrt(trading_days_per_year)
    
    return vol


class EWMAVolatilityEstimator:
    """
    Stateful EWMA volatility estimator for online/streaming calculations.
    Maintains state for efficient incremental updates.
    """
    
    def __init__(
        self,
        half_life: int = 126,
        finite_sample_correction: bool = True
    ):
        """
        Initialize EWMA estimator.
        
        Parameters
        ----------
        half_life : int
            Half-life for EWMA
        finite_sample_correction : bool
            Apply finite-sample correction
        """
        self.half_life = half_life
        self.beta = calculate_ewma_decay_factor(half_life)
        self.finite_sample_correction = finite_sample_correction
        
        # State variables
        self.k = 0  # Current time index
        self.weighted_sum = 0.0  # Weighted sum of squared returns
        self.variance = 0.0
        self.volatility = 0.0
    
    def update(self, return_value: float) -> tuple[float, float]:
        """
        Update estimator with new return and calculate variance/volatility.
        
        Parameters
        ----------
        return_value : float
            New return value
            
        Returns
        -------
        tuple[float, float]
            (variance, volatility) at current time
        """
        # Update weighted sum: multiply previous sum by beta and add new squared return
        self.weighted_sum = self.beta * self.weighted_sum + return_value ** 2
        
        # Increment time
        self.k += 1
        
        # Calculate variance with finite-sample correction
        if self.finite_sample_correction:
            correction = (1 - self.beta) / (1 - self.beta ** self.k)
        else:
            correction = (1 - self.beta)
        
        self.variance = correction * self.weighted_sum
        self.volatility = np.sqrt(self.variance)
        
        return self.variance, self.volatility
    
    def reset(self):
        """Reset estimator state."""
        self.k = 0
        self.weighted_sum = 0.0
        self.variance = 0.0
        self.volatility = 0.0


def annualize_volatility(
    daily_vol: Union[float, np.ndarray],
    trading_days_per_year: int = 252
) -> Union[float, np.ndarray]:
    """
    Annualize daily volatility.
    
    Parameters
    ----------
    daily_vol : float or np.ndarray
        Daily volatility
    trading_days_per_year : int, default=252
        Number of trading days per year
        
    Returns
    -------
    float or np.ndarray
        Annualized volatility
    """
    return daily_vol * np.sqrt(trading_days_per_year)


def daily_target_volatility(
    annual_target: float,
    trading_days_per_year: int = 252
) -> float:
    """
    Convert annual target volatility to daily.
    
    Parameters
    ----------
    annual_target : float
        Annual target volatility (e.g., 0.15 for 15%)
    trading_days_per_year : int, default=252
        Number of trading days per year
        
    Returns
    -------
    float
        Daily target volatility
    """
    return annual_target / np.sqrt(trading_days_per_year)
