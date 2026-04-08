"""
Performance metrics calculation for volatility targeting strategies.

Metrics include:
- Volatility tracking error
- Annualized return
- Annualized volatility
- Sharpe ratio
- Kalmar ratio (return / max drawdown)
- Maximum drawdown
- Turnover

# Custom — Context7 found no library equivalent (paper Table 1 metrics)
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.backtesting import BacktestResult


def calculate_annualized_return(
    returns: np.ndarray,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate annualized return from daily returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    trading_days_per_year : int, default=252
        Number of trading days per year
        
    Returns
    -------
    float
        Annualized return
    """
    total_return = np.prod(1 + returns) - 1
    n_days = len(returns)
    n_years = n_days / trading_days_per_year
    
    if n_years > 0:
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        annualized_return = 0.0
    
    return annualized_return


def calculate_annualized_volatility(
    returns: np.ndarray,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate annualized volatility from daily returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    trading_days_per_year : int, default=252
        Number of trading days per year
        
    Returns
    -------
    float
        Annualized volatility
    """
    daily_vol = np.std(returns, ddof=1)
    annualized_vol = daily_vol * np.sqrt(trading_days_per_year)
    return annualized_vol


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from daily returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
        
    Returns
    -------
    float
        Maximum drawdown (positive value)
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = np.abs(np.min(drawdown))
    return max_dd


def calculate_sharpe_ratio(
    returns: np.ndarray,
    rf_returns: np.ndarray,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Sharpe = (annual_return - mean_annual_rf_rate) / annual_volatility
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    rf_returns : np.ndarray
        Daily risk-free returns
    trading_days_per_year : int, default=252
        Number of trading days per year
        
    Returns
    -------
    float
        Sharpe ratio
    """
    annual_return = calculate_annualized_return(returns, trading_days_per_year)
    annual_vol = calculate_annualized_volatility(returns, trading_days_per_year)
    
    # Annualize risk-free rate
    mean_daily_rf = np.mean(rf_returns)
    mean_annual_rf = (1 + mean_daily_rf) ** trading_days_per_year - 1
    
    if annual_vol > 0:
        sharpe = (annual_return - mean_annual_rf) / annual_vol
    else:
        sharpe = 0.0
    
    return sharpe


def calculate_kalmar_ratio(
    returns: np.ndarray,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate Kalmar ratio (return / max drawdown).
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns
    trading_days_per_year : int, default=252
        Number of trading days per year
        
    Returns
    -------
    float
        Kalmar ratio
    """
    annual_return = calculate_annualized_return(returns, trading_days_per_year)
    max_dd = calculate_max_drawdown(returns)
    
    if max_dd > 0:
        kalmar = annual_return / max_dd
    else:
        kalmar = 0.0
    
    return kalmar


def calculate_volatility_tracking_error(
    index_volatility: np.ndarray,
    target_volatility: float,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate volatility tracking error (annualized).
    
    Tracking Error = sqrt(252) * mean(|σ̂_k^ind - σ_tar|)
    
    Parameters
    ----------
    index_volatility : np.ndarray
        Daily index volatility estimates
    target_volatility : float
        Target volatility (daily)
    trading_days_per_year : int, default=252
        Number of trading days per year
        
    Returns
    -------
    float
        Annualized tracking error
    """
    # Calculate absolute deviations
    deviations = np.abs(index_volatility - target_volatility)
    
    # Mean absolute deviation
    mean_deviation = np.mean(deviations)
    
    # Annualize
    annualized_tracking_error = mean_deviation * np.sqrt(trading_days_per_year)
    
    return annualized_tracking_error


def calculate_all_metrics(
    result: BacktestResult,
    target_volatility: Optional[float] = None,
    trading_days_per_year: int = 252
) -> dict:
    """
    Calculate all performance metrics for a backtest result.
    
    Parameters
    ----------
    result : BacktestResult
        Backtest result
    target_volatility : float, optional
        Target volatility (daily) for tracking error calculation
    trading_days_per_year : int, default=252
        Number of trading days per year
        
    Returns
    -------
    dict
        Dictionary of all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['annualized_return'] = calculate_annualized_return(
        result.index_returns, trading_days_per_year
    )
    metrics['annualized_volatility'] = calculate_annualized_volatility(
        result.index_returns, trading_days_per_year
    )
    metrics['max_drawdown'] = calculate_max_drawdown(result.index_returns)
    
    # Risk-adjusted metrics
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(
        result.index_returns, result.rf_returns, trading_days_per_year
    )
    metrics['kalmar_ratio'] = calculate_kalmar_ratio(
        result.index_returns, trading_days_per_year
    )
    
    # Tracking error (if index volatility available)
    if result.index_volatility is not None and target_volatility is not None:
        metrics['tracking_error'] = calculate_volatility_tracking_error(
            result.index_volatility, target_volatility, trading_days_per_year
        )
    else:
        metrics['tracking_error'] = None
    
    # Turnover
    if result.transaction_costs is not None:
        # Turnover can be approximated from transaction costs
        # Since cost = turnover * cost_rate, turnover = cost / cost_rate
        # But we'll calculate it properly in the backtester
        metrics['total_transaction_costs'] = np.sum(result.transaction_costs)
    else:
        metrics['total_transaction_costs'] = None
    
    return metrics


def create_metrics_table(
    results: dict[str, BacktestResult],
    target_volatility: Optional[float] = None,
    turnover_values: Optional[dict[str, float]] = None,
    trading_days_per_year: int = 252
) -> pd.DataFrame:
    """
    Create a formatted table of metrics for multiple strategies.
    
    Parameters
    ----------
    results : dict[str, BacktestResult]
        Dictionary mapping strategy names to backtest results
    target_volatility : float, optional
        Target volatility (daily)
    turnover_values : dict[str, float], optional
        Dictionary mapping strategy names to turnover values
    trading_days_per_year : int, default=252
        Number of trading days per year
        
    Returns
    -------
    pd.DataFrame
        Formatted metrics table
    """
    metrics_list = []
    
    for strategy_name, result in results.items():
        metrics = calculate_all_metrics(result, target_volatility, trading_days_per_year)
        metrics['strategy'] = strategy_name
        
        # Add turnover if provided
        if turnover_values is not None and strategy_name in turnover_values:
            metrics['turnover'] = turnover_values[strategy_name]
        
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    
    # Reorder columns
    column_order = ['strategy', 'tracking_error', 'annualized_return', 'annualized_volatility',
                    'sharpe_ratio', 'kalmar_ratio', 'max_drawdown', 'turnover']
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    return df


def format_metrics_for_display(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format metrics DataFrame for display (percentages, rounding).
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics DataFrame
        
    Returns
    -------
    pd.DataFrame
        Formatted DataFrame
    """
    df = metrics_df.copy()
    
    # Format as percentages
    percentage_cols = ['tracking_error', 'annualized_return', 'annualized_volatility', 
                       'max_drawdown', 'turnover']
    
    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col] * 100  # Convert to percentage
    
    return df
