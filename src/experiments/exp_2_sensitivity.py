"""
Experiment 2: Parameter Sensitivity Analysis

Grid search over controller parameters (gain g, smoothing θ)
Period: June 8, 2000 - December 31, 2009
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.data_loader import load_experiment_data
from src.strategies import BuyAndHoldStrategy, FeedbackControllerStrategy
from src.backtesting import Backtester, BacktestResult
from src.metrics import calculate_all_metrics, calculate_kalmar_ratio
from src.volatility import daily_target_volatility
from src.visualization import plot_parameter_heatmap


def run_experiment_2(
    ticker: str = 'IVV',
    start_date: str = '2000-06-08',
    end_date: str = '2009-12-31',
    target_vol_annual: float = 0.15,
    leverage_limit: float = 1.5,
    transaction_cost_bps: float = 5.0,
    half_life: int = 126,
    results_dir: str = 'results',
    api_key: str = None
) -> Dict:
    """
    Run Experiment 2: Parameter sensitivity analysis.
    
    Parameters
    ----------
    ticker : str
        ETF ticker (default: IVV)
    start_date : str
        Start date
    end_date : str
        End date
    target_vol_annual : float
        Annual target volatility
    leverage_limit : float
        Maximum leverage
    transaction_cost_bps : float
        Transaction cost in basis points
    half_life : int
        EWMA half-life
    results_dir : str
        Directory to save results
    api_key : str
        Massive API key
        
    Returns
    -------
    Dict
        Dictionary containing grid results
    """
    print("=" * 80)
    print("EXPERIMENT 2: Parameter Sensitivity Analysis")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print()
    
    # Load data
    print("Loading data...")
    prices, returns, rf_returns = load_experiment_data(ticker, start_date, end_date, api_key)
    print(f"Data loaded: {len(returns)} trading days")
    print()
    
    # Convert target volatility to daily
    target_vol_daily = daily_target_volatility(target_vol_annual)
    
    # Define parameter grid
    # Gain g: {0, e^0, e^0.5, e^1, ..., e^5} = 12 values
    g_values = np.array([0] + [np.exp(i * 0.5) for i in range(11)])
    
    # Smoothing θ: {0, 0.1, 0.2, ..., 0.9} = 10 values
    theta_values = np.arange(0, 1.0, 0.1)
    
    print(f"Parameter grid: {len(g_values)} x {len(theta_values)} = {len(g_values) * len(theta_values)} combinations")
    print(f"Gain values: {g_values}")
    print(f"Smoothing values: {theta_values}")
    print()
    
    # Calculate benchmark (IVV buy-and-hold) Kalmar ratio
    print("Calculating benchmark (IVV buy-and-hold)...")
    benchmark_strategy = BuyAndHoldStrategy()
    backtester = Backtester(transaction_cost_bps=transaction_cost_bps)
    benchmark_result = backtester.run(benchmark_strategy, returns, rf_returns)
    benchmark_kalmar = calculate_kalmar_ratio(benchmark_result.index_returns)
    print(f"Benchmark Kalmar Ratio: {benchmark_kalmar:.4f}")
    print()
    
    # Initialize result grids
    tracking_error_grid = np.zeros((len(g_values), len(theta_values)))
    kalmar_change_grid = np.zeros((len(g_values), len(theta_values)))
    turnover_grid = np.zeros((len(g_values), len(theta_values)))
    
    # Run grid search
    print("Running grid search...")
    total_runs = len(g_values) * len(theta_values)
    run_count = 0
    
    for i, g in enumerate(g_values):
        for j, theta in enumerate(theta_values):
            run_count += 1
            
            if run_count % 10 == 0:
                print(f"  Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")
            
            # Create strategy with current parameters
            strategy = FeedbackControllerStrategy(
                target_volatility=target_vol_daily,
                gain=g,
                smoothing=theta,
                leverage_limit=leverage_limit,
                half_life=half_life,
                warmup_days=10
            )
            
            # Run backtest
            result = backtester.run(strategy, returns, rf_returns)
            
            # Calculate metrics
            metrics = calculate_all_metrics(result, target_vol_daily)
            
            # Store results
            tracking_error_grid[i, j] = metrics['tracking_error'] if metrics['tracking_error'] is not None else np.nan
            
            kalmar = metrics['kalmar_ratio']
            kalmar_change_grid[i, j] = kalmar - benchmark_kalmar
            
            turnover = backtester.calculate_turnover(result)
            turnover_grid[i, j] = turnover
    
    print(f"  Progress: {total_runs}/{total_runs} (100.0%)")
    print("Grid search completed.")
    print()
    
    # Create visualizations
    print("Creating heatmaps...")
    os.makedirs(results_dir, exist_ok=True)
    
    # Heatmap 1: Tracking Error
    plot_parameter_heatmap(
        tracking_error_grid,
        g_values,
        theta_values,
        metric_name='Tracking Error',
        title='Volatility Tracking Error vs. Parameters',
        save_path=f"{results_dir}/exp2_heatmap_tracking_error.png",
        cmap='viridis'
    )
    print(f"  - Saved: {results_dir}/exp2_heatmap_tracking_error.png")
    
    # Heatmap 2: Change in Kalmar Ratio
    plot_parameter_heatmap(
        kalmar_change_grid,
        g_values,
        theta_values,
        metric_name='Δ Kalmar Ratio',
        title='Change in Kalmar Ratio vs. IVV',
        save_path=f"{results_dir}/exp2_heatmap_kalmar_change.png",
        cmap='RdYlGn'
    )
    print(f"  - Saved: {results_dir}/exp2_heatmap_kalmar_change.png")
    
    # Heatmap 3: Turnover
    plot_parameter_heatmap(
        turnover_grid,
        g_values,
        theta_values,
        metric_name='Turnover',
        title='Portfolio Turnover vs. Parameters',
        save_path=f"{results_dir}/exp2_heatmap_turnover.png",
        cmap='plasma'
    )
    print(f"  - Saved: {results_dir}/exp2_heatmap_turnover.png")
    
    print()
    
    # Summary statistics
    print("Summary Statistics:")
    print("=" * 80)
    print(f"Tracking Error - Min: {np.nanmin(tracking_error_grid):.4f}, Max: {np.nanmax(tracking_error_grid):.4f}, Mean: {np.nanmean(tracking_error_grid):.4f}")
    print(f"Kalmar Change  - Min: {np.nanmin(kalmar_change_grid):.4f}, Max: {np.nanmax(kalmar_change_grid):.4f}, Mean: {np.nanmean(kalmar_change_grid):.4f}")
    print(f"Turnover       - Min: {np.nanmin(turnover_grid):.4f}, Max: {np.nanmax(turnover_grid):.4f}, Mean: {np.nanmean(turnover_grid):.4f}")
    print()
    
    # Find optimal parameters (example: minimize tracking error)
    min_idx = np.unravel_index(np.nanargmin(tracking_error_grid), tracking_error_grid.shape)
    print(f"Minimum Tracking Error: g={g_values[min_idx[0]]:.2f}, θ={theta_values[min_idx[1]]:.2f}")
    print()
    
    print("Experiment 2 completed successfully!")
    print("=" * 80)
    
    return {
        'g_values': g_values,
        'theta_values': theta_values,
        'tracking_error_grid': tracking_error_grid,
        'kalmar_change_grid': kalmar_change_grid,
        'turnover_grid': turnover_grid,
        'benchmark_kalmar': benchmark_kalmar
    }


if __name__ == '__main__':
    run_experiment_2()
