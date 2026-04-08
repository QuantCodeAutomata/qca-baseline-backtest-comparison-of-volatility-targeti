"""
Experiment 3: Out-of-Sample Generalization Test

Tests controller on 44 ETFs over 2010-2024 period.
Compares Open-Loop vs Feedback Controller distributions.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List
from src.data_loader import load_experiment_data, ETF_COHORT
from src.strategies import BuyAndHoldStrategy, OpenLoopStrategy, FeedbackControllerStrategy
from src.backtesting import Backtester, BacktestResult
from src.metrics import calculate_all_metrics
from src.volatility import daily_target_volatility
from src.visualization import plot_distribution_comparison


def run_experiment_3(
    etf_list: List[str] = None,
    start_date: str = '2010-01-01',
    end_date: str = '2024-12-31',
    target_vol_annual: float = 0.15,
    leverage_limit: float = 1.5,
    transaction_cost_bps: float = 5.0,
    half_life: int = 126,
    controller_gain: float = 55.0,
    controller_smoothing: float = 0.6,
    results_dir: str = 'results',
    api_key: str = None
) -> Dict:
    """
    Run Experiment 3: Multi-asset generalization test.
    
    Parameters
    ----------
    etf_list : List[str], optional
        List of ETF tickers (default: ETF_COHORT)
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
    controller_gain : float
        Feedback controller gain
    controller_smoothing : float
        Feedback controller smoothing
    results_dir : str
        Directory to save results
    api_key : str
        Massive API key
        
    Returns
    -------
    Dict
        Dictionary containing cohort results
    """
    print("=" * 80)
    print("EXPERIMENT 3: Out-of-Sample Generalization Test")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date}")
    print()
    
    if etf_list is None:
        etf_list = ETF_COHORT
    
    print(f"ETF Cohort: {len(etf_list)} ETFs")
    print(f"ETFs: {', '.join(etf_list[:10])}...")
    print()
    
    # Convert target volatility to daily
    target_vol_daily = daily_target_volatility(target_vol_annual)
    
    # Initialize result storage
    open_loop_tracking_errors = []
    controller_tracking_errors = []
    
    open_loop_kalmar_changes = []
    controller_kalmar_changes = []
    
    open_loop_turnovers = []
    controller_turnovers = []
    
    successful_etfs = []
    failed_etfs = []
    
    # Process each ETF
    print("Processing ETFs...")
    backtester = Backtester(transaction_cost_bps=transaction_cost_bps)
    
    for idx, ticker in enumerate(etf_list):
        print(f"  [{idx+1}/{len(etf_list)}] {ticker}...", end=' ')
        
        try:
            # Load data
            prices, returns, rf_returns = load_experiment_data(ticker, start_date, end_date, api_key)
            
            # Check data quality
            if len(returns) < 100:
                print(f"SKIP (insufficient data: {len(returns)} days)")
                failed_etfs.append(ticker)
                continue
            
            # Run buy-and-hold benchmark
            benchmark_strategy = BuyAndHoldStrategy()
            benchmark_result = backtester.run(benchmark_strategy, returns, rf_returns)
            benchmark_metrics = calculate_all_metrics(benchmark_result, target_vol_daily)
            benchmark_kalmar = benchmark_metrics['kalmar_ratio']
            
            # Run open-loop strategy
            open_loop_strategy = OpenLoopStrategy(
                target_volatility=target_vol_daily,
                leverage_limit=leverage_limit,
                half_life=half_life
            )
            open_loop_result = backtester.run(open_loop_strategy, returns, rf_returns)
            open_loop_metrics = calculate_all_metrics(open_loop_result, target_vol_daily)
            open_loop_turnover = backtester.calculate_turnover(open_loop_result)
            
            # Run feedback controller strategy
            controller_strategy = FeedbackControllerStrategy(
                target_volatility=target_vol_daily,
                gain=controller_gain,
                smoothing=controller_smoothing,
                leverage_limit=leverage_limit,
                half_life=half_life,
                warmup_days=10
            )
            controller_result = backtester.run(controller_strategy, returns, rf_returns)
            controller_metrics = calculate_all_metrics(controller_result, target_vol_daily)
            controller_turnover = backtester.calculate_turnover(controller_result)
            
            # Store results
            if open_loop_metrics['tracking_error'] is not None:
                open_loop_tracking_errors.append(open_loop_metrics['tracking_error'])
            if controller_metrics['tracking_error'] is not None:
                controller_tracking_errors.append(controller_metrics['tracking_error'])
            
            open_loop_kalmar_changes.append(open_loop_metrics['kalmar_ratio'] - benchmark_kalmar)
            controller_kalmar_changes.append(controller_metrics['kalmar_ratio'] - benchmark_kalmar)
            
            open_loop_turnovers.append(open_loop_turnover)
            controller_turnovers.append(controller_turnover)
            
            successful_etfs.append(ticker)
            print("OK")
            
        except Exception as e:
            print(f"FAILED ({str(e)[:50]})")
            failed_etfs.append(ticker)
            continue
    
    print()
    print(f"Successfully processed: {len(successful_etfs)}/{len(etf_list)} ETFs")
    if failed_etfs:
        print(f"Failed ETFs: {', '.join(failed_etfs)}")
    print()
    
    # Convert to arrays
    open_loop_tracking_errors = np.array(open_loop_tracking_errors)
    controller_tracking_errors = np.array(controller_tracking_errors)
    open_loop_kalmar_changes = np.array(open_loop_kalmar_changes)
    controller_kalmar_changes = np.array(controller_kalmar_changes)
    open_loop_turnovers = np.array(open_loop_turnovers)
    controller_turnovers = np.array(controller_turnovers)
    
    # Summary statistics
    print("Summary Statistics:")
    print("=" * 80)
    print("\nTracking Error:")
    print(f"  Open-Loop:  Mean={np.mean(open_loop_tracking_errors):.4f}, Std={np.std(open_loop_tracking_errors):.4f}, Median={np.median(open_loop_tracking_errors):.4f}")
    print(f"  Controller: Mean={np.mean(controller_tracking_errors):.4f}, Std={np.std(controller_tracking_errors):.4f}, Median={np.median(controller_tracking_errors):.4f}")
    
    print("\nChange in Kalmar Ratio:")
    print(f"  Open-Loop:  Mean={np.mean(open_loop_kalmar_changes):.4f}, Std={np.std(open_loop_kalmar_changes):.4f}, Median={np.median(open_loop_kalmar_changes):.4f}")
    print(f"  Controller: Mean={np.mean(controller_kalmar_changes):.4f}, Std={np.std(controller_kalmar_changes):.4f}, Median={np.median(controller_kalmar_changes):.4f}")
    
    print("\nTurnover:")
    print(f"  Open-Loop:  Mean={np.mean(open_loop_turnovers):.2f}, Std={np.std(open_loop_turnovers):.2f}, Median={np.median(open_loop_turnovers):.2f}")
    print(f"  Controller: Mean={np.mean(controller_turnovers):.2f}, Std={np.std(controller_turnovers):.2f}, Median={np.median(controller_turnovers):.2f}")
    print()
    
    # Create visualizations
    print("Creating distribution plots...")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Tracking Error Distribution
    plot_distribution_comparison(
        open_loop_tracking_errors,
        controller_tracking_errors,
        metric_name='Tracking Error',
        title='Tracking Error Distribution Across ETF Cohort',
        save_path=f"{results_dir}/exp3_dist_tracking_error.png",
        bins=15
    )
    print(f"  - Saved: {results_dir}/exp3_dist_tracking_error.png")
    
    # Plot 2: Kalmar Change Distribution
    plot_distribution_comparison(
        open_loop_kalmar_changes,
        controller_kalmar_changes,
        metric_name='Change in Kalmar Ratio',
        title='Change in Kalmar Ratio Distribution',
        save_path=f"{results_dir}/exp3_dist_kalmar_change.png",
        bins=15
    )
    print(f"  - Saved: {results_dir}/exp3_dist_kalmar_change.png")
    
    # Plot 3: Turnover Distribution
    plot_distribution_comparison(
        open_loop_turnovers,
        controller_turnovers,
        metric_name='Turnover',
        title='Turnover Distribution',
        save_path=f"{results_dir}/exp3_dist_turnover.png",
        bins=15
    )
    print(f"  - Saved: {results_dir}/exp3_dist_turnover.png")
    
    print()
    print("Experiment 3 completed successfully!")
    print("=" * 80)
    
    return {
        'successful_etfs': successful_etfs,
        'failed_etfs': failed_etfs,
        'open_loop_tracking_errors': open_loop_tracking_errors,
        'controller_tracking_errors': controller_tracking_errors,
        'open_loop_kalmar_changes': open_loop_kalmar_changes,
        'controller_kalmar_changes': controller_kalmar_changes,
        'open_loop_turnovers': open_loop_turnovers,
        'controller_turnovers': controller_turnovers
    }


if __name__ == '__main__':
    run_experiment_3()
