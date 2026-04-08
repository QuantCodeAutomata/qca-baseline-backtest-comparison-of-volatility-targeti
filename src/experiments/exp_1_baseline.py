"""
Experiment 1: Baseline Backtest Comparison on IVV (2000-2024)

Compares three strategies:
1. Buy-and-Hold (IVV)
2. Open-Loop Volatility Targeting
3. Feedback Controller

Period: June 8, 2000 - December 31, 2024
"""

import os
import numpy as np
import pandas as pd
from typing import Dict
from src.data_loader import load_experiment_data
from src.strategies import BuyAndHoldStrategy, OpenLoopStrategy, FeedbackControllerStrategy
from src.backtesting import Backtester, BacktestResult
from src.metrics import create_metrics_table, format_metrics_for_display
from src.volatility import daily_target_volatility
from src.visualization import (
    plot_cumulative_returns,
    plot_running_volatility,
    plot_weights,
    plot_drawdown
)


def run_experiment_1(
    ticker: str = 'IVV',
    start_date: str = '2000-06-08',
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
    Run Experiment 1: Baseline comparison.
    
    Parameters
    ----------
    ticker : str
        ETF ticker (default: IVV)
    start_date : str
        Start date
    end_date : str
        End date
    target_vol_annual : float
        Annual target volatility (default: 0.15 = 15%)
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
        Dictionary containing results and metrics
    """
    print("=" * 80)
    print("EXPERIMENT 1: Baseline Backtest Comparison")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Target Volatility: {target_vol_annual*100:.1f}% (annual)")
    print(f"Leverage Limit: {leverage_limit}")
    print(f"Transaction Cost: {transaction_cost_bps} bps")
    print()
    
    # Load data
    print("Loading data...")
    prices, returns, rf_returns = load_experiment_data(ticker, start_date, end_date, api_key)
    print(f"Data loaded: {len(returns)} trading days")
    print()
    
    # Convert target volatility to daily
    target_vol_daily = daily_target_volatility(target_vol_annual)
    
    # Initialize strategies
    print("Initializing strategies...")
    strategies = [
        BuyAndHoldStrategy(),
        OpenLoopStrategy(
            target_volatility=target_vol_daily,
            leverage_limit=leverage_limit,
            half_life=half_life
        ),
        FeedbackControllerStrategy(
            target_volatility=target_vol_daily,
            gain=controller_gain,
            smoothing=controller_smoothing,
            leverage_limit=leverage_limit,
            half_life=half_life,
            warmup_days=10
        )
    ]
    print()
    
    # Run backtests
    print("Running backtests...")
    backtester = Backtester(transaction_cost_bps=transaction_cost_bps)
    
    results = {}
    turnover_values = {}
    
    for strategy in strategies:
        print(f"  - {strategy.name}...")
        result = backtester.run(strategy, returns, rf_returns)
        results[strategy.name] = result
        
        # Calculate turnover
        turnover = backtester.calculate_turnover(result)
        turnover_values[strategy.name] = turnover
    
    print("Backtests completed.")
    print()
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics_df = create_metrics_table(
        results,
        target_volatility=target_vol_daily,
        turnover_values=turnover_values
    )
    
    # Format for display
    metrics_display = format_metrics_for_display(metrics_df)
    
    print("\nPerformance Metrics:")
    print("=" * 80)
    print(metrics_display.to_string(index=False))
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Cumulative returns
    plot_cumulative_returns(
        results,
        title=f"Cumulative Returns - {ticker} ({start_date} to {end_date})",
        save_path=f"{results_dir}/exp1_cumulative_returns.png"
    )
    print(f"  - Saved: {results_dir}/exp1_cumulative_returns.png")
    
    # Plot 2: Running volatility
    plot_running_volatility(
        results,
        target_volatility=target_vol_daily,
        title=f"Running Annualized Volatility - {ticker}",
        save_path=f"{results_dir}/exp1_running_volatility.png"
    )
    print(f"  - Saved: {results_dir}/exp1_running_volatility.png")
    
    # Plot 3: Portfolio weights
    plot_weights(
        results,
        title=f"Portfolio Weights - {ticker}",
        save_path=f"{results_dir}/exp1_weights.png"
    )
    print(f"  - Saved: {results_dir}/exp1_weights.png")
    
    # Plot 4: Drawdown
    plot_drawdown(
        results,
        title=f"Drawdown - {ticker}",
        save_path=f"{results_dir}/exp1_drawdown.png"
    )
    print(f"  - Saved: {results_dir}/exp1_drawdown.png")
    
    print()
    print("Experiment 1 completed successfully!")
    print("=" * 80)
    
    return {
        'results': results,
        'metrics': metrics_df,
        'metrics_display': metrics_display,
        'turnover': turnover_values
    }


if __name__ == '__main__':
    run_experiment_1()
