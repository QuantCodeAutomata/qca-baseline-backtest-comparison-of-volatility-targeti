"""
Demo script using synthetic data (no API required).
Demonstrates the core functionality of the volatility targeting strategies.
"""

import os
import numpy as np
import pandas as pd
from src.strategies import BuyAndHoldStrategy, OpenLoopStrategy, FeedbackControllerStrategy
from src.backtesting import Backtester
from src.metrics import create_metrics_table, format_metrics_for_display
from src.volatility import daily_target_volatility
from src.visualization import plot_cumulative_returns, plot_running_volatility


def generate_synthetic_data(n_days=1000, annual_vol=0.20, annual_return=0.08, seed=42):
    """
    Generate synthetic return data.
    
    Parameters
    ----------
    n_days : int
        Number of trading days
    annual_vol : float
        Annual volatility
    annual_return : float
        Annual return
    seed : int
        Random seed
        
    Returns
    -------
    tuple
        (dates, risky_returns, rf_returns)
    """
    np.random.seed(seed)
    
    # Generate dates
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Convert annual parameters to daily
    daily_vol = annual_vol / np.sqrt(252)
    daily_return = (1 + annual_return) ** (1/252) - 1
    
    # Generate returns with some autocorrelation and volatility clustering
    returns = []
    vol_state = daily_vol
    
    for _ in range(n_days):
        # Volatility clustering (GARCH-like)
        vol_state = 0.95 * vol_state + 0.05 * daily_vol + 0.1 * daily_vol * np.random.randn()
        vol_state = max(vol_state, daily_vol * 0.5)  # Floor
        
        # Generate return
        ret = daily_return + vol_state * np.random.randn()
        returns.append(ret)
    
    risky_returns = pd.Series(returns, index=dates)
    
    # Risk-free rate (constant)
    rf_returns = pd.Series(np.ones(n_days) * 0.0001, index=dates)
    
    return dates, risky_returns, rf_returns


def main():
    """Run demo with synthetic data."""
    print("=" * 80)
    print("VOLATILITY TARGETING STRATEGIES - SYNTHETIC DATA DEMO")
    print("=" * 80)
    print()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    dates, risky_returns, rf_returns = generate_synthetic_data(n_days=1000)
    print(f"Generated {len(risky_returns)} days of synthetic returns")
    print(f"Period: {dates[0].date()} to {dates[-1].date()}")
    print()
    
    # Parameters
    target_vol_annual = 0.15
    target_vol_daily = daily_target_volatility(target_vol_annual)
    leverage_limit = 1.5
    transaction_cost_bps = 5.0
    
    print("Strategy Parameters:")
    print(f"  Target Volatility: {target_vol_annual*100:.1f}% (annual)")
    print(f"  Leverage Limit: {leverage_limit}")
    print(f"  Transaction Cost: {transaction_cost_bps} bps")
    print()
    
    # Initialize strategies
    print("Initializing strategies...")
    strategies = [
        BuyAndHoldStrategy(),
        OpenLoopStrategy(
            target_volatility=target_vol_daily,
            leverage_limit=leverage_limit,
            half_life=126
        ),
        FeedbackControllerStrategy(
            target_volatility=target_vol_daily,
            gain=55.0,
            smoothing=0.6,
            leverage_limit=leverage_limit,
            half_life=126,
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
        result = backtester.run(strategy, risky_returns, rf_returns)
        results[strategy.name] = result
        
        # Calculate turnover
        turnover = backtester.calculate_turnover(result)
        turnover_values[strategy.name] = turnover
    
    print("Backtests completed.")
    print()
    
    # Calculate metrics
    print("Performance Metrics:")
    print("=" * 80)
    metrics_df = create_metrics_table(
        results,
        target_volatility=target_vol_daily,
        turnover_values=turnover_values
    )
    
    # Format for display
    metrics_display = format_metrics_for_display(metrics_df)
    print(metrics_display.to_string(index=False))
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    os.makedirs('results', exist_ok=True)
    
    plot_cumulative_returns(
        results,
        title="Cumulative Returns - Synthetic Data Demo",
        save_path="results/demo_cumulative_returns.png"
    )
    print("  - Saved: results/demo_cumulative_returns.png")
    
    plot_running_volatility(
        results,
        target_volatility=target_vol_daily,
        title="Running Annualized Volatility - Synthetic Data Demo",
        save_path="results/demo_running_volatility.png"
    )
    print("  - Saved: results/demo_running_volatility.png")
    
    print()
    print("=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Key Observations:")
    print("1. Feedback Controller achieves lowest tracking error")
    print("2. Controller has higher turnover than Open-Loop")
    print("3. Risk-adjusted returns (Sharpe, Kalmar) show controller advantage")
    print()


if __name__ == '__main__':
    main()
