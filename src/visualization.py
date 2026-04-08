"""
Visualization utilities for experiment results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
from src.backtesting import BacktestResult

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_cumulative_returns(
    results: Dict[str, BacktestResult],
    title: str = "Cumulative Returns",
    save_path: Optional[str] = None
):
    """
    Plot cumulative returns for multiple strategies.
    
    Parameters
    ----------
    results : Dict[str, BacktestResult]
        Dictionary mapping strategy names to backtest results
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    for strategy_name, result in results.items():
        plt.plot(result.dates, result.cumulative_returns, label=strategy_name, linewidth=2)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_running_volatility(
    results: Dict[str, BacktestResult],
    target_volatility: float,
    title: str = "Running Annualized Volatility",
    save_path: Optional[str] = None,
    trading_days_per_year: int = 252
):
    """
    Plot running annualized volatility for multiple strategies.
    
    Parameters
    ----------
    results : Dict[str, BacktestResult]
        Dictionary mapping strategy names to backtest results
    target_volatility : float
        Target volatility (daily)
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    trading_days_per_year : int
        Number of trading days per year
    """
    plt.figure(figsize=(12, 6))
    
    # Convert target to annualized
    target_annual = target_volatility * np.sqrt(trading_days_per_year)
    
    for strategy_name, result in results.items():
        if result.index_volatility is not None:
            # Annualize volatility
            annual_vol = result.index_volatility * np.sqrt(trading_days_per_year)
            plt.plot(result.dates, annual_vol, label=strategy_name, linewidth=2)
    
    # Plot target line
    plt.axhline(y=target_annual, color='red', linestyle='--', linewidth=2, label='Target')
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_weights(
    results: Dict[str, BacktestResult],
    title: str = "Portfolio Weights",
    save_path: Optional[str] = None
):
    """
    Plot portfolio weights over time.
    
    Parameters
    ----------
    results : Dict[str, BacktestResult]
        Dictionary mapping strategy names to backtest results
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    for strategy_name, result in results.items():
        plt.plot(result.dates, result.weights, label=strategy_name, linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Weight in Risky Asset', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_parameter_heatmap(
    grid_results: np.ndarray,
    g_values: np.ndarray,
    theta_values: np.ndarray,
    metric_name: str,
    title: str,
    save_path: Optional[str] = None,
    cmap: str = 'viridis'
):
    """
    Plot heatmap for parameter sensitivity analysis.
    
    Parameters
    ----------
    grid_results : np.ndarray
        2D array of metric values (shape: len(g_values) x len(theta_values))
    g_values : np.ndarray
        Array of gain values
    theta_values : np.ndarray
        Array of smoothing values
    metric_name : str
        Name of the metric
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    cmap : str
        Colormap name
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        grid_results,
        xticklabels=[f'{t:.1f}' for t in theta_values],
        yticklabels=[f'{g:.1f}' for g in g_values],
        cmap=cmap,
        annot=False,
        fmt='.3f',
        cbar_kws={'label': metric_name}
    )
    
    plt.xlabel('Smoothing (θ)', fontsize=12)
    plt.ylabel('Gain (g)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_distribution_comparison(
    open_loop_values: np.ndarray,
    controller_values: np.ndarray,
    metric_name: str,
    title: str,
    save_path: Optional[str] = None,
    bins: int = 20
):
    """
    Plot histogram comparison of metric distributions.
    
    Parameters
    ----------
    open_loop_values : np.ndarray
        Metric values for open-loop strategy
    controller_values : np.ndarray
        Metric values for controller strategy
    metric_name : str
        Name of the metric
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    bins : int
        Number of histogram bins
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(open_loop_values, bins=bins, alpha=0.6, label='Open-Loop', color='blue', edgecolor='black')
    plt.hist(controller_values, bins=bins, alpha=0.6, label='Feedback Controller', color='green', edgecolor='black')
    
    # Add vertical lines for means
    plt.axvline(np.mean(open_loop_values), color='blue', linestyle='--', linewidth=2, label=f'Open-Loop Mean: {np.mean(open_loop_values):.3f}')
    plt.axvline(np.mean(controller_values), color='green', linestyle='--', linewidth=2, label=f'Controller Mean: {np.mean(controller_values):.3f}')
    
    plt.xlabel(metric_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_drawdown(
    results: Dict[str, BacktestResult],
    title: str = "Drawdown",
    save_path: Optional[str] = None
):
    """
    Plot drawdown over time for multiple strategies.
    
    Parameters
    ----------
    results : Dict[str, BacktestResult]
        Dictionary mapping strategy names to backtest results
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    for strategy_name, result in results.items():
        cumulative = np.cumprod(1 + result.index_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        plt.plot(result.dates, drawdown * 100, label=strategy_name, linewidth=2)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
