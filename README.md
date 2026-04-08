# Baseline Backtest Comparison of Volatility Targeting Strategies

This repository implements a comprehensive quantitative finance experiment comparing volatility targeting strategies, including a novel feedback-based volatility controller against traditional open-loop methods.

## Overview

The project reproduces three key experiments from a research paper on volatility targeting:

1. **Experiment 1**: Baseline backtest comparison on IVV (2000-2024)
2. **Experiment 2**: Parameter sensitivity analysis via grid search
3. **Experiment 3**: Out-of-sample generalization test on 44 ETFs (2010-2024)

## Project Structure

```
.
├── src/
│   ├── data_loader.py          # Data fetching using Massive API
│   ├── volatility.py            # EWMA volatility estimation (custom)
│   ├── strategies.py            # Three strategies: Buy-Hold, Open-Loop, Feedback Controller
│   ├── backtesting.py           # Backtesting engine with transaction costs
│   ├── metrics.py               # Performance metrics calculation
│   ├── experiments/
│   │   ├── exp_1_baseline.py    # Experiment 1: Baseline comparison
│   │   ├── exp_2_sensitivity.py # Experiment 2: Parameter sensitivity
│   │   └── exp_3_generalization.py # Experiment 3: Multi-asset test
│   └── visualization.py         # Plotting utilities
├── tests/
│   ├── test_volatility.py
│   ├── test_strategies.py
│   ├── test_backtesting.py
│   └── test_metrics.py
├── results/
│   └── RESULTS.md               # Experiment results and metrics
├── data/                        # Cached data files
├── requirements.txt
└── run_experiments.py           # Main script to run all experiments

```

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Set the following environment variable:
```bash
export MASSIVE_API_KEY="your_massive_api_key"
```

## Usage

Run all experiments:
```bash
python run_experiments.py
```

Run tests:
```bash
pytest tests/
```

## Methodology

### Custom Implementations

All core algorithms are implemented from scratch following the paper's exact specifications:

- **EWMA Volatility Estimation**: Custom implementation with finite-sample correction
  ```
  (σ̂_k)^2 = ((1-β)/(1-β^k)) * Σ_{j=1 to k} β^(k-j) * r_j^2
  ```

- **Feedback Controller**: Novel closed-loop volatility targeting with log-error feedback
  ```
  e_k = log(σ̂_k^ind / σ_tar)
  κ_k = (1-θ) * clip(-g * e_k, [κ_min, κ_max]) + θ * κ_{k-1}
  w_k = min(exp(κ_k) * σ_tar / σ̂_k, L)
  ```

- **Open-Loop Rule**: Standard volatility targeting
  ```
  w_k = min(σ_tar / σ̂_k, L)
  ```

### Libraries Used

- **Data**: massive (Polygon.io API wrapper)
- **Computation**: numpy, pandas, scipy
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest
- **Performance Analytics**: quantstats (for standard metrics only)

## Results

All results, including performance metrics tables and plots, are saved in `results/RESULTS.md`.

## License

MIT License
