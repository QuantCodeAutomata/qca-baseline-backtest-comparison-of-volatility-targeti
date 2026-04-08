# Repository Context and Learnings

## Project Overview
This repository implements volatility targeting strategies for quantitative finance, comparing:
1. Buy-and-Hold strategy
2. Open-Loop volatility targeting
3. Feedback Controller (novel closed-loop approach)

## Key Implementation Details

### Custom Implementations
All core algorithms are implemented from scratch following paper specifications:

1. **EWMA Volatility Estimation** (`src/volatility.py`)
   - Finite-sample correction: `(1-β)/(1-β^k)`
   - Half-life parameter: 126 trading days
   - Both batch and online (stateful) implementations

2. **Feedback Controller** (`src/strategies.py`)
   - Log-volatility error feedback
   - Exponential smoothing of control parameter
   - Warmup period: 10 days in open-loop mode
   - Parameters: gain=55, smoothing=0.6

3. **Backtesting Engine** (`src/backtesting.py`)
   - Chronological simulation (no look-ahead bias)
   - Transaction costs: 5 bps spread
   - Weight drift calculation after returns

### Data Sources
- **ETF Prices**: Massive API (Polygon.io wrapper)
- **Risk-Free Rate**: FRED (Federal Funds Effective Rate)
- **Convention**: 360-day year for risk-free rate conversion

### Testing Strategy
- 63 comprehensive tests covering:
  - Edge cases (single day, zero returns, extreme values)
  - Mathematical properties (non-negativity, bounds)
  - Methodology adherence (finite-sample correction, no look-ahead)
  - State management (reset, consistency)

## Common Patterns

### EWMA Calculation
```python
beta = exp(-log(2) / half_life)
var_k = ((1-beta)/(1-beta^k)) * sum(beta^(k-j) * r_j^2)
```

### Controller Update
```python
e_k = log(sigma_ind / sigma_tar)
kappa_k = (1-theta) * clip(-g * e_k) + theta * kappa_{k-1}
w_k = min(exp(kappa_k) * sigma_tar / sigma_risky, L)
```

### Transaction Cost
```python
cost = |w_k - w_{k-1}^post| * 0.0005
where w_{k-1}^post = w_{k-1} * (1 + r_k) / (1 + r_k^ind)
```

## Performance Expectations

### Experiment 1 (IVV 2000-2024)
- Tracking Error: Controller << Open-Loop << Buy-Hold
- Sharpe Ratio: Controller > Open-Loop ≈ Buy-Hold
- Turnover: Controller >> Open-Loop >> Buy-Hold

### Experiment 2 (Parameter Sensitivity)
- Robust region: g ∈ [20, 100], θ ∈ [0.4, 0.8]
- Trade-off: Higher gain → lower tracking error, higher turnover

### Experiment 3 (Multi-Asset)
- Controller generalizes well across 44 ETFs
- Consistent tracking improvement vs open-loop

## Dependencies
- Core: numpy, pandas, scipy
- Visualization: matplotlib, seaborn
- Data: massive, pandas-datareader
- Testing: pytest

## Known Limitations
1. Requires MASSIVE_API_KEY environment variable
2. Data availability depends on Polygon.io coverage
3. Grid search in Exp 2 is computationally intensive (120 runs)
4. Exp 3 processes subset of ETFs by default (10 vs 44)
