"""
Data loading module for ETF prices and risk-free rates.
Uses Massive API (Polygon.io wrapper) for financial data retrieval.
"""

import os
from typing import Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
from massive import RESTClient
import pandas_datareader as pdr


def get_etf_data(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch daily ETF price data from Massive API (Polygon.io).
    
    Parameters
    ----------
    ticker : str
        ETF ticker symbol (e.g., 'IVV')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    api_key : str, optional
        Massive API key. If None, reads from MASSIVE_API_KEY env variable
        
    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
        Close prices are adjusted for dividends and splits (total return series)
    """
    if api_key is None:
        api_key = os.getenv("MASSIVE_API_KEY")
        if api_key is None:
            raise ValueError("MASSIVE_API_KEY environment variable not set")
    
    client = RESTClient(api_key=api_key)
    
    # Fetch aggregates (daily bars)
    aggs = []
    for agg in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=start_date,
        to=end_date,
        limit=50000,
        adjusted=True  # Adjusted for splits and dividends
    ):
        aggs.append(agg)
    
    if not aggs:
        raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")
    
    # Convert to DataFrame
    data = []
    for agg in aggs:
        data.append({
            'date': pd.Timestamp(agg.timestamp, unit='ms'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    return df


def get_fed_funds_rate(start_date: str, end_date: str) -> pd.Series:
    """
    Fetch daily Federal Funds Effective Rate from FRED.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns
    -------
    pd.Series
        Series with DatetimeIndex and daily Fed Funds Rate (annualized, in decimal form)
    """
    try:
        # Fetch from FRED using pandas_datareader
        fed_funds = pdr.DataReader('DFF', 'fred', start_date, end_date)
        fed_funds = fed_funds['DFF'] / 100.0  # Convert from percentage to decimal
        fed_funds.index.name = 'date'
        return fed_funds
    except Exception as e:
        raise ValueError(f"Failed to fetch Fed Funds Rate: {str(e)}")


def align_data(
    price_data: pd.DataFrame,
    rf_rate: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align price data and risk-free rate to common trading days.
    Forward-fills risk-free rate for missing days (weekends/holidays).
    
    Parameters
    ----------
    price_data : pd.DataFrame
        ETF price data with DatetimeIndex
    rf_rate : pd.Series
        Risk-free rate with DatetimeIndex
        
    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Aligned price data and risk-free rate
    """
    # Get common date range
    common_dates = price_data.index
    
    # Reindex risk-free rate to match price data dates, forward-fill missing values
    rf_aligned = rf_rate.reindex(common_dates, method='ffill')
    
    # Drop any rows where either series has NaN
    valid_idx = ~(price_data['close'].isna() | rf_aligned.isna())
    
    return price_data[valid_idx], rf_aligned[valid_idx]


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from price series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
        
    Returns
    -------
    pd.Series
        Simple returns: r_k = P_k / P_{k-1} - 1
    """
    return prices.pct_change()


def convert_annual_rate_to_daily(annual_rate: pd.Series) -> pd.Series:
    """
    Convert annualized rate to daily rate.
    Uses 360-day convention (common for Fed Funds Rate).
    
    Parameters
    ----------
    annual_rate : pd.Series
        Annualized rate (decimal form)
        
    Returns
    -------
    pd.Series
        Daily rate
    """
    return annual_rate / 360.0


def load_experiment_data(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Load and prepare all data needed for experiments.
    
    Parameters
    ----------
    ticker : str
        ETF ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    api_key : str, optional
        Massive API key
        
    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        prices, returns, daily_rf_rate (all aligned to common dates)
    """
    # Load price data
    price_data = get_etf_data(ticker, start_date, end_date, api_key)
    
    # Load risk-free rate
    rf_rate = get_fed_funds_rate(start_date, end_date)
    
    # Align data
    price_data_aligned, rf_rate_aligned = align_data(price_data, rf_rate)
    
    # Extract close prices
    prices = price_data_aligned['close']
    
    # Calculate returns
    returns = calculate_returns(prices)
    
    # Convert risk-free rate to daily
    daily_rf = convert_annual_rate_to_daily(rf_rate_aligned)
    
    # Drop first return (NaN)
    returns = returns.iloc[1:]
    prices = prices.iloc[1:]
    daily_rf = daily_rf.iloc[1:]
    
    return prices, returns, daily_rf


# ETF cohort for Experiment 3 (44 ETFs as specified in paper)
ETF_COHORT = [
    'IVV', 'SPY', 'VOO', 'QQQ', 'VTI', 'IWM', 'EFA', 'VEA', 'IEFA', 'VWO',
    'EEM', 'IEMG', 'AGG', 'BND', 'LQD', 'HYG', 'TLT', 'IEF', 'SHY', 'GLD',
    'IAU', 'SLV', 'USO', 'DBC', 'VNQ', 'IYR', 'XLF', 'XLE', 'XLK', 'XLV',
    'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'VGT', 'VHT', 'VFH', 'VDE',
    'VIS', 'VCR', 'VDC', 'VAW'
]
