import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional

def inspect_columns(df):
    result = pd.DataFrame({
        'unique': df.nunique() == len(df),
        'cardinality': df.nunique(),
        'with_null': df.isna().any(),
        'null_pct': round((df.isnull().sum() / len(df)) * 100, 2),
        '1st_row': df.iloc[0],
        'random_row': df.iloc[np.random.randint(low=0, high=len(df))],
        'last_row': df.iloc[-1],
        'dtype': df.dtypes
    })
    return result


def query_data(df, product, date=None):
    query_str = f'product == "{product}"'
    if date is not None:
        # Assuming you have a date column or can extract date from open_time
        # You may need to adjust this based on your actual date format
        query_str += f' & date.dt.date == "{date}"'
    
    return df.query(query_str).copy()


def time_series_plot(df, cols, symbol, date=None):
    # Filter data
    filtered_df = df[df['symbol'] == symbol].copy()
    if date is not None:
        # Assuming open_time is a datetime column
        filtered_df = filtered_df[filtered_df['open_time'].dt.date.astype(str) == date]
    
    # Set index to open_time for plotting
    plot_df = filtered_df.loc[:, ['open_time'] + cols].set_index('open_time')
    
    # Create plot
    return plot_df.plot(figsize=(20, 8), title=f"symbol={symbol}, date={date if date else 'all'}")


def calc_mid_price(df: pd.DataFrame) -> pd.Series:
    return (df['high'] + df['low']) / 2


def calc_typical_price(df: pd.DataFrame) -> pd.Series:
    return (df['high'] + df['low'] + df['close']) / 3


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def log_return(series: np.ndarray):
    return np.log(series).diff()


def log_return_df2(series: np.ndarray):
    return np.log(series).diff(2)


def flatten_name(prefix, src_names):
    ret = []
    for c in src_names:
        if c[0] in ['symbol', 'open_time']:
            ret.append(c[0])
        else:
            ret.append('.'.join([prefix] + list(c)))
    return ret


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    middle_band = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return pd.DataFrame({
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band
    })


def aggregate_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    # Ensure open_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
        df = df.copy()
        df['open_time'] = pd.to_datetime(df['open_time'])
    
    # Resample and aggregate
    resampled = df.resample(timeframe, on='open_time').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_asset_volume': 'sum',
        'number_of_trades': 'sum',
        'taker_buy_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum'
    })
    
    # Add symbol columns back if they exist
    if 'symbol' in df.columns:
        resampled['symbol'] = df['symbol'].iloc[0]
    if 'symbol_simple' in df.columns:
        resampled['symbol_simple'] = df['symbol_simple'].iloc[0]
    
    return resampled.reset_index()


def calculate_macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    })
