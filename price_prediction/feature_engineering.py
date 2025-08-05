import pandas as pd
import numpy as np

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def create_features(df, path=None):
    # Price features
    df["return_1"] = df["price"].pct_change()
    df["return_3"] = df["price"].pct_change(3)
    df["return_6"] = df["price"].pct_change(6)
    df["volatility"] = df["return_1"].rolling(window=12).std()
    df["future_return"] = df["price"].pct_change().shift(-1)
    df["target"] = (df["future_return"] > 0).astype(int)

    # Volume features
    df["volume_change"] = df["volume"].pct_change()
    df["volume_ma_6"] = df["volume"].rolling(window=6).mean()
    df["volume_ma_12"] = df["volume"].rolling(window=12).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_12"]

    # Technical indicators
    df["rsi_14"] = calculate_rsi(df["price"], 14)
    df["sma_6"] = df["price"].rolling(window=6).mean()
    df["sma_12"] = df["price"].rolling(window=12).mean()
    df["ema_6"] = df["price"].ewm(span=6, adjust=False).mean()
    df["ema_12"] = df["price"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Momentum features
    df["momentum_3"] = df["price"] / df["price"].shift(3) - 1
    df["momentum_6"] = df["price"] / df["price"].shift(6) - 1

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.weekday >= 5
    df["is_active_hours"] = (df["hour"] >= 9) & (df["hour"] <= 17)
    
    df = df.dropna()
    df.to_csv(path,index=False)
    return df

df = pd.read_csv("mock_data.csv", parse_dates=["timestamp"])
create_features(df,path="mock_features.csv")