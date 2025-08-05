import pandas as pd
import numpy as np
from datetime import datetime

# Create mock data
n = 1000
timestamps = pd.date_range(datetime.now(), periods=n, freq="H")
price = np.cumsum(np.random.randn(n)) + 100

# Create sentiment that updates once per day
daily_sentiment = np.clip(np.random.randn(n // 24 + 1) * 0.5, -1, 1)
sentiment = np.repeat(daily_sentiment, 24)[:n]

# Create volume data - typically higher during "active" hours and with some randomness
# We'll simulate higher volume during 9am-5pm (assuming UTC)
hour_of_day = np.array([ts.hour for ts in timestamps])
is_active_hours = (hour_of_day >= 9) & (hour_of_day <= 17)
base_volume = np.random.lognormal(mean=3, sigma=0.5, size=n) * 100
volume_multiplier = np.where(is_active_hours, 2.5, 0.8)
volume = np.round(base_volume * volume_multiplier * (1 + sentiment * 0.2), 2)  # Volume slightly influenced by sentiment

df = pd.DataFrame({
    "name": "BTC",
    "timestamp": timestamps,
    "price": price,
    "sentiment": sentiment,
    "volume": volume
})

df.to_csv("mock_data.csv", index=False)