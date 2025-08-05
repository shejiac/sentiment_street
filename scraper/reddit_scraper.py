import praw
import os
import time
import datetime
import psycopg2
from dotenv import load_dotenv
from prawcore.exceptions import ResponseException
import json
import requests  #NEW

load_dotenv()

# ==========================
# CONFIGURATION
# ==========================
SECONDS_IN_A_DAY = 86400
NOW_UTC = int(time.time())
LIMIT_POSTS = 100  # 

#NEW
COINS = {
    "Bitcoin": {"subreddit": "Bitcoin", "coingecko_id": "bitcoin"},
    "Ethereum": {"subreddit": "ethereum", "coingecko_id": "ethereum"},
    "Solana": {"subreddit": "solana", "coingecko_id": "solana"},
    "Cardano": {"subreddit": "cardano", "coingecko_id": "cardano"},
    "Dogecoin": {"subreddit": "dogecoin", "coingecko_id": "dogecoin"},
    "Avalanche": {"subreddit": "Avax", "coingecko_id": "avalanche-2"},
    "Polkadot": {"subreddit": "dot", "coingecko_id": "polkadot"},
    "Polygon": {"subreddit": "0xPolygon", "coingecko_id": "polygon"},
    "Litecoin": {"subreddit": "litecoin", "coingecko_id": "litecoin"},
    "Chainlink": {"subreddit": "Chainlink", "coingecko_id": "chainlink"},
    "Uniswap": {"subreddit": "Uniswap", "coingecko_id": "uniswap"},
    "Stellar": {"subreddit": "Stellar", "coingecko_id": "stellar"},
    "Cosmos": {"subreddit": "cosmosnetwork", "coingecko_id": "cosmos"},
    "VeChain": {"subreddit": "Vechain", "coingecko_id": "vechain"},
    "Monero": {"subreddit": "Monero", "coingecko_id": "monero"},
    "Aave": {"subreddit": "Aave_Official", "coingecko_id": "aave"},
    "Tezos": {"subreddit": "tezos", "coingecko_id": "tezos"},
    "Algorand": {"subreddit": "algorand", "coingecko_id": "algorand"},
    "NEAR Protocol": {"subreddit": "NearProtocol", "coingecko_id": "near"},
    "Fantom": {"subreddit": "FantomFoundation", "coingecko_id": "fantom"}
    }

#END_NEW

# ==========================
# Reddit API (ENV VARS)
# ==========================
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "sentiment_street/0.1 by u/the_user")

try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        check_for_async=False
    )
except ResponseException as e:
    print("Reddit API authentication failed:", e)
    exit(1)

# ==========================
# PostgreSQL Connection
# ==========================
conn = psycopg2.connect(
    dbname=os.getenv("PG_DB"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    sslmode="require",
)
cursor = conn.cursor()

# ==========================
# Create Tables if Needed
# ==========================
cursor.execute("""
CREATE TABLE IF NOT EXISTS reddit_posts (
    post_id TEXT PRIMARY KEY,
    subreddit TEXT,
    title TEXT,
    body TEXT,
    author TEXT,
    created_utc INTEGER,
    upvotes INTEGER,
    score INTEGER,
    num_comments INTEGER,
    flair TEXT,
    comments TEXT,
    upvote_ratio FLOAT,
    tickers TEXT
);
""")

#NEW
cursor.execute("""
ALTER TABLE reddit_posts ADD COLUMN IF NOT EXISTS coin_name TEXT;
""")
#END_NEW

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_reddit_posts_created_utc
ON reddit_posts (created_utc);
""")

#NEW
cursor.execute("""
CREATE TABLE IF NOT EXISTS coin_metrics (
    coin_name TEXT PRIMARY KEY,
    price_usd FLOAT,
    market_cap_usd FLOAT,
    volume_24h_usd FLOAT,
    fetched_at TIMESTAMP DEFAULT NOW()
);
""")
#END_NEW

#NEW
cursor.execute("""
DO $$
BEGIN
    BEGIN
        ALTER TABLE reddit_posts 
        ALTER COLUMN comments TYPE JSONB USING comments::JSONB,
        ALTER COLUMN flair TYPE JSONB USING flair::JSONB,
        ALTER COLUMN tickers TYPE JSONB USING tickers::JSONB;
    EXCEPTION WHEN others THEN
        -- Ignore if already JSONB or can't convert
        RAISE NOTICE 'Skipping column type conversion.';
    END;
END
$$;
""")
#END_NEW

conn.commit()

# ==========================
# Delete old data
# ==========================
DAYS_TO_KEEP = 50
cursor.execute("""
    DELETE FROM reddit_posts
    WHERE created_utc < EXTRACT(EPOCH FROM NOW()) - (%s * 86400);
""", (DAYS_TO_KEEP,))
conn.commit()
print(f"Deleted posts older than {DAYS_TO_KEEP} days.")

# ==========================
# CoinGecko Market Data
# ==========================
#NEW
def fetch_coin_metrics(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        response = requests.get(url)
        data = response.json()
        market_data = data.get("market_data", {})
        return {
            "price_usd": market_data.get("current_price", {}).get("usd", 0),
            "market_cap_usd": market_data.get("market_cap", {}).get("usd", 0),
            "volume_24h_usd": market_data.get("total_volume", {}).get("usd", 0)
        }
    except Exception as e:
        print(f"Failed to fetch metrics for {coin_id}: {e}")
        return None
#END_NEW

# ==========================
# Start Scraping
# ==========================
print(f"Reddit scraping started at {datetime.datetime.now()}")
inserted_posts = 0
inserted_comments = 0

try:  # Added this try block to properly wrap the scraping logic
    if reddit is None or conn is None:
        print("Cannot proceed with scraping due to previous connection errors.")
    else:
        for coin_name, info in COINS.items():
            subreddit_name = info["subreddit"]
            coingecko_id = info["coingecko_id"]
            
            print(f"\n--- Processing {coin_name} (Subreddit: r/{subreddit_name}) ---")

            # --- Fetch and store market data ---
            metrics = fetch_coin_metrics(coingecko_id)
            if metrics:
                try:
                    cursor.execute("""
                        INSERT INTO coin_metrics (coin_name, price_usd, market_cap_usd, volume_24h_usd)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (coin_name) DO UPDATE SET
                            price_usd = EXCLUDED.price_usd,
                            market_cap_usd = EXCLUDED.market_cap_usd,
                            volume_24h_usd = EXCLUDED.volume_24h_usd,
                            fetched_at = NOW();
                    """, (
                        coin_name, metrics["price_usd"], metrics["market_cap_usd"], metrics["volume_24h_usd"]
                    ))
                    conn.commit()
                    print(f"Stored latest market metrics for {coin_name}.")
                except Exception as e:
                    print(f"Error storing coin metrics for {coin_name}: {e}")
                    conn.rollback()

            try:
                subreddit = reddit.subreddit(subreddit_name)  # Added this line which was missing
                # We want posts from "today" (last 24 hours from NOW_UTC)
                current_utc = int(time.time())
                
                for post in subreddit.new(limit=LIMIT_POSTS):
                
            except ResponseException as reddit_error:
                print(f"Reddit API error for r/{subreddit_name}: {reddit_error.response.text}")
            except Exception as general_error:
                print(f"Error during scraping for r/{subreddit_name}: {general_error}")

except Exception as main_error:  # Catch any unexpected errors in the main scraping process
    print(f"An unexpected error occurred in the main scraping process: {main_error}")
finally:  # Now this is properly part of a try-finally block
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print(f"\nScraping finished at {datetime.datetime.now()}")
    print(f"Total inserted posts: {inserted_posts}")
    print(f"Total inserted comments: {inserted_comments}")