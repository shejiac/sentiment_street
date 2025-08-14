import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import altair as alt
from typing import List, Dict

# --------------------
# Configuration
# --------------------
st.set_page_config(layout="wide", page_title="Market-Wide Price & Sentiment Dashboard", page_icon="🧭")

BACKEND_API_BASE_URL = "http://localhost:8000/api"  # <-- Replace with real backend
# CoinGecko generic endpoint - for multiple coins you should construct ids dynamically
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# --------------------
# Utility helpers
# --------------------
def safe_get_json(url, params=None, raise_on_fail=False):
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        if raise_on_fail:
            raise
        st.warning(f"Network/backend request failed for {url}: {e}")
        return None

def interpret_correlation(coef: float) -> str:
    coef = abs(coef)
    if coef >= 0.9:
        return "Very Strong"
    if coef >= 0.7:
        return "Strong"
    if coef >= 0.5:
        return "Moderate"
    if coef >= 0.3:
        return "Weak"
    return "Negligible"

# --------------------
# Data fetching (cached)
# --------------------
@st.cache_data(ttl=60)
def fetch_market_overview(selected_ids: List[str] = None):
    """
    Returns market-wide data for selected coins (list of coin ids expected by backend),
    if selected_ids is None, fetch top coins from CoinGecko as fallback.
    Expected return: list of coin dicts with keys:
      id, symbol, name, current_price, price_change_percentage_24h,
      market_cap, total_volume, circulating_supply, sentiment_score, sentiment_change_24h
    """
    try:
        # Try backend first
        resp = safe_get_json(f"{BACKEND_API_BASE_URL}/market/overview")
        if resp:
            df = pd.DataFrame(resp)
            return df
    except Exception:
        pass

    # Fallback: query CoinGecko top n coins and fabricate sentiment columns
    coins_url = f"{COINGECKO_BASE}/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 50, "page": 1}
    data = safe_get_json(coins_url, params=params) or []
    rows = []
    for item in data:
        rows.append({
            "id": item.get("id"),
            "symbol": item.get("symbol").upper(),
            "name": item.get("name"),
            "current_price": item.get("current_price", 0.0),
            "price_change_percentage_24h": item.get("price_change_percentage_24h", 0.0),
            "market_cap": item.get("market_cap", 0.0),
            "total_volume": item.get("total_volume", 0.0),
            "circulating_supply": item.get("circulating_supply", np.nan),
            # mock sentiment fields
            "sentiment_score": np.random.uniform(-1, 1),
            "sentiment_change_24h": np.random.uniform(-0.5, 0.5),
        })
    df = pd.DataFrame(rows)
    return df

@st.cache_data(ttl=60)
def fetch_market_sentiment_topics():
    """
    Expected return format:
    {
      "top_topics": [{"topic": str, "score": float, "rank": int}, ...],
      "top_posts": [{"title":..., "url":..., "score":..., "id":...}, ...],
      "top_comments": [{"body":..., "url":..., "score":..., "id":...}, ...],
      "market_sentiment_score": float,
      "market_sentiment_change_24h": float
    }
    """
    try:
        resp = safe_get_json(f"{BACKEND_API_BASE_URL}/market/topics")
        if resp:
            return resp
    except Exception:
        pass

    # Mock fallback
    return {
        "top_topics": [
            {"topic": "ETF approval", "score": 0.82, "rank": 1},
            {"topic": "Layer-2 adoption", "score": 0.63, "rank": 2},
            {"topic": "Regulatory risk", "score": 0.45, "rank": 3},
        ],
        "top_posts": [
            {"title": "Why ETF changes everything", "url": "#", "score": 1240, "id": "p1"},
            {"title": "New L2 launches today", "url": "#", "score": 980, "id": "p2"},
            {"title": "Regulation update — what to watch", "url": "#", "score": 760, "id": "p3"},
        ],
        "top_comments": [
            {"body": "This is the right move", "url": "#", "score": 400, "id": "c1"},
            {"body": "Not sure I'd buy now", "url": "#", "score": 360, "id": "c2"},
            {"body": "Gamechanger", "url": "#", "score": 310, "id": "c3"},
        ],
        "market_sentiment_score": np.random.uniform(-0.2, 0.8),
        "market_sentiment_change_24h": np.random.uniform(-0.2, 0.2)
    }

@st.cache_data(ttl=60)
def fetch_time_series(period="7D", coin_id: str = None):
    """
    Returns a DataFrame with timestamp, price, sentiment_score for either market-wide or specific coin.
    period: '7D' or '30D' (function uses D days)
    """
    # Try backend
    try:
        url = f"{BACKEND_API_BASE_URL}/sentiment/time-series"
        params = {"period": period, "coin_id": coin_id} if coin_id else {"period": period}
        resp = safe_get_json(url, params=params)
        if resp:
            df = pd.DataFrame(resp)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except Exception:
        pass

    # Fallback mock
    days = 7 if period == "7D" else 30
    timestamps = pd.date_range(end=datetime.datetime.now(), periods=days, freq='D')
    price = np.cumsum(np.random.normal(0, 1, days)) + 1000 * (1 + np.random.uniform(-0.1, 0.1))
    sentiment = np.cumsum(np.random.normal(0, 0.05, days)) + np.random.uniform(-0.2, 0.6)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": price,
        "sentiment_score": sentiment
    })
    return df

@st.cache_data(ttl=60)
def fetch_correlation(period="7D", coin_id: str = None):
    """
    Attempt to fetch correlation coefficient from backend or compute from time-series if backend unavailable.
    Returns {"coefficient": float, "interpretation": str}
    """
    try:
        url = f"{BACKEND_API_BASE_URL}/sentiment/correlation"
        params = {"period": period, "coin_id": coin_id} if coin_id else {"period": period}
        resp = safe_get_json(url, params=params)
        if resp and "coefficient" in resp:
            coef = float(resp.get("coefficient", 0.0))
            return {"coefficient": coef, "interpretation": interpret_correlation(coef)}
    except Exception:
        pass

    # Compute locally from mock time-series
    df = fetch_time_series(period, coin_id)
    if "price" in df and "sentiment_score" in df and len(df) > 1:
        coef = df["price"].corr(df["sentiment_score"]) or 0.0
    else:
        coef = 0.0
    return {"coefficient": float(np.nan_to_num(coef)), "interpretation": interpret_correlation(float(np.nan_to_num(coef)))}

@st.cache_data(ttl=300)
def fetch_coin_discussions(coin_id: str):
    """
    coin_id: expected slug like 'bitcoin'
    Returns dict with top_topics, top_posts, top_comments
    """
    try:
        resp = safe_get_json(f"{BACKEND_API_BASE_URL}/coins/{coin_id}/discussions")
        if resp:
            return resp
    except Exception:
        pass

    # fallback mock
    return {
        "top_topics": [
            {"topic": f"{coin_id} scaling", "score": 0.7, "rank": 1},
            {"topic": f"{coin_id} roadmap", "score": 0.5, "rank": 2},
            {"topic": f"{coin_id} whales", "score": 0.3, "rank": 3},
        ],
        "top_posts": [
            {"title": f"{coin_id} huge update", "url": "#", "score": 480},
            {"title": f"Why {coin_id} may pump", "url": "#", "score": 320},
            {"title": f"{coin_id} news summary", "url": "#", "score": 210},
        ],
        "top_comments": [
            {"body": "Love the devs", "url": "#", "score": 120},
            {"body": "Not convinced", "url": "#", "score": 95},
            {"body": "Hodl", "url": "#", "score": 88},
        ]
    }

@st.cache_data(ttl=60)
def fetch_topics_list():
    """
    Returns topic list for Section 3. Expected format: list of dicts with name, sentiment_score, sentiment_change_24h, links_to_top_discussions
    """
    try:
        resp = safe_get_json(f"{BACKEND_API_BASE_URL}/topics")
        if resp:
            return pd.DataFrame(resp)
    except Exception:
        pass

    # Mock fallback
    rows = [
        {"name": "ETF Approval", "sentiment_score": 0.78, "sentiment_change_24h": 0.12, "top_discussions": [{"title":"ETF thread","url":"#"}]},
        {"name": "Regulation", "sentiment_score": 0.10, "sentiment_change_24h": -0.08, "top_discussions": [{"title":"Reg thread","url":"#"}]},
        {"name": "Layer2", "sentiment_score": 0.52, "sentiment_change_24h": 0.30, "top_discussions": [{"title":"L2 thread","url":"#"}]},
        {"name": "Whales", "sentiment_score": -0.12, "sentiment_change_24h": -0.02, "top_discussions": [{"title":"Whale watch","url":"#"}]},
    ]
    return pd.DataFrame(rows)

# --------------------
# UI - Top: Market Summary Panel
# --------------------
st.title("Market Summary — Price & Sentiment Dashboard")
st.markdown("Overview across selected coins: price movement, sentiment trends, top discussions, and correlations.")

# Fetch market overview data
market_df = fetch_market_overview()
market_topics = fetch_market_sentiment_topics()

# Basic market metrics
market_sentiment = market_topics.get("market_sentiment_score", 0.0)
market_sentiment_change = market_topics.get("market_sentiment_change_24h", 0.0)

# Compute market-wide price change: if backend provides, use it, else average selected coins
if "price_change_percentage_24h" in market_df.columns:
    market_price_change = market_df["price_change_percentage_24h"].mean()
else:
    market_price_change = 0.0

col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 2])

with col_a:
    st.metric("Market-wide Sentiment", f"{market_sentiment:.2f}", delta=f"{market_sentiment_change:+.2f}")

with col_b:
    st.metric("Market-wide Price Change (avg. %)", f"{market_price_change:+.2f}%", delta=f"{market_price_change:+.2f}%")

with col_c:
    # Rise or drop today for both
    sentiment_trend = "Rise" if market_sentiment_change >= 0 else "Drop"
    price_trend = "Rise" if market_price_change >= 0 else "Drop"
    st.info(f"Sentiment: **{sentiment_trend}** today — Price: **{price_trend}** today")

with col_d:
    st.write("Top Market Topics")
    for t in market_topics.get("top_topics", [])[:3]:
        st.write(f"- {t.get('topic')} (score {t.get('score'):.2f})")

st.markdown("---")

# Top 3 posts and comments (market-wide)
st.subheader("Top Market-wide Discussions & Comments")
posts = market_topics.get("top_posts", [])[:3]
comments = market_topics.get("top_comments", [])[:3]

c1, c2 = st.columns(2)
with c1:
    st.write("Top 3 Posts")
    for p in posts:
        st.markdown(f"• [{p.get('title')}]({p.get('url', '#')}) — score {p.get('score')}")
with c2:
    st.write("Top 3 Comments")
    for cm in comments:
        st.markdown(f"• [{(cm.get('body')[:80]+'...') if len(cm.get('body',''))>80 else cm.get('body')}]({cm.get('url','#')}) — score {cm.get('score')}")

st.markdown("---")

# Rank coins by sentiment, sentiment change, and % price change
st.subheader("Coins: sentiment & price leaders / laggards")
if not market_df.empty:
    df_display = market_df.copy()
    # create label for sentiment
    def sentiment_label(x):
        if x >= 0.6:
            return "Very Positive"
        if x >= 0.2:
            return "Positive"
        if x >= -0.2:
            return "Neutral"
        if x >= -0.6:
            return "Negative"
        return "Very Negative"
    df_display["sentiment_label"] = df_display["sentiment_score"].apply(sentiment_label)
    # top 3 highest sentiment
    top3_sent = df_display.nlargest(3, "sentiment_score")[["name", "symbol", "sentiment_score"]]
    low3_sent = df_display.nsmallest(3, "sentiment_score")[["name", "symbol", "sentiment_score"]]
    top3_sent_change = df_display.nlargest(3, "sentiment_change_24h")[["name", "symbol", "sentiment_change_24h"]]
    top3_pct_price = df_display.nlargest(3, "price_change_percentage_24h")[["name", "symbol", "price_change_percentage_24h"]]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write("Top 3 ↑ Sentiment")
        st.table(top3_sent.rename(columns={"name":"Coin","symbol":"Sym","sentiment_score":"Score"}))
    with c2:
        st.write("Top 3 ↓ Sentiment")
        st.table(low3_sent.rename(columns={"name":"Coin","symbol":"Sym","sentiment_score":"Score"}))
    with c3:
        st.write("Top 3 ↑ Sentiment Change (24h)")
        st.table(top3_sent_change.rename(columns={"name":"Coin","symbol":"Sym","sentiment_change_24h":"Δ Sentiment (24h)"}))
    with c4:
        st.write("Top 3 ↑ % Price Change (24h)")
        st.table(top3_pct_price.rename(columns={"name":"Coin","symbol":"Sym","price_change_percentage_24h":"% Δ (24h)"}))

    # Price-sentiment divergence (one increasing other decreasing)
    diverging = df_display[
        ((df_display["price_change_percentage_24h"] > 0) & (df_display["sentiment_change_24h"] < 0)) |
        ((df_display["price_change_percentage_24h"] < 0) & (df_display["sentiment_change_24h"] > 0))
    ][["name","symbol","price_change_percentage_24h","sentiment_change_24h"]]
    st.write("Coins with Price ↔ Sentiment Divergence (price and sentiment moving opposite directions):")
    if diverging.empty:
        st.write("None detected in the selected dataset.")
    else:
        st.table(diverging.rename(columns={"name":"Coin","symbol":"Sym","price_change_percentage_24h":"% Δ Price (24h)","sentiment_change_24h":"Δ Sentiment (24h)"}))

else:
    st.info("No market coin data available. Check backend or CoinGecko connectivity.")

st.markdown("---")

# --------------------
# Market-wide Price-Sentiment Correlation Graph
# --------------------
st.subheader("Market-wide Price-Sentiment Correlation")
corr_period = st.selectbox("Correlation period", options=["7D", "30D"], index=0, key="market_corr_period")

ts = fetch_time_series(corr_period, coin_id=None)
corr_info = fetch_correlation(corr_period, coin_id=None)

if not ts.empty:
    base = alt.Chart(ts).encode(x='timestamp:T')
    price_line = base.mark_line().encode(y=alt.Y('price', title='Price', scale=alt.Scale(zero=False)))
    sentiment_line = base.mark_line(strokeDash=[4,2]).encode(y=alt.Y('sentiment_score', title='Sentiment Score'), color=alt.value("orange"))

    layered = alt.layer(price_line, sentiment_line).resolve_scale(y='independent').properties(
        height=300, title=f"Market Price vs Sentiment ({corr_period}) — Corr: {corr_info['coefficient']:.2f} ({corr_info['interpretation']})"
    ).interactive()
    st.altair_chart(layered, use_container_width=True)
else:
    st.warning("No time-series data available for market-wide correlation.")

st.markdown("---")

# --------------------
# Section 2: Per-coin Detailed Chart & Table (scrollable)
# --------------------
st.header("Coins Table — Detailed & Sortable (click/select to inspect a coin)")
st.caption("Table defaults to sort by market cap. Click a coin in the dropdown to see per-coin features.")

# Prepare table for display similar to CoinMarketCap columns:
selected_coin_name = None
if not market_df.empty:
    display_cols = ["name","symbol","current_price","price_change_percentage_24h",
                    "sentiment_label","sentiment_change_24h","market_cap","total_volume","circulating_supply"]
    df_table = market_df.copy()
    df_table["sentiment_label"] = df_table["sentiment_score"].apply(lambda x: "Pos" if x>0.2 else ("Neg" if x<-0.2 else "Neutral"))
    df_table = df_table.rename(columns={
        "name":"Name","symbol":"Symbol","current_price":"Price (USD)",
        "price_change_percentage_24h":"% Δ (24h)","sentiment_label":"Sentiment",
        "sentiment_change_24h":"Δ Sent (24h)","market_cap":"Market Cap",
        "total_volume":"Volume (24h)","circulating_supply":"Circulating Supply"
    })
    # Sorting control
    sort_by = st.selectbox("Sort table by", options=list(df_table.columns), index=list(df_table.columns).index("Market Cap") if "Market Cap" in df_table.columns else 0)
    ascending = st.checkbox("Ascending", value=False)
    df_table_sorted = df_table.sort_values(by=sort_by, ascending=ascending)

    # Search / select coin
    coin_names = df_table_sorted["Name"].tolist()
    selected_coin_name = st.selectbox("Search / Select a coin to inspect", options=[""] + coin_names, index=0)
    st.dataframe(df_table_sorted.style.format({
        "Price (USD)": "${:,.2f}",
        "% Δ (24h)": "{:+.2f}%",
        "Market Cap": "${:,.0f}",
        "Volume (24h)": "${:,.0f}",
        "Circulating Supply": "{:,.0f}"
    }), use_container_width=True, height=300)
else:
    st.info("No coin table available.")

# If a coin is selected show detailed panel
if selected_coin_name:
    coin_row = market_df[market_df["name"] == selected_coin_name].iloc[0].to_dict()
    coin_id = coin_row.get("id")
    st.markdown(f"## {coin_row.get('name')} ({coin_row.get('symbol')}) — Details")

    # price & sentiment metric
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Price (USD)", f"${coin_row.get('current_price',0):,.2f}", delta=f"{coin_row.get('price_change_percentage_24h',0):+.2f}%")
    with c2:
        st.metric("Sentiment Score", f"{coin_row.get('sentiment_score',0):.2f}", delta=f"{coin_row.get('sentiment_change_24h',0):+.2f}")
    with c3:
        st.metric("Market Cap", f"${coin_row.get('market_cap',0):,.0f}")

    # Per-coin correlation & time series toggles
    st.subheader("Price vs Sentiment (Per-coin)")
    coin_corr_period = st.radio("Period", options=["7D","30D"], index=0, horizontal=True, key=f"coin_corr_{coin_id}")
    coin_ts = fetch_time_series(coin_corr_period, coin_id=coin_id)
    coin_corr = fetch_correlation(coin_corr_period, coin_id=coin_id)
    if not coin_ts.empty:
        base = alt.Chart(coin_ts).encode(x='timestamp:T')
        p_line = base.mark_line().encode(y=alt.Y('price', title='Price'))
        s_line = base.mark_line(strokeDash=[4,2]).encode(y=alt.Y('sentiment_score', title='Sentiment Score'), color=alt.value("orange"))
        layered = alt.layer(p_line, s_line).resolve_scale(y='independent').properties(
            height=300, title=f"{coin_row.get('name')} — Corr: {coin_corr['coefficient']:.2f} ({coin_corr['interpretation']})"
        ).interactive()
        st.altair_chart(layered, use_container_width=True)
    else:
        st.warning("No time-series data for this coin.")

    # Coin-specific topics / posts / comments
    discussions = fetch_coin_discussions(coin_id)
    st.subheader("Top Coin-specific Topics & Discussions")
    t1, t2 = st.columns(2)
    with t1:
        st.write("Top Topics")
        for topic in discussions.get("top_topics", [])[:3]:
            st.write(f"- {topic.get('topic')} (score {topic.get('score'):.2f})")
    with t2:
        st.write("Top Posts")
        for p in discussions.get("top_posts", [])[:3]:
            st.markdown(f"- [{p.get('title')}]({p.get('url', '#')}) — score {p.get('score')}")
    st.write("Top Comments")
    for c in discussions.get("top_comments", [])[:3]:
        st.markdown(f"- [{(c.get('body')[:90] + '...') if len(c.get('body',''))>90 else c.get('body')}]({c.get('url','#')}) — score {c.get('score')}")

    # Optional: today/this week's forecast placeholder (backend would supply model forecast)
    st.subheader("Short-term Forecast (optional)")
    try:
        forecast = safe_get_json(f"{BACKEND_API_BASE_URL}/coins/{coin_id}/forecast") or {}
    except:
        forecast = {}
    if forecast:
        st.write(forecast)
    else:
        st.info("No forecast available (backend). Placeholder: model output would appear here.")

st.markdown("---")

# --------------------
# Section 3: Topic Chart (scrollable)
# --------------------
st.header("Topics — sentiment & change (click to view discussions)")
topics_df = fetch_topics_list()

if not topics_df.empty:
    topics_df_sorted = topics_df.sort_values(by="sentiment_change_24h", ascending=False).reset_index(drop=True)
    # Display table
    def _format_row(r):
        return {
            "Topic": r["name"],
            "Sentiment": f"{r['sentiment_score']:.2f}",
            "Δ Sent (24h)": f"{r['sentiment_change_24h']:+.2f}",
            "Top Discussion": r.get("top_discussions",[{}])[0].get("title","#")
        }
    topic_display = pd.DataFrame([_format_row(r) for _, r in topics_df_sorted.iterrows()])
    st.dataframe(topic_display, use_container_width=True, height=300)

    st.write("Click a topic below to open its top discussions:")
    for _, row in topics_df_sorted.iterrows():
        st.markdown(f"**{row['name']}** — Sentiment {row['sentiment_score']:.2f} (Δ {row['sentiment_change_24h']:+.2f})")
        for d in row.get("top_discussions", [])[:3]:
            st.markdown(f"- [{d.get('title','discussion')}]({d.get('url','#')})")
        st.markdown("---")
else:
    st.info("No topics data available.")

# --------------------
# Footer / Credits
# --------------------
st.markdown("---")
st.caption("Designed for: market-wide price + sentiment analytics. Backend API endpoints are expected at BACKEND_API_BASE_URL (update variable). Mock fallback data is used when backend or CoinGecko is unavailable.")
