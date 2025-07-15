import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_N = 5  # Top N most relevant examples to keep per topic
SIMILARITY_THRESHOLD = 0.4  # Minimum confidence score
DYNAMIC_TOPIC_PREFIX = "NEW_"  # Prefix for auto-discovered topics

def clean_text_data(df):
    """Ensure all text fields are strings and handle missing values"""
    text_cols = ['title', 'comments']
    for col in text_cols:
        if col in df:
            df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
    return df

df = pd.read_csv("data/raw_reddit_data.csv")
df = clean_text_data(df)
df['combined'] = df['title'] + " - " + df['comments']

topic_data = {
    "Price Movement": {
        "examples": [
            "BTC just hit a new ATH of $70k!",
            "Why did Bitcoin dump 15% today?",
            "Price broke resistance at $30k.",
            "Market is tanking after CPI data release.",
            "Bitcoin surged after ETF approval rumors.",
            "Volatility is insane today.",
            "Bulls are losing control.",
            "This pullback looks healthy.",
            "Bitcoin holding strong above 20k.",
            "Bear market is confirmed?"
        ],
        "keywords": ["price", "ATH", "support", "resistance", "bullish", "bearish", "pump", "dump", "volatility", "crash", "surge"]
    },

    "Trading & TA": {
        "examples": [
            "RSI is signaling overbought territory.",
            "MACD just crossed bearish on the daily chart.",
            "I'm watching this ascending triangle breakout.",
            "EMA 200 is acting as support again.",
            "Volume is drying up — breakout coming?",
            "Golden cross confirmed!",
            "This looks like a classic bull flag.",
            "Setting a stop loss below key support.",
            "Targeting $30k if this level holds.",
            "Breakout from symmetrical triangle imminent."
        ],
        "keywords": ["RSI", "MACD", "technical analysis", "chart", "pattern", "candlestick", "indicator", "EMA", "trendline", "breakout", "volume", "stop loss"]
    },

    "Mining": {
        "examples": [
            "Bitcoin mining uses too much electricity.",
            "Hashrate dropped significantly this week.",
            "New ASIC miners are more energy efficient.",
            "China’s mining ban crushed global hashrate.",
            "Mining difficulty just adjusted upwards.",
            "Hydropower helps reduce mining carbon footprint.",
            "Miners are capitulating.",
            "North America is becoming the mining hub.",
            "Mining rewards are getting smaller.",
            "Can home mining still be profitable?"
        ],
        "keywords": ["mining", "hashrate", "ASIC", "difficulty", "energy", "miners", "proof of work", "block reward", "carbon", "electricity"]
    },

    "Regulation & Policy": {
        "examples": [
            "The SEC rejected another Bitcoin ETF.",
            "New crypto tax laws coming in 2025.",
            "El Salvador legalized Bitcoin.",
            "India plans to ban private cryptocurrencies.",
            "Gensler testifies on crypto before Congress.",
            "EU Parliament passes MiCA legislation.",
            "Bitcoin classified as a commodity, not security.",
            "FATF pushes for stricter KYC.",
            "Biden’s executive order on crypto regulation.",
            "Japan tightening crypto exchange laws."
        ],
        "keywords": ["regulation", "SEC", "ETF", "ban", "compliance", "KYC", "AML", "legislation", "approval", "policy", "tax", "law", "Gensler", "MiCA"]
    },

    "Adoption & Use Cases": {
        "examples": [
            "Starbucks now accepts Bitcoin!",
            "Bitcoin used for remittances in Africa.",
            "McDonald's accepts BTC in Switzerland.",
            "More countries exploring CBDCs.",
            "Lightning Network brings fast payments.",
            "Another city installs Bitcoin ATMs.",
            "Bitcoin donations spike during crisis.",
            "Retailers start taking crypto payments.",
            "BTC accepted for real estate in Portugal.",
            "University tuition payable with BTC."
        ],
        "keywords": ["adoption", "payment", "merchant", "accept", "real-world use", "legal tender", "ATM", "use case", "remittance", "retail"]
    },

    "Technology & Network": {
        "examples": [
            "Taproot activation improves smart contracts.",
            "Lightning Network capacity reaches new high.",
            "Running a full node is easier now.",
            "SegWit improves transaction throughput.",
            "Bitcoin Core releases v25.0.",
            "Mempool congestion is causing delays.",
            "New layer 2 protocol integrates with BTC.",
            "On-chain fees are spiking.",
            "Privacy features are getting better.",
            "How to set up a Lightning node?"
        ],
        "keywords": ["lightning", "taproot", "segwit", "node", "layer 2", "upgrade", "soft fork", "mempool", "core", "transaction", "on-chain"]
    },

    "Security & Hacks": {
        "examples": [
            "Ledger suffered a major data breach.",
            "Phishing scam drained $2 million in BTC.",
            "Use cold wallets, not exchanges.",
            "Seed phrase should never be shared.",
            "Mt. Gox repayment causing fear.",
            "Scammers impersonating influencers on X.",
            "Another exchange got hacked!",
            "Multisig is more secure.",
            "SIM swap attack stole my crypto.",
            "Hardware wallets are essential."
        ],
        "keywords": ["hack", "phishing", "scam", "wallet", "security", "breach", "exploit", "seed phrase", "cold storage", "private key"]
    },

    "Macro & Market Sentiment": {
        "examples": [
            "Bitcoin reacts to Fed's interest rate pause.",
            "Recession fears drive BTC demand.",
            "Dollar index impacting Bitcoin volatility.",
            "BTC behaves like digital gold.",
            "Inflation reports boost crypto market.",
            "Bitcoin decoupling from equities?",
            "Stocks down, crypto green.",
            "CPI numbers better than expected.",
            "Fear & Greed Index signals extreme fear.",
            "Will Bitcoin outperform gold this year?"
        ],
        "keywords": ["macro", "inflation", "interest rate", "fed", "usd", "DXY", "gold", "recession", "risk", "CPI", "fear and greed"]
    },

    "Memes & Culture": {
        "examples": [
            "HODL until $100k!",
            "WAGMI bros! 🚀",
            "Laser eyes till the moon!",
            "Degen plays only.",
            "When Lambo?",
            "Don't be a paper hands.",
            "Rug pull incoming? 😂",
            "Diamond hands all the way.",
            "We’re so back!",
            "Bitcoin is freedom, frens."
        ],
        "keywords": ["HODL", "WAGMI", "moon", "laser eyes", "lambo", "rekt", "diamond hands", "degen", "NGMI", "vibes"]
    },

    "Scams & FUD": {
        "examples": [
            "Bitconnect was a total scam.",
            "Fake giveaways on Twitter are back.",
            "This FUD is just market manipulation.",
            "Binance insolvency rumors again.",
            "Another rug pull just happened.",
            "Scammers cloning influencer accounts.",
            "Tether FUD resurfaces.",
            "Fear spreads after fake ETF news.",
            "Ponzi alert in Telegram group!",
            "Scam bots spamming the comments."
        ],
        "keywords": ["scam", "fud", "fraud", "giveaway", "rug pull", "fake news", "bitconnect", "ponzi", "manipulation", "bots"]
    }
}

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Precompute topic embeddings
topic_embeddings = {
    topic: embedding_model.encode(data["examples"])
    for topic, data in topic_data.items()
}

def classify_with_predefined(text):
    """Classify text using predefined topics"""
    text_emb = embedding_model.encode(text)
    best_topic = None
    best_score = -1
    
    for topic, example_embs in topic_embeddings.items():
        similarity = util.cos_sim(text_emb, example_embs).max().item()
        if similarity > best_score:
            best_score = similarity
            best_topic = topic
            
    return best_topic, best_score

def run_bertopic(texts):
    """Discover new topics using BERTopic with proper validation"""
    if len(texts) < 10:  # Minimum documents required
        print(f"Not enough documents for BERTopic (only {len(texts)} available)")
        return None, None, None
    
    # Initialize models with conservative parameters
    umap_model = UMAP(n_neighbors=5, n_components=3, min_dist=0.1)
    hdbscan_model = HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2)
    
    try:
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=5,
            verbose=False
        )
        
        # Validate embeddings first
        embeddings = embedding_model.encode(texts)
        if len(embeddings) == 0 or embeddings.shape[0] == 0:
            raise ValueError("Empty embeddings generated")
            
        topics, probs = topic_model.fit_transform(texts, embeddings)
        return topic_model, topics, probs
        
    except Exception as e:
        print(f"BERTopic failed: {str(e)}")
        return None, None, None

# First pass: classify with predefined topics
results = df['combined'].apply(lambda x: classify_with_predefined(x))
df['topic'] = results.apply(lambda x: x[0])
df['similarity'] = results.apply(lambda x: x[1])

# Second pass: discover new topics from unclassified
unclassified = df[df['topic'].isna() | (df['similarity'] < SIMILARITY_THRESHOLD)]
if len(unclassified) >= 10:  # Only run if enough unclassified docs
    print(f"Running BERTopic on {len(unclassified)} unclassified comments...")
    topic_model, new_topics, _ = run_bertopic(unclassified['combined'].tolist())
    
    if topic_model is not None:
        # Process new topics
        topic_info = topic_model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]
        
        # Update dataframe
        for idx, row in unclassified.iterrows():
            loc = unclassified.index.get_loc(idx)
            if loc < len(new_topics):  # Safety check
                topic_id = new_topics[loc]
                if topic_id != -1:
                    df.at[idx, 'topic'] = f"{DYNAMIC_TOPIC_PREFIX}{topic_id}"
                    df.at[idx, 'similarity'] = 0.9  # High confidence for new topics

        print(f"Discovered {len(valid_topics)} new topics")
    else:
        print("No new topics discovered")
else:
    print(f"Only {len(unclassified)} unclassified comments - skipping BERTopic")

top_comments = (
    df[~df['topic'].isna()]
    .sort_values('similarity', ascending=False)
    .groupby('topic')
    .head(TOP_N)
    .sort_values(['topic', 'similarity'], ascending=[True, False])
)

top_comments_display = top_comments[['topic', 'combined', 'similarity']]
top_comments_display['similarity'] = top_comments_display['similarity'].round(3)

print("\n=== Final Classification Results ===")
print(df[['combined', 'topic', 'similarity']])

print("\n=== Top 5 Comments per Topic ===")
for topic, group in top_comments.groupby('topic'):
    print(f"\n--- {topic} ---")
    for _, row in group.iterrows():
        comment = row['combined']
        truncated = (comment[:97] + "...") if len(comment) > 100 else comment
        print(f"[{row['similarity']:.2f}] {truncated}")

print("\n=== New Topics Discovered ===")
if 'topic_model' in locals() and topic_model is not None:
    print(topic_model.get_topic_info())
else:
    print("No topic model available. Skipping topic discovery output.")