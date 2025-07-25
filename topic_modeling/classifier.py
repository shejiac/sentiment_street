import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
import re

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# === Parameters ===
DATA_PATH = os.path.join("notebooks", "comments_df_with_scores.csv")
EXAMPLES_JSON = "topic_examples.json"
TOP_N_COMMENTS_PER_TOPIC = 5
UNCLASSIFIED_EXPORT_PATH = "unclassified_for_review.csv"
SIMILARITY_THRESHOLD = 0.15
MIN_WORD_COUNT = 4

def is_junk_comment(text):
    if not isinstance(text, str) or not text.strip():
        return True

    tokens = text.split()
    
    # Rule 1: Too short
    if len(tokens) < MIN_WORD_COUNT:
        return True

    # Rule 2: Only stopwords
    if all(word.lower() in STOPWORDS for word in tokens):
        return True

    # Rule 3: Contains only special characters or gibberish
    if re.fullmatch(r"[^a-zA-Z0-9\s]+", text):
        return True

    # Rule 4: Repeated characters (e.g., "loooool", "?????", "!!!!")
    if re.fullmatch(r"(.)\1{4,}", text.replace(" ", "")):
        return True

    # Rule 5: Contains only emojis or emoticons
    if re.fullmatch(r"[\U00010000-\U0010ffff\W\s]+", text):
        return True

    # Rule 6: Mostly non-alphabetic (e.g., links, symbols)
    num_alpha = sum(c.isalpha() for c in text)
    num_total = len(text)
    if num_alpha / max(num_total, 1) < 0.3:
        return True

    # Rule 7: One or two words repeated many times (e.g., "yes yes yes yes yes")
    if len(set(tokens)) <= 2 and len(tokens) > 5:
        return True

    return False

df = pd.read_csv(DATA_PATH)[['cleaned_comment']]
df = df[~df['cleaned_comment'].map(is_junk_comment)].reset_index(drop=True)

with open(EXAMPLES_JSON, "r", encoding="utf-8") as f:
    topic_data = json.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Prepare training data ===
train_texts = []
train_labels = []
for topic, data in topic_data.items():
    for example in data["examples"]:
        if not is_junk_comment(example):
            train_texts.append(example)
            train_labels.append(topic)

train_embeddings = embedding_model.encode(train_texts, convert_to_tensor=False)

# Encode labels for sklearn
label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_labels)

# === Train Logistic Regression ===
clf = LogisticRegression(max_iter=1000)
clf.fit(train_embeddings, train_y)

# === Predict on actual comments ===
comment_embeddings = embedding_model.encode(df['cleaned_comment'].tolist(), show_progress_bar=False)
pred_probs = clf.predict_proba(comment_embeddings)
pred_labels = clf.predict(comment_embeddings)

# Assign topic and confidence
max_probs = pred_probs.max(axis=1)
df['suggested_topic'] = label_encoder.inverse_transform(pred_labels)
df['similarity'] = max_probs  # Treating classifier confidence like similarity

# Use threshold
df['topic'] = df.apply(lambda x: x['suggested_topic'] if x['similarity'] >= SIMILARITY_THRESHOLD else None, axis=1)

# === Export unclassified ===
unclassified = df[df['topic'].isna()].copy()
unclassified['manual_topic'] = ""
unclassified[['cleaned_comment', 'suggested_topic', 'similarity', 'manual_topic']].to_csv(
    UNCLASSIFIED_EXPORT_PATH, index=False
)
print(f"Exported {len(unclassified)} unclassified comments to {UNCLASSIFIED_EXPORT_PATH}")

# === Summarize top comments per topic ===
classified = df[df['topic'].notna()]
top_comments = (
    classified
    .sort_values(by="similarity", ascending=False)
    .groupby("topic")
    .head(TOP_N_COMMENTS_PER_TOPIC)
    .reset_index(drop=True)
)

topic_stats = (
    top_comments
    .groupby('topic')
    .agg(
        comment_count=('cleaned_comment', 'count'),
        avg_similarity=('similarity', 'mean')
    )
    .sort_values(by=['comment_count', 'avg_similarity'], ascending=[False, False])
    .reset_index()
)
topic_stats['rank'] = range(1, len(topic_stats) + 1)

# === Save summaries ===
today_str = datetime.today().strftime('%Y%m%d')
topic_summary_path = f"topic_rank_summary_{today_str}.csv"
topic_stats[['rank', 'topic', 'comment_count', 'avg_similarity']].to_csv(topic_summary_path, index=False)
print(f"Saved topic rank summary to: {topic_summary_path}")

top_comments = pd.merge(top_comments, topic_stats[['topic', 'rank']], on='topic', how='left')
top_comments.rename(columns={'cleaned_comment': 'comment', 'rank': 'topic_rank'}, inplace=True)
top_comments = top_comments.sort_values(by=['topic_rank', 'similarity'], ascending=[True, False])
top_comments[['topic_rank', 'topic', 'similarity', 'comment']].to_csv(
    f"top_comments_by_topic_{today_str}.csv", index=False
)
print(f"Saved top comments by topic to: top_comments_by_topic_{today_str}.csv")
