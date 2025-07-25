import os
import time
import psycopg2
from dotenv import load_dotenv

import pandas as pd
from sent_analysis_functions import sentiment_analysis_pipeline
from io import StringIO

load_dotenv()

# ==========================
# CONFIGURATION
# ==========================
SUBREDDIT_NAME = "Bitcoin"
SECONDS_IN_A_DAY = 86400
NOW_UTC = int(time.time())
LIMIT_POSTS = 100  # <-- You can adjust this safely

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
# Retrieve posts and comments from DB
# ==========================
retrieve_from = NOW_UTC - (SECONDS_IN_A_DAY * 1)  # 1 day ago
posts_df = pd.read_sql(
    """ SELECT post_id, title, body, score
    FROM reddit_posts
    WHERE created_utc >= %s """,
    conn,
    params=(retrieve_from,),
)
comments_df = pd.read_sql(
    """ SELECT comment_id, body, score
    FROM reddit_comments
    WHERE created_utc >= %s """,
    conn,
    params=(retrieve_from,),
)

# ==========================
# Pass DataFrames through sentiment pipeline
# ==========================
posts_scores_df = sentiment_analysis_pipeline(
    posts_df, id_column="post_id", text_columns=["title", "body"]
)
comments_scores_df = sentiment_analysis_pipeline(
    comments_df, id_column="comment_id", text_columns=["body"]
)

print(
    f"Posts processed: {len(posts_scores_df)}, Comments processed: {len(comments_scores_df)}"
)
print("Columns in posts_scores_df:", posts_scores_df.columns)
print("Columns in comments_scores_df:", comments_scores_df.columns)

# ==========================
# Create Tables if Needed
# ==========================
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS reddit_posts_scores (
    post_id TEXT PRIMARY KEY,
    title TEXT,
    body TEXT,
    score INTEGER,
    cleaned_title TEXT,
    compound_score_title FLOAT,
    sentiment_label_title TEXT,
    cleaned_body TEXT,
    compound_score_body FLOAT,
    sentiment_label_body TEXT
);
"""
)

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS reddit_comments_scores (
    comment_id TEXT PRIMARY KEY,
    body TEXT,
    score INTEGER,
    cleaned_body TEXT,
    compound_score_body FLOAT,
    sentiment_label_body TEXT
);
"""
)
conn.commit()

# ==========================
# Insert DataFrame entries into the tables
# ==========================
# Don't keep the sentiment scores list, just the compound score and sentiment label
posts_scores_df = posts_scores_df[["posts_id", "title", "body", "score", "cleaned_title", "compound_score_title", "sentiment_label_title", "cleaned_body", "compound_score_body", "sentiment_label_body"]]
posts_scores_df = posts_scores_df.where(
    pd.notnull(posts_scores_df), None
)  # Replace all NaN value with None so that it's empty in CSV
posts_scores_buffer = StringIO()
posts_scores_df.to_csv(posts_scores_buffer, index=False, header=False)
posts_scores_buffer.seek(0)

# Don't keep the sentiment scores list, just the compound score and sentiment label
comments_scores_df = comments_scores_df[["comment_id", "body", "score", "cleaned_body", "compound_score_body", "sentiment_label_body"]]
comments_scores_df = comments_scores_df.where(
    pd.notnull(comments_scores_df), None
)  # Replace all NaN value with None so that it's empty in CSV (comments_scores_df doesn't usually have nulls, but just to be sure)
comments_scores_buffer = StringIO()
comments_scores_df.to_csv(comments_scores_buffer, index=False, header=False)
comments_scores_buffer.seek(0)

try:
    cursor.copy_from(
        posts_scores_buffer,
        "reddit_posts_scores",
        sep=",",
        null="",
    )
except Exception as e:
    print("Error during insertion to reddit_posts_scores:", e)

try:

    cursor.copy_from(
        comments_scores_buffer,
        "reddit_comments_scores",
        sep=",",
        null="",
    )
except Exception as e:
    print("Error during insertion to reddit_comments_scores:", e)

cursor.close()
conn.close()
print(f"Done. Inserted posts and comments with sentiment scores into PostgreSQL.")
