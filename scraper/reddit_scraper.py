import praw
import os
import time
import datetime
import psycopg2
import ssl
from dotenv import load_dotenv
from prawcore.exceptions import ResponseException

load_dotenv()

# ==========================
# CONFIGURATION
# ==========================
SUBREDDIT_NAME = "Bitcoin"
SECONDS_IN_A_DAY = 86400
NOW_UTC = int(time.time())

# ==========================
# Reddit API (ENV VARS)
# ==========================
# Load environment variables
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "sentiment_street/0.1 by u/the_user")

# Initialize Reddit instance
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

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_reddit_posts_created_utc
ON reddit_posts (created_utc);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS reddit_comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT REFERENCES reddit_posts(post_id) ON DELETE CASCADE,
    author TEXT,
    body TEXT,
    score INTEGER,
    created_utc INTEGER
);
""")
conn.commit()

# ==========================
# Delete old posts & comments
# ==========================
DAYS_TO_KEEP = 3
cursor.execute("""
    DELETE FROM reddit_posts
    WHERE created_utc < EXTRACT(EPOCH FROM NOW()) - (%s * 86400);
""", (DAYS_TO_KEEP,))
conn.commit()
print(f"Deleted posts and comments older than {DAYS_TO_KEEP} days.")

# ==========================
# Start Scraping
# ==========================
print(f"Scraping started at {datetime.datetime.now()}")
subreddit = reddit.subreddit(SUBREDDIT_NAME)
inserted = 0

try:
    for post in subreddit.new(limit=None):
        if int(post.created_utc) < NOW_UTC - SECONDS_IN_A_DAY:
            break

        post.comments.replace_more(limit=0)
        top_comments = post.comments.list()[:20]
        comments_text = " || ".join(
            [f"{c.id}: {c.body}" for c in top_comments]
        )
        flair = post.link_flair_text or ""

        # Insert post
        cursor.execute("""
            INSERT INTO reddit_posts (
                post_id, subreddit, title, body, author,
                created_utc, upvotes, score, num_comments,
                flair, comments, upvote_ratio, tickers
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (post_id) DO NOTHING;
        """, (
            post.id,
            post.subreddit.display_name,
            post.title,
            post.selftext,
            str(post.author) if post.author else "nan",
            int(post.created_utc),
            post.ups,
            post.score,
            post.num_comments,
            flair,
            comments_text,
            post.upvote_ratio,
            ""  # tickers placeholder
        ))

        # Insert comments
        for c in top_comments:
            try:
                cursor.execute("""
                    INSERT INTO reddit_comments (
                        comment_id, post_id, author, body, score, created_utc
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (comment_id) DO NOTHING;
                """, (
                    c.id,
                    post.id,
                    str(c.author) if c.author else "nan",
                    c.body,
                    c.score,
                    int(c.created_utc)
                ))
            except Exception as comment_error:
                print(f"Failed to insert comment {c.id}: {comment_error}")

        conn.commit()
        inserted += 1
        print(f"Inserted post: {post.id} — {post.title[:50]}...")
        time.sleep(1)

except ResponseException as reddit_error:
    print("Reddit API error:", reddit_error.response.text)
except Exception as general_error:
    print("Error during scraping:", general_error)

cursor.close()
conn.close()
print(f"Done. Inserted {inserted} posts into PostgreSQL.")
