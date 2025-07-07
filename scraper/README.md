# Reddit Scraper

This project collects daily Reddit posts and comments from the r/Bitcoin subreddit, storing them in a PostgreSQL database hosted on Neon. It uses praw for Reddit API access and is fully automated via GitHub Actions.

# Project Structure
<ul>
  <li><strong>scraper/</strong>
    <ul>
      <li><code>reddit_scraper.py</code> – Reddit scraper script</li>
      <li><code>requirements.txt</code> – Specific to Reddit scraper</li>
    </ul>
  </li>
  <li><code>.env</code> – Environment variables (not committed)</li>
  <li><strong>.github/</strong>
    <ul>
      <li><strong>workflows/</strong>
        <ul>
          <li><code>cron.yml</code> – GitHub Actions cron job (runs daily at 12AM SGT)</li>
        </ul>
      </li>
    </ul>
  </li>
</ul>


# Requirements
Python 3.10+
PostgreSQL (Neon.tech)
praw, psycopg2, python-dotenv


**Install with:**
pip install -r requirements.txt


# env Setup

Create a .env file with the following:

<h4><strong>PostgreSQL (Neon or local)</strong></h4>
<pre><code>
PG_DB=neondb
PG_USER=neondb_owner
PG_PASSWORD=your_postgres_password
PG_HOST=your_neon_host
PG_PORT=5432
</code></pre>

<h4><strong>Reddit API</strong></h4>
<pre><code>
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=sentiment_scraper/0.1 by u/your_username
</code></pre>

Note: An .env file is shared through Telegram, containing these credentials, don't commit them pls haha


# Usage

To run the scraper locally:
python scraper/reddit_scraper.py

It will:
<ul>Fetch r/Bitcoin posts from the past 24 hours</ul>
<ul>Extract top-level comments (max 20 per post)</ul>
<ul>Save results into reddit_posts and reddit_comments tables</ul>
<ul>Delete any data older than 3 days</ul>


# GitHub Actions Automation

This repo includes a cron job that runs daily:

File: .github/workflows/cron.yml
on:
  schedule:
    - cron: '0 16 * * *'  # Runs at 12:00 AM Singapore Time (UTC+8)

Secrets must be added under Settings > Secrets and variables > Actions:
<ul>PG_DB, PG_USER, PG_PASSWORD, PG_HOST, PG_PORT</ul>
<ul>REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT</ul>

Note: The secrets have already been added


# Sample SQL Queries to be run in Neon.tech

-- View recent posts
SELECT * FROM reddit_posts ORDER BY created_utc DESC LIMIT 10;

-- View recent comments
SELECT * FROM reddit_comments ORDER BY created_utc DESC LIMIT 10;

Thank you!

