# SentimentStreet

SentimentStreet is a web application that collects, analyzes, and visualizes cryptocurrency-related discussions on Reddit. It integrates sentiment analysis, topic modeling, and price prediction into a unified automated pipeline, with a simple UI for exploring results.

---

## 📌 Project Overview

The system:
- Scrapes Reddit posts and comments from crypto-related subreddits.
- Enriches data with relevant market metrics from CoinGecko.
- Analyzes sentiment trends at both the post and comment levels.
- Identifies trending topics in the crypto community.
- Predicts short-term cryptocurrency price movements.
- Provides an interactive dashboard for monitoring results.

**Data Flow:**
Scrape → Store in DB → Export for analysis → Sentiment/Topic Models → Store scores → Predict prices → Display in UI

---

## 🎯 Objectives
- Automate the collection of Reddit crypto discussions and relevant market data.
- Analyze sentiment trends at both post and comment level.
- Identify trending topics in crypto communities.
- Predict short-term crypto price movements using sentiment and technical features.
- Provide a user-friendly UI for data exploration.

---

## 🏗 System Architecture

**Main Components**
1. Data Collection — Reddit Scraper + CoinGecko API integration  
2. Database Storage — PostgreSQL (via [Neon.tech](https://neon.com/))  
3. Automated Analysis Pipeline — Sentiment Analysis + Topic Modeling (HuggingFace transformers)  
4. Price Prediction Models — Regression & Classification  
5. User Interface — Streamlit app

---

## 🛠 Technologies Used

- Languages: Python  
- APIs: Reddit API (PRAW), CoinGecko API  
- Database: PostgreSQL (Neon.tech)  
- ML Frameworks: HuggingFace Transformers, Sentence Transformers, scikit-learn, XGBoost, LightGBM, CatBoost  
- UI Framework: Streamlit  
- Other Libraries: pandas, numpy, matplotlib/plotly, PRAW, psycopg2, json  

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- Neon.tech credentials
- Reddit API credentials
- CoinGecko API key (optional, for demo frontend)

### 1. Create a virtual environment (example using Conda)
\`\`\`bash
conda create -n sentimentstreet python=3.12
conda activate sentimentstreet
\`\`\`

### 2. Clone the repository and install dependencies
\`\`\`bash
git clone <repository_url>
cd <project_folder>
pip install -r requirements.txt
\`\`\`

### 3. Configure environment variables  
Create a \`.env\` file in the root directory:

\`\`\`env
# Reddit API credentials
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent

# Neon.tech PostgreSQL credentials
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_dbname
POSTGRES_HOST=your_host
POSTGRES_PORT=5432
\`\`\`

---

## 🚀 Usage

Run each component as needed:

### Daily Scraper
\`\`\`bash
python scraper/reddit_scraper.py
\`\`\`

### Automated Text Analysis (Sentiment + Topic Modeling)
\`\`\`bash
python pipelines/automated_text_analysis.py
\`\`\`

### Train Price Prediction Models
\`\`\`bash
python price_prediction/train_model.py
\`\`\`

### Launch the UI
\`\`\`bash
streamlit run streamlit_frontend.py
\`\`\`

---

## ⏱ Automated Scheduling

A daily cron job is` set up to:
- Run \`reddit_scraper.py\` to collect new data.
- Run \`automated_text_analysis.py\` to process sentiment & topics.

**Automated Workflow:**
1. Scrape relevant Reddit posts & comments.
2. Store results in PostgreSQL.
3. Run sentiment analysis & topic modeling automatically.

---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue to discuss what you’d like to change.

---

## 📧 Contact
For questions or feedback, please reach out via email or open an issue in the repository.
