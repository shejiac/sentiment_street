# ==== IMPORTS ==== #
# General DS libs
import numpy as np
import pandas as pd

# Data cleaning utils
from data_cleaning_utils import clean_text

# transformers libs
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    BertConfig,
    pipeline,
)
from sentence_transformers import SentenceTransformer

# Loading models
import joblib

# Type hinting
from typing import List

# Garbage collection to free up memory
import gc

# Delete CUDA cache if already loaded to save memory
# This means the model is never cached
try:
    del tokenizer
except:
    print("tokenizer already deleted")

try:
    del model
except:
    print("model already deleted")

torch.cuda.empty_cache()
gc.collect()



# ==== MODEL SETUP ==== #
# Sentiment analysis model setup
model_name = "ProsusAI/finbert"
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}

tokenizer = BertTokenizerFast.from_pretrained(
    model_name,
    use_fast=True,
    padding_side="left",
    padding_token="[PAD]",
)

config = BertConfig.from_pretrained(
    model_name,
    num_labels=3,
    label2id=label2id,
    id2label=id2label,
)

model = BertForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
)

sentiment_classifier = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, top_k=3
)

# Topic modeling model setup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_classifier = joblib.load("models/topic_modeling/topic_classifier.pkl")
topic_label_encoder = joblib.load("models/topic_modeling/topic_label_encoder.pkl")



# ==== CALCULATE COMPOUND SCORE ==== #
def calc_compound_score(score_list):
    """Calculate compound score of a piece of text as positive probability - negative probability"""
    # If the score list is NaN, leave it be
    if isinstance(score_list, float):
        if pd.isna(score_list):
            return np.nan

    return score_list[0]["score"] - score_list[1]["score"]

# ==== EXTRACT SENTIMENT LABEL FROM SCORE ==== #
def extract_sentiment_label(score, min_neutral_score=-0.33, max_neutral_score=0.33):
    """Extract sentiment label from the compound score"""
    # If the score is NaN, leave it be
    if pd.isna(score):
        return np.nan

    # Positive if score from 0.33 to 1.0, negative if score from -1.0 to -0.33, neutral otherwise
    if score > max_neutral_score:
        return "positive"
    elif score < min_neutral_score:
        return "negative"
    else:
        return "neutral"



# ==== TEXT ANALYSIS PIPELINE ==== #
def text_analysis_pipeline(
    df: pd.DataFrame,
    id_column: str,
    text_columns: List[str],
    similarity_threshold: float = 0.5,
    min_neutral_score: float = -0.33,
    max_neutral_score: float = 0.33,
):
    """Run sentiment analysis & topic classification on a DataFrame containing text data

    Args:
        df (pd.DataFrame): DataFrame containing text data.
        id_column (str): Column containing unique identifiers for each row.
        text_columns (str): List of columns containing text data.
        similarity_threshold (float): Threshold for topic classification confidence.
        min_neutral_score (float): Minimum score for neutral sentiment.
        max_neutral_score (float): Maximum score for neutral sentiment.
    Returns:
        pd.DataFrame: DataFrame with text analysis results."""

    df = df.copy()  # Work on a copy of the DataFrame to avoid modifying the original

    # Alert if the DataFrame is empty
    if df.empty:
        print("DataFrame is empty. Returning an empty DataFrame.")
        return None

    # Alert if id and text columns are not in the DataFrame
    if id_column not in df.columns or not all(
        col in df.columns for col in text_columns
    ):
        print(
            f"The DataFrame does not contain the specified ID column or text columns."
        )
        return None

    # Warn if torch is current using CPU instead of GPU
    if not torch.cuda.is_available():
        print(
            "Warning: Torch is currently using CPU. Sentiment analysis will be slower than expected."
        )

    # Repeat for each text column
    for text_column in text_columns:
        # Clean the text data
        cleaned_text_column = f"cleaned_{text_column}"
        df[cleaned_text_column] = df[text_column].apply(clean_text)

        # Keep only non-empty cleaned text for sentiment analysis
        non_na_text_df = df[df[cleaned_text_column] != ""][
            [id_column, cleaned_text_column]
        ]

        # Run sentiment analysis using the FinBERT model
        sentiments = sentiment_classifier(
            non_na_text_df[cleaned_text_column].tolist(),
            truncation=True,
            max_length=512,
        )
        sentiment_scores_column = f"sentiment_scores_{text_column}"
        non_na_text_df[f"sentiment_scores_{text_column}"] = sentiments

        # Obtain topics from the topic classifier
        pred_labels = topic_classifier.predict(
            non_na_text_df[cleaned_text_column].tolist()
        )
        topic_column = f"topic_{text_column}"
        non_na_text_df[topic_column] = topic_label_encoder.inverse_transform(
            pred_labels
        )
        # Obtain similarity scores from the topic classifier
        similarity_column = f"similarity_{text_column}"
        pred_probs = topic_classifier.predict_proba(
            non_na_text_df[cleaned_text_column].tolist()
        )
        max_probs = np.max(pred_probs, axis=1)
        non_na_text_df[similarity_column] = max_probs
        # Apply threshold, if lower than threshold, set topic as None
        non_na_text_df[topic_column] = non_na_text_df.apply(
            lambda x: (
                x[topic_column]
                if x[similarity_column] >= similarity_threshold  # Pipeline argument
                else None
            ),
            axis=1,
        )

        # Merge sentiment scores & predicted topics back to the original DataFrame
        df = df.merge(
            non_na_text_df[
                [id_column, cleaned_text_column, sentiment_scores_column, topic_column]
            ],
            on=[id_column, cleaned_text_column],
            how="left",  # left join to keep all original rows, some may have NaN scores
        )

        # Calculate compound scores
        compound_score_column = f"compound_score_{text_column}"
        df[compound_score_column] = df[sentiment_scores_column].apply(
            calc_compound_score
        )

        # Extract sentiment labels
        sentiment_label_column = f"sentiment_label_{text_column}"
        df[sentiment_label_column] = df[compound_score_column].apply(
            lambda x: extract_sentiment_label(
                x,
                min_neutral_score=min_neutral_score,  # Pipeline arguments
                max_neutral_score=max_neutral_score,  # Pipeline arguments
            )
        )

    # Once for loop is complete, return the DataFrame with sentiment analysis results
    return df


# === AGGREGATE DAILY COIN SENTIMENT === #
def get_daily_coin_sentiment(df: pd.DataFrame, coin_name: str) -> pd.DataFrame:
    """
    Aggregates sentiment scores for a specific coin on a daily basis.
    Assumes df has 'created_utc' and 'compound_score_body' (from comments) and 'coin_name'.
    """
    if df.empty:
        return pd.DataFrame()

    # Convert created_utc to datetime objects and set as index
    df["date"] = pd.to_datetime(df["created_utc"], unit="s").dt.date

    # Filter for the specific coin
    coin_df = df[df["coin_name"] == coin_name].copy()
    if coin_df.empty:
        print(f"No data for coin: {coin_name}")
        return pd.DataFrame()

    # Aggregate compound scores per day
    # We can use mean, median, or sum of compound scores. Mean is a good starting point.
    daily_sentiment = (
        coin_df.groupby("date")["compound_score_body"].mean().reset_index()
    )
    daily_sentiment.rename(
        columns={"compound_score_body": "daily_avg_sentiment"}, inplace=True
    )
    daily_sentiment["coin_name"] = coin_name

    return daily_sentiment
