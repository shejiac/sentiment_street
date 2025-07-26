# General DS libs
import numpy as np
import pandas as pd

# Data cleaning utils
from data_cleaning_utils import clean_text, is_junk_comment

# transformers libs
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    BertConfig,
    pipeline,
)

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

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=3)


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


# ==== SENTIMENT ANALYSIS PIPELINE ==== #
def sentiment_analysis_pipeline(
    df: pd.DataFrame, id_column: str, text_columns: List[str]
):
    """Run sentiment analysis on a DataFrame containing text data

    Args:
        df (pd.DataFrame): DataFrame containing text data.
        id_column (str): Column containing unique identifiers for each row.
        text_columns (str): List of columns containing text data.
    Returns:
        pd.DataFrame: DataFrame with sentiment analysis results."""

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
        sentiments = classifier(
            non_na_text_df[cleaned_text_column].tolist(),
            truncation=True,
            max_length=512,
        )

        # Attach sentiment scores back to the DataFrame
        sentiment_scores_column = f"sentiment_scores_{text_column}"
        non_na_text_df[sentiment_scores_column] = sentiments

        # Merge sentiment scores back to the original DataFrame
        df = df.merge(
            non_na_text_df[[id_column, cleaned_text_column, sentiment_scores_column]],
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
                x, min_neutral_score=-0.33, max_neutral_score=0.33
            )
        )

    # Once for loop is complete, return the DataFrame with sentiment analysis results
    return df


# ==== END OF SENTIMENT ANALYSIS PIPELINE ==== #
