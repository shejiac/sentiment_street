# Run this file to set up a sentiment analysis model.
# Current method of set up: Using ready-made FinBERT pipeline
# Run this script from the root directory of the project.
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    BertConfig,
)

MODEL_NAME = "ProsusAI/finbert"
SAVE_PATH = "models/finbert"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
PADDING_TOKEN = "[PAD]"

tokenizer = BertTokenizerFast.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    padding_side="left",
    padding_token=PADDING_TOKEN,
)

config = BertConfig.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
)

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    device_map="auto",
)

# Save the model and tokenizer
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
