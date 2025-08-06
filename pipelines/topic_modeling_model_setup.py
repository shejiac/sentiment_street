# Run this script to train a topic classification model using logistic regression.
# Run this script from the root directory of the project.
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import joblib

from data_cleaning_utils import is_junk_comment, clean_text

# === Parameters ===
EXAMPLES_JSON = "topic_modeling/topic_examples.json"

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


# === Save the model and label encoder for future use ===
joblib.dump(clf, "pipelines/models/topic_classifier.pkl")
joblib.dump(label_encoder, "pipelines/models/topic_label_encoder.pkl")
