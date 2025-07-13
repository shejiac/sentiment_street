from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import pipeline
import pandas as pd

df = pd.read_csv("sentiment_street/data/raw_reddit_data.csv")
df['combined'] = df['title'].astype(str) + " - " + df['comments'].astype(str)

safe_length = 1500
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['combined'].tolist())

def best_k_by_silhouette(embeddings, k_range=range(2, 8)):
    best_k, best_score = 2, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        score = silhouette_score(embeddings, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

k = best_k_by_silhouette(embeddings)
kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
df['cluster'] = kmeans.labels_

summarizer = pipeline("summarization", model="google/flan-t5-base")

def summarize_cluster_texts(texts, chunk_size=10):
    summaries = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        joined_chunk = " ".join(chunk)
        if len(joined_chunk) > safe_length:
            joined_chunk = joined_chunk[:safe_length]
        try:
            s = summarizer(joined_chunk, max_length=20, min_length=5, do_sample=False)
            summaries.append(s[0]['summary_text'])
        except:
            continue
    merged_summary_text = " ".join(summaries)
    if len(merged_summary_text) > safe_length:
        merged_summary_text = merged_summary_text[:safe_length]
    final_summary = summarizer(merged_summary_text, max_length=15, min_length=5, do_sample=False)
    return final_summary[0]['summary_text']

cluster_names = {}
for cid in range(k):
    texts = df[df['cluster'] == cid]['combined'].tolist()
    cluster_names[cid] = summarize_cluster_texts(texts)

df['topic'] = df['cluster'].map(cluster_names)

print(df[['title', 'comments', 'topic']])