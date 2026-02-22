---
title: Book Recommender
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
short_description: 'Book search/rec based on vector search: Query - Book desc'
---

Link to app: https://huggingface.co/spaces/rakeshjv2000/book-recommender

# 📚 Semantic Book Recommender

A semantic book recommendation system powered by vector search and emotion-aware ranking.

## 🚀 How It Works

1. User enters a book description.
2. The query is converted into an embedding using  
   `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face Inference API).
3. FAISS performs cosine similarity search over precomputed book embeddings.
4. Results are optionally filtered by category.
5. Results are optionally re-ranked by emotional tone (joy, surprise, anger, fear, sadness).

---

## 🧠 Features

- Semantic similarity search (FAISS)
- Emotion-based re-ranking
- Category filtering
- Fast retrieval with precomputed embeddings
- Hosted on Hugging Face Spaces (Gradio)

---

## 🛠 Tech Stack

- Python
- Gradio
- FAISS
- Hugging Face Inference API
- Pandas / NumPy

---
