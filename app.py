# app.py
import os
import time
import numpy as np
import pandas as pd
import faiss
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

BOOKS_CSV = "books_with_emotions.csv"
FAISS_INDEX_PATH = "books.index"
ID_MAP_PATH = "id_map.npy"

HF_TOKEN = os.getenv("HF_TOKEN")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing. Set in .env (local) or HF Spaces Secrets.")

client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

# -----------------------------
# LOAD DATA
# -----------------------------
books = pd.read_csv(BOOKS_CSV)
books["isbn13"] = books["isbn13"].astype(str)

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

index = faiss.read_index(FAISS_INDEX_PATH)
id_map = np.load(ID_MAP_PATH, allow_pickle=True).astype(str)

# -----------------------------
# EMBEDDING
# -----------------------------
def hf_embed_query(text: str, retry=3, sleep_s=2.0) -> np.ndarray:
    last_err = None
    for attempt in range(retry):
        try:
            out = client.feature_extraction(text, model=HF_EMBEDDING_MODEL)
            arr = np.array(out, dtype=np.float32)

            # token-level -> mean pool
            if arr.ndim == 2:
                v = arr.mean(axis=0)
            elif arr.ndim == 1:
                v = arr
            else:
                v = arr.reshape(-1, arr.shape[-1]).mean(axis=0)

            v = v.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(v)
            return v
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"HF query embedding failed: {last_err}")

# -----------------------------
# RETRIEVAL + FILTERING
# -----------------------------
TONE_TO_COL = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}

def retrieve_semantic_recommendations(query: str, category: str, tone: str, initial_top_k=80, final_top_k=16):
    qv = hf_embed_query(query)
    scores, idx = index.search(qv, initial_top_k)

    retrieved_isbns = id_map[idx[0]].tolist()
    retrieved_scores = scores[0].tolist()

    rank_df = pd.DataFrame({
        "isbn13": [str(x) for x in retrieved_isbns],
        "rank": list(range(len(retrieved_isbns))),
        "sim": retrieved_scores,
    })

    recs = (
        books.merge(rank_df, on="isbn13", how="inner")
             .sort_values("rank")
             .copy()
    )

    if category and category != "All":
        recs = recs[recs["simple_categories"] == category]

    recs = recs.head(initial_top_k)

    if tone and tone != "All":
        col = TONE_TO_COL.get(tone)
        if col in recs.columns:
            recs = recs.sort_values(by=col, ascending=False)

    return recs.head(final_top_k).copy()

# -----------------------------
# UI HELPERS
# -----------------------------
def format_authors(authors_raw: str) -> str:
    authors_raw = str(authors_raw or "")
    authors_split = [a.strip() for a in authors_raw.split(";") if a.strip()]
    if len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    if len(authors_split) > 2:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    return authors_raw

def truncate(text: str, n_words=28) -> str:
    words = str(text or "").split()
    return " ".join(words[:n_words]) + ("…" if len(words) > n_words else "")

def emotion_chips(row) -> str:
    cols = ["joy", "surprise", "anger", "fear", "sadness"]
    vals = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            try:
                vals.append((c, float(row[c])))
            except:
                pass
    vals.sort(key=lambda x: x[1], reverse=True)
    top = vals[:2]
    if not top:
        return ""
    return " ".join([f"<span class='chip'>{k}: {v:.2f}</span>" for k, v in top])

def cards_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<div class='empty'>No results found. Try a different query or set filters to All.</div>"

    cards = []
    for _, row in df.iterrows():
        title = row.get("title", "")
        authors = format_authors(row.get("authors", ""))
        cat = row.get("simple_categories", "Unknown")
        desc = truncate(row.get("description", ""), 28)
        img = row.get("large_thumbnail", "cover-not-found.jpg")
        sim = row.get("sim", None)
        sim_str = f"{float(sim):.3f}" if sim is not None else "—"
        chips = emotion_chips(row)

        cards.append(f"""
        <div class="card">
          <div class="cover">
            <img src="{img}" onerror="this.onerror=null;this.src='cover-not-found.jpg';" />
          </div>
          <div class="info">
            <div class="title">{title}</div>
            <div class="authors">{authors}</div>
            <div class="meta">
              <span class="badge">{cat}</span>
              <span class="score">Similarity: {sim_str}</span>
            </div>
            <div class="desc">{desc}</div>
            <div class="chips">{chips}</div>
          </div>
        </div>
        """)

    return f"<div class='grid'>{''.join(cards)}</div>"

# -----------------------------
# MAIN ACTION
# -----------------------------
def run_search(query, category, tone, top_k):
    if not query or not query.strip():
        return "<div class='empty'>Type a short description to get recommendations.</div>"
    recs = retrieve_semantic_recommendations(
        query=query.strip(),
        category=category,
        tone=tone,
        initial_top_k=80,
        final_top_k=int(top_k),
    )
    return cards_html(recs)

# -----------------------------
# FANCY CSS
# -----------------------------
CSS = """
:root { --radius: 18px; }

.wrap { max-width: 1200px; margin: 0 auto; }
.hero { padding: 18px 18px 6px 18px; }
.hero h1 { margin: 0; font-size: 28px; }
.hero p { margin: 6px 0 0 0; opacity: 0.85; }

.grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(220px, 1fr));
  gap: 14px;
  padding: 8px 2px 2px 2px;
}

@media (max-width: 1100px) { .grid { grid-template-columns: repeat(3, minmax(220px, 1fr)); } }
@media (max-width: 850px)  { .grid { grid-template-columns: repeat(2, minmax(220px, 1fr)); } }
@media (max-width: 520px)  { .grid { grid-template-columns: 1fr; } }

.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: var(--radius);
  overflow: hidden;
  background: rgba(255,255,255,0.04);
  box-shadow: 0 8px 30px rgba(0,0,0,0.18);
  display: flex;
  flex-direction: column;
  min-height: 360px;
}

.cover { width: 100%; height: 220px; overflow: hidden; background: rgba(0,0,0,0.10); }
.cover img { width: 100%; height: 100%; object-fit: cover; display:block; }

.info { padding: 12px 12px 14px 12px; display:flex; flex-direction: column; gap: 6px; }
.title { font-weight: 700; font-size: 14.5px; line-height: 1.2; }
.authors { opacity: 0.85; font-size: 12.5px; }

.meta { display:flex; align-items:center; justify-content: space-between; gap: 8px; margin-top: 2px;}
.badge {
  font-size: 11px;
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.12);
}
.score { font-size: 11px; opacity: 0.85; }

.desc { font-size: 12px; opacity: 0.85; line-height: 1.35; margin-top: 2px; }
.chips { display:flex; flex-wrap: wrap; gap: 6px; margin-top: 6px;}
.chip {
  font-size: 10.5px;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.10);
}

.empty {
  padding: 14px;
  border-radius: var(--radius);
  border: 1px dashed rgba(255,255,255,0.18);
  opacity: 0.9;
}
"""

categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

EXAMPLES = [
    ("A cozy small-town mystery with a charming detective and light humor", "All", "All", 16),
    ("A deeply emotional story about healing after loss and finding hope", "All", "Sad", 16),
    ("A fast-paced sci-fi adventure with space travel and big surprises", "All", "Surprising", 16),
    ("A smart non-fiction book that explains psychology and human behavior", "All", "All", 16),
]

with gr.Blocks() as demo:
    with gr.Column(elem_classes=["wrap"]):
        gr.HTML("""
        <div class="hero">
          <h1>📚 Semantic Book Recommender</h1>
          <p>Describe a book vibe you want. Get semantic matches + optional category filter + emotion-based ranking.</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                query = gr.Textbox(
                    label="What kind of book are you looking for?",
                    placeholder="e.g., A suspenseful mystery with a clever twist and strong characters",
                    lines=2
                )

                with gr.Row():
                    ex1 = gr.Button("✨ Cozy mystery")
                    ex2 = gr.Button("💔 Emotional healing")
                    ex3 = gr.Button("🚀 Sci-fi adventure")
                    ex4 = gr.Button("🧠 Smart non-fiction")

            with gr.Column(scale=1):
                gr.Markdown("### Filters")
                category = gr.Dropdown(choices=categories, value="All", label="Category")
                tone = gr.Dropdown(choices=tones, value="All", label="Emotional tone")
                top_k = gr.Slider(4, 24, value=16, step=4, label="Number of results")
                btn = gr.Button("🔎 Find recommendations", variant="primary")

        gr.Markdown("### Results")
        results = gr.HTML("<div class='empty'>Search results will appear here.</div>")

        # Normal search
        btn.click(run_search, inputs=[query, category, tone, top_k], outputs=results)

        # Example buttons: set query + optionally set tone, then run search
        ex1.click(lambda: EXAMPLES[0][0], inputs=None, outputs=query)\
           .then(lambda: "All", inputs=None, outputs=category)\
           .then(lambda: "All", inputs=None, outputs=tone)\
           .then(lambda: 16, inputs=None, outputs=top_k)\
           .then(run_search, inputs=[query, category, tone, top_k], outputs=results)

        ex2.click(lambda: EXAMPLES[1][0], inputs=None, outputs=query)\
           .then(lambda: "All", inputs=None, outputs=category)\
           .then(lambda: "Sad", inputs=None, outputs=tone)\
           .then(lambda: 16, inputs=None, outputs=top_k)\
           .then(run_search, inputs=[query, category, tone, top_k], outputs=results)

        ex3.click(lambda: EXAMPLES[2][0], inputs=None, outputs=query)\
           .then(lambda: "All", inputs=None, outputs=category)\
           .then(lambda: "Surprising", inputs=None, outputs=tone)\
           .then(lambda: 16, inputs=None, outputs=top_k)\
           .then(run_search, inputs=[query, category, tone, top_k], outputs=results)

        ex4.click(lambda: EXAMPLES[3][0], inputs=None, outputs=query)\
           .then(lambda: "All", inputs=None, outputs=category)\
           .then(lambda: "All", inputs=None, outputs=tone)\
           .then(lambda: 16, inputs=None, outputs=top_k)\
           .then(run_search, inputs=[query, category, tone, top_k], outputs=results)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Glass(), css=CSS)
