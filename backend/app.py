"""
Jesse Livermore RAG ChatBot + Strategy Backtest — HF Spaces Backend
Start command: gunicorn app:app --bind 0.0.0.0:7860 --timeout 120 --workers 1 --preload
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import Groq
from dotenv import load_dotenv
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backtest_engine import run_comparison

load_dotenv()

CSV_PATH    = "final_dataset.csv"
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.3-70b-versatile"
FAISS_PATH  = "faiss_index"
TOP_K       = 3
HYBRID_POOL = 10     # fetch 10 candidates from FAISS before reranking
SEMANTIC_W  = 0.6    # 60% weight on FAISS semantic score
TFIDF_W     = 0.4    # 40% weight on TF-IDF keyword score
PORT        = int(os.environ.get("PORT", 7860))

app = Flask(__name__)
CORS(app, origins=["*"])

print("Loading embedding model...", flush=True)
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

if os.path.exists(FAISS_PATH):
    print("Loading saved FAISS index...", flush=True)
    vector_db = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Index loaded — {len(df)} Q&A pairs.", flush=True)
else:
    print("Building FAISS index...", flush=True)
    df = pd.read_csv(CSV_PATH)
    documents = [Document(page_content=row["Answers"], metadata={"label": row["Label"], "question": row["Questions"]}) for _, row in df.iterrows()]
    splitter  = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks    = splitter.split_documents(documents)
    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.save_local(FAISS_PATH)
    print(f"Index built — {len(chunks)} chunks.", flush=True)

# Build TF-IDF matrix over all questions at startup (fast, <1s)
print("Building TF-IDF index...", flush=True)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tfidf_matrix     = tfidf_vectorizer.fit_transform(df["Questions"].fillna("").tolist())
print(f"TF-IDF ready — {tfidf_matrix.shape[0]} rows, {tfidf_matrix.shape[1]} features.", flush=True)

print("Flask app ready — routes active.", flush=True)


def hybrid_rerank(query, faiss_docs):
    """Rerank FAISS candidates using 60% semantic + 40% TF-IDF keyword score."""
    # Step 1 — deduplicate by source question, keep first occurrence
    seen        = set()
    unique_docs = []
    for doc in faiss_docs:
        key = doc.metadata.get("question", doc.page_content[:80])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    if not unique_docs:
        return faiss_docs

    # Step 2 — score each unique candidate
    query_tfidf = tfidf_vectorizer.transform([query])
    scored      = []
    for rank, doc in enumerate(unique_docs):
        semantic_score  = 1.0 - (rank / len(unique_docs))
        source_question = doc.metadata.get("question", doc.page_content)
        doc_tfidf       = tfidf_vectorizer.transform([source_question])
        tfidf_score     = float(cosine_similarity(query_tfidf, doc_tfidf)[0][0])
        final_score     = SEMANTIC_W * semantic_score + TFIDF_W * tfidf_score
        scored.append((final_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:TOP_K]]


@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Livermore ChatBot API is running."})


@app.route("/stats")
def stats():
    return jsonify({
        "total_pairs": len(df),
        "labels":      df["Label"].value_counts().to_dict(),
        "chunks":      vector_db.index.ntotal,
        "model":       GROQ_MODEL,
        "retrieval":   f"Hybrid (FAISS {int(SEMANTIC_W*100)}% + TF-IDF {int(TFIDF_W*100)}%)"
    })


@app.route("/ask", methods=["POST"])
def ask():
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    api_key = os.environ.get("GROQ_API_KEY") or (data.get("api_key") or "").strip()
    if not api_key:
        return jsonify({"error": "Groq API key not configured."}), 400

    # Hybrid retrieval: fetch HYBRID_POOL from FAISS, rerank to TOP_K
    candidates = vector_db.similarity_search(query, k=HYBRID_POOL)
    retrieved  = hybrid_rerank(query, candidates)

    context = "\n\n".join([doc.page_content for doc in retrieved])
    labels  = list({doc.metadata.get("label", "") for doc in retrieved})
    prompt  = f"""You are a Jesse Livermore trading expert chatbot trained on "Reminiscences of a Stock Operator".
Answer using only the context below. Speak in first person as Livermore. Be concise and direct.

Context:
{context}

Question: {query}
Answer:"""
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4
    )
    return jsonify({
        "answer":  response.choices[0].message.content.strip(),
        "context": context,
        "labels":  labels
    })


@app.route("/backtest", methods=["POST"])
def backtest():
    data       = request.get_json(force=True)
    symbol     = (data.get("symbol") or "GOOGL").upper().strip()
    start_date = data.get("start_date", "2020-01-01")
    end_date   = data.get("end_date",   "2025-12-31")
    try:
        stock = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if stock.empty:
            return jsonify({"error": f"No data found for {symbol}."}), 400
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)

        d = stock[["Close"]].copy()
        d["50MA"]   = d["Close"].rolling(50).mean()
        d["200MA"]  = d["Close"].rolling(200).mean()
        d["20High"] = d["Close"].rolling(20).max()
        d["20Low"]  = d["Close"].rolling(20).min()

        # 3-state signal: 1 = enter long, -1 = exit to flat, 0 = hold current
        d["Signal"] = 0
        d.loc[
            (d["Close"] > d["20High"].shift(1)) &
            (d["Close"] > d["50MA"]) &
            (d["Close"] > d["200MA"]),
            "Signal"
        ] = 1
        d.loc[d["Close"] < d["20Low"].shift(1), "Signal"] = -1

        # Explicit state machine — hold position until exit signal fires
        pos = 0
        positions = []
        for sig in d["Signal"]:
            if sig == 1:
                pos = 1
            elif sig == -1:
                pos = 0
            positions.append(pos)
        d["Position"] = positions

        # Correct compounding with shift(1) for realistic next-day execution
        d["Daily Return"]     = d["Close"].pct_change()
        d["Strategy Return"]  = d["Daily Return"] * d["Position"].shift(1)

        d["BH Cumulative"]    = (1 + d["Daily Return"]).cumprod() - 1
        d["Strat Cumulative"] = (1 + d["Strategy Return"]).cumprod() - 1

        dc          = d.dropna(subset=["BH Cumulative", "Strat Cumulative"])
        bh_total    = float(dc["BH Cumulative"].iloc[-1])
        strat_total = float(dc["Strat Cumulative"].iloc[-1])
        trade_count = int(pd.Series(positions).diff().fillna(0).ne(0).sum())

        step = max(1, len(dc) // 200)
        ds   = dc.iloc[::step]

        return jsonify({
            "symbol":          symbol,
            "start_date":      start_date,
            "end_date":        end_date,
            "bh_return":       round(bh_total * 100, 2),
            "strategy_return": round(strat_total * 100, 2),
            "outperformance":  round((strat_total - bh_total) * 100, 2),
            "trade_count":     trade_count,
            "dates":           ds.index.strftime("%Y-%m-%d").tolist(),
            "bh_series":       [round(v * 100, 2) for v in ds["BH Cumulative"].tolist()],
            "strategy_series": [round(v * 100, 2) for v in ds["Strat Cumulative"].tolist()],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare", methods=["GET"])
def compare():
    start  = request.args.get("start", "2020-01-01")
    end    = request.args.get("end",   "2025-01-01")
    result = run_comparison(start, end)
    return jsonify(result)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)