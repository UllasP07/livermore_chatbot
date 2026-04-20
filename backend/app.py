"""
Jesse Livermore RAG ChatBot + Strategy Backtest — HF Spaces Backend
Start command: gunicorn app:app --bind 0.0.0.0:7860 --timeout 120 --workers 1 --preload
"""

import os
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

load_dotenv()

CSV_PATH    = "Livermore_QA_Dataset_Extended.csv"
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.3-70b-versatile"
FAISS_PATH  = "faiss_index"
TOP_K       = 3
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
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter  = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks    = splitter.split_documents(documents)
    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.save_local(FAISS_PATH)
    print(f"Index built — {len(chunks)} chunks.", flush=True)

print("Flask app ready — routes active.", flush=True)

@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Livermore ChatBot API is running."})

@app.route("/stats")
def stats():
    return jsonify({"total_pairs": len(df), "labels": df["Label"].value_counts().to_dict(), "chunks": vector_db.index.ntotal, "model": GROQ_MODEL})

@app.route("/ask", methods=["POST"])
def ask():
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    api_key = os.environ.get("GROQ_API_KEY") or (data.get("api_key") or "").strip()
    if not api_key:
        return jsonify({"error": "Groq API key not configured."}), 400
    retrieved = vector_db.similarity_search(query, k=TOP_K)
    context   = "\n\n".join([doc.page_content for doc in retrieved])
    labels    = list({doc.metadata.get("label", "") for doc in retrieved})
    prompt = f"""You are a Jesse Livermore trading expert chatbot trained on "Reminiscences of a Stock Operator".
Answer using only the context below. Speak in first person as Livermore. Be concise and direct.

Context:
{context}

Question: {query}
Answer:"""
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=400, temperature=0.4)
    return jsonify({"answer": response.choices[0].message.content.strip(), "context": context, "labels": labels})

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

        # Signal: 1 = long, 0 = flat (never short)
        d["Signal"] = 0
        d.loc[(d["Close"] > d["20High"].shift(1)) &
              (d["Close"] > d["50MA"]) &
              (d["Close"] > d["200MA"]), "Signal"] = 1
        d.loc[d["Close"] < d["20Low"].shift(1), "Signal"] = 0

        # Hold position until exit signal (forward fill)
        d["Position"] = d["Signal"].replace(0, pd.NA).ffill().fillna(0)

        # Correct compounding returns
        d["Daily Return"]     = d["Close"].pct_change()
        d["Strategy Return"]  = d["Daily Return"] * d["Position"].shift(1)

        d["BH Cumulative"]    = (1 + d["Daily Return"]).cumprod() - 1
        d["Strat Cumulative"] = (1 + d["Strategy Return"]).cumprod() - 1

        bh_total    = float(d["BH Cumulative"].dropna().iloc[-1])
        strat_total = float(d["Strat Cumulative"].dropna().iloc[-1])
        trade_count = int((d["Position"].diff().fillna(0) != 0).sum())

        dc   = d.dropna(subset=["BH Cumulative", "Strat Cumulative"])
        step = max(1, len(dc) // 200)
        ds   = dc.iloc[::step]

        return jsonify({
            "symbol": symbol, "start_date": start_date, "end_date": end_date,
            "bh_return":       round(bh_total * 100, 2),
            "strategy_return": round(strat_total * 100, 2),
            "outperformance":  round((strat_total - bh_total) * 100, 2),
            "trade_count":     trade_count,
            "dates":           ds.index.strftime("%Y-%m-%d").tolist(),
            "bh_series":       [round(v*100,2) for v in ds["BH Cumulative"].tolist()],
            "strategy_series": [round(v*100,2) for v in ds["Strat Cumulative"].tolist()],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)