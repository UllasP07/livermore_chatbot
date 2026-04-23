# Jesse Livermore — GenAI Trading ChatBot & Strategy Backtest

> RAG-powered chatbot trained on **15,000 Q&A pairs** from *Reminiscences of a Stock Operator* (1923) and *How to Trade in Stocks* (1940), with an integrated Livermore breakout strategy backtester.

---

## What This Project Does

| Part | Description |
|---|---|
| **Part I — ChatBot** | Ask questions about Livermore's trading philosophy, psychology, and strategy. Answers are retrieved from a curated Q&A dataset using FAISS + LangChain, then synthesized by Llama 3.3 70B via Groq. |
| **Part I — Stock Models** | RNN & LSTM models fine-tuned on GOOGL stock data (2020–2024 train, 2025 test). |
| **Part II — Backtest** | Livermore's breakout strategy backtested on the Magnificent 7 stocks (2020–2025). Entry on 20-day high breakout confirmed by 50MA & 200MA. Exit on 20-day low break. |

---

## Live Demo

| Service | URL |
|---|---|
| Frontend (Netlify) | `https://your-site.netlify.app` |
| Backend (HF Spaces) | `https://Ullas07-livermore-chatbot.hf.space` |

---

## Repo Structure

```
livermore_chatbot/
├── app.py                          ← Flask RAG API + backtest route
├── Dockerfile                      ← HF Spaces Docker config
├── requirements.txt
├── Livermore_QA_Dataset_15k.csv    ← 15,000 Q&A pairs (ADD THIS)
├── faiss_index/                    ← Pre-built FAISS index (ADD THIS)
│   ├── index.faiss
│   └── index.pkl
│
└── frontend/                       ← Deploy to Netlify
    ├── index.html                  ← Full chat + backtest UI
    └── netlify.toml
```

> ⚠️ All backend files must be at **repo root level** (not inside a `backend/` subfolder).
> HF Spaces serves from root — nested folders will cause import errors.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dataset | 15,000 hand-crafted Q&A pairs · 6 labels · CSV |
| Embeddings | `sentence-transformers/paraphrase-MiniLM-L6-v2` |
| Vector DB | FAISS (pre-built index, loaded from disk) |
| RAG Framework | LangChain 0.2+ |
| LLM | Llama 3.3 70B via Groq API (`llama-3.3-70b-versatile`) |
| Backend | Flask + Gunicorn |
| Backend Hosting | Hugging Face Spaces (Docker, CPU Basic, 16 GB RAM) |
| Frontend | Vanilla HTML / CSS / JS + Chart.js |
| Frontend Hosting | Netlify (free tier, global CDN) |
| Stock Data | yfinance |
| ML Models | TensorFlow / Keras (RNN + LSTM) |

---

## Deployment

### Prerequisites

- GitHub account
- Hugging Face account (free) — [huggingface.co](https://huggingface.co)
- Netlify account (free) — [netlify.com](https://netlify.com)
- Groq API key (free) — [console.groq.com](https://console.groq.com)

---

### Step 1 — Build the FAISS Index (Google Colab)

The FAISS index must be pre-built and committed to the repo.
Running it on HF Spaces at startup would take 3+ minutes and likely OOM on free tier.

```python
# Run this in Google Colab
!pip install faiss-cpu sentence-transformers langchain langchain-huggingface langchain-text-splitters -q

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

df = pd.read_csv("Livermore_QA_Dataset_15k.csv")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
)

documents = [
    Document(
        page_content=row["Answers"],
        metadata={"label": row["Label"], "question": row["Questions"]}
    )
    for _, row in df.iterrows()
]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")  # should be ~23,000+

vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("faiss_index")

# Download the index
import shutil
shutil.make_archive("faiss_index", "zip", "faiss_index")
from google.colab import files
files.download("faiss_index.zip")
```

Unzip and place both files at repo root:
```
faiss_index/index.faiss
faiss_index/index.pkl
```

---

### Step 2 — Push to GitHub

```bash
git init
git add .
git commit -m "Jesse Livermore ChatBot — initial commit"
git remote add origin https://github.com/YOUR_USERNAME/livermore_chatbot.git
git push -u origin main
```

---

### Step 3 — Deploy Backend → Hugging Face Spaces

> ⚠️ **Do NOT use Render free tier.** sentence-transformers requires ~500 MB RAM.
> Render's free tier has a 512 MB hard cap and will crash on startup.
> HF Spaces CPU Basic gives 16 GB RAM and works reliably.

1. Go to [huggingface.co](https://huggingface.co) → **New Space**
2. Set:
   - **Space name:** `livermore-chatbot`
   - **SDK:** Docker
   - **Visibility:** Public
3. In Space **Settings → Variables and secrets**, add:
   - Secret name: `GROQ_API_KEY`
   - Value: your key from [console.groq.com](https://console.groq.com)
4. In Space **Settings → Repository**, link your GitHub repo
5. HF Spaces will build and deploy automatically

Your backend URL will be:
```
https://YOUR_USERNAME-livermore-chatbot.hf.space
```

**Dockerfile** (already in repo root):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "--preload"]
```

> ℹ️ Free HF Spaces pause after 48 hours of inactivity.
> Go to Space **Settings → Resume** before your presentation.

---

### Step 4 — Deploy Frontend → Netlify

1. Go to [netlify.com](https://netlify.com) → **Add new site → Import from Git**
2. Connect your GitHub repo
3. Set:
   - **Base directory:** `frontend`
   - **Publish directory:** `frontend`
   - Leave build command empty
4. Click **Deploy**
5. Your frontend is live at `https://your-site-name.netlify.app`

Netlify auto-deploys every time you push to `main`.

---

### Step 5 — Use the ChatBot

1. Open your Netlify URL
2. Paste your **HF Spaces backend URL** in the top field
3. Type a question — e.g. *"How did Livermore cut his losses?"*
4. Click **Strategy Backtest** tab → select a stock → **Run Backtest**

---

## API Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/stats` | GET | Returns total pairs, chunk count, labels |
| `/ask` | POST | RAG query → LLM answer |
| `/backtest` | POST | Livermore breakout strategy backtest |

### `/ask` request body
```json
{ "query": "What is the line of least resistance?" }
```

### `/backtest` request body
```json
{
  "symbol": "GOOGL",
  "start_date": "2020-01-01",
  "end_date": "2025-12-31"
}
```

---

## Backtest Strategy Logic

**Entry signal:** `Close > 20-day high` AND `Close > 50MA` AND `Close > 200MA`

**Exit signal:** `Close < 20-day low`

**Position:** Long only (never short). Holds until exit signal fires.

```python
# 3-state signal: 1=enter, -1=exit, 0=hold current position
d["Signal"] = 0
d.loc[(d["Close"] > d["20High"].shift(1)) &
      (d["Close"] > d["50MA"]) &
      (d["Close"] > d["200MA"]), "Signal"] = 1
d.loc[d["Close"] < d["20Low"].shift(1), "Signal"] = -1

# State machine — hold until explicit exit
pos = 0
positions = []
for sig in d["Signal"]:
    if sig == 1:   pos = 1
    elif sig == -1: pos = 0
    positions.append(pos)
```

Returns are calculated using `cumprod` (correct multiplicative compounding).

---

## Dataset

| Stat | Value |
|---|---|
| Total Q&A pairs | 15,000 |
| Per label | 2,500 |
| Labels | Personal Life, Strategy Development, Timing, Risk Management, Adaptability, Psychology |
| Columns | `Questions`, `Answers`, `Label` |
| Primary source | *Reminiscences of a Stock Operator* (Lefèvre, 1923) |
| Supplementary | *How to Trade in Stocks* (Livermore, 1940) |

---

## Local Development

```bash
# Backend
pip install -r requirements.txt
python app.py
# → http://localhost:7860

# Frontend
# Open frontend/index.html in browser
# Set backend URL to http://localhost:7860
```

Or run with Gunicorn:
```bash
gunicorn app:app --bind 0.0.0.0:7860 --timeout 120 --workers 1 --preload
```

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| OOM crash on startup | Render free tier only 512 MB | Use HF Spaces (16 GB) |
| `KeyError: 'Close'` on yfinance | yfinance 0.2+ returns MultiIndex columns | `stock.columns = stock.columns.get_level_values(0)` |
| FAISS index slow to build | Building 23k embeddings takes 3+ minutes | Pre-build in Colab, commit to repo |
| `ImportError: HuggingFaceEmbeddings` | LangChain 0.2+ split modules | Use `langchain_huggingface` not `langchain_community` |
| Model not found on Groq | `llama3-8b-8192` decommissioned | Use `llama-3.3-70b-versatile` |
| Backtest Total Trades = 1 | Exit signal coded as `0` same as no-signal, `ffill` swallowed all exits | Use 3-state signal: `1`, `-1`, `0` with explicit state machine |

---

## Project Structure — Northeastern University GenAI Final Project

```
CSYE 7380 — Generative AI  ·  Spring 2026  ·  Team Livermore
```