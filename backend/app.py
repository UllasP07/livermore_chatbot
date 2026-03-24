import os, sys
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

CSV_PATH    = "Livermore_QA_Dataset_Extended.csv"
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.3-70b-versatile"
FAISS_PATH  = "faiss_index"   # folder where index is saved
TOP_K       = 3
PORT        = int(os.environ.get("PORT", 10000))

app = Flask(__name__)
CORS(app, origins=["*"])

# ── Load embedding model ──────────────────────────────────────────────────────
print("Loading embedding model...", flush=True)
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ── Load or build FAISS index ─────────────────────────────────────────────────
if os.path.exists(FAISS_PATH):
    print("Loading saved FAISS index...", flush=True)
    vector_db = FAISS.load_local(FAISS_PATH, embedding_model,
                                 allow_dangerous_deserialization=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Index loaded — {len(df)} Q&A pairs.", flush=True)
else:
    print("Building FAISS index for first time...", flush=True)
    df = pd.read_csv(CSV_PATH)
    documents = [
        Document(
            page_content=row["Answers"],
            metadata={"label": row["Label"], "question": row["Questions"]}
        )
        for _, row in df.iterrows()
    ]
    splitter  = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks    = splitter.split_documents(documents)
    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.save_local(FAISS_PATH)
    print(f"Index built and saved — {len(chunks)} chunks.", flush=True)

print("Flask app ready.", flush=True)