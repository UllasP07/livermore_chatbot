"""
Part 1 : FAISS-only vs Hybrid retrieval comparison (96 questions)
Part 2 : LLM-as-Judge scoring on answer quality (20 questions)
Run    : python evaluate.py
Needs  : Flask server running at localhost:7860 (python app.py in another terminal)
Output : evaluation_comparison.csv
         evaluation_llm_judge.csv
         evaluation_summary.json
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH        = "Livermore_QA_Dataset_Extended.csv"
EMBED_MODEL     = "sentence-transformers/paraphrase-MiniLM-L6-v2"
FAISS_PATH      = "faiss_index"
GROQ_MODEL      = "llama-3.3-70b-versatile"
SERVER_URL      = "http://localhost:7860"

TOP_K           = 3
HYBRID_POOL     = 10
SEMANTIC_W      = 0.6
TFIDF_W         = 0.4
N_SAMPLES       = 96    # total questions for Part 1 (16 per label)
N_LLM_JUDGE     = 20   # subset for Part 2 (to save API calls)

OUT_COMPARISON  = "evaluation_comparison.csv"
OUT_LLM_JUDGE   = "evaluation_llm_judge.csv"
OUT_SUMMARY     = "evaluation_summary.json"

# ── Load everything ────────────────────────────────────────────────────────────
print("=" * 60, flush=True)
print("  Loading models and indexes...", flush=True)
print("=" * 60, flush=True)

embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
df              = pd.read_csv(CSV_PATH)

if os.path.exists(FAISS_PATH):
    print("Loading saved FAISS index...", flush=True)
    vector_db = FAISS.load_local(
        FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )
else:
    print("Building FAISS index...", flush=True)
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

print("Building TF-IDF index...", flush=True)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tfidf_vectorizer.fit_transform(df["Questions"].fillna("").tolist())
print("All indexes ready.\n", flush=True)


# ── Retrieval helpers ──────────────────────────────────────────────────────────
def faiss_only_retrieve(query):
    """Pure FAISS retrieval — no TF-IDF."""
    return vector_db.similarity_search(query, k=TOP_K)


def hybrid_retrieve(query):
    """Hybrid FAISS + TF-IDF retrieval with deduplication."""
    candidates  = vector_db.similarity_search(query, k=HYBRID_POOL)
    query_tfidf = tfidf_vectorizer.transform([query])

    seen        = set()
    unique_docs = []
    for doc in candidates:
        key = doc.metadata.get("question", doc.page_content[:80])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    if not unique_docs:
        return candidates[:TOP_K]

    scored = []
    for rank, doc in enumerate(unique_docs):
        semantic_score  = 1.0 - (rank / len(unique_docs))
        source_question = doc.metadata.get("question", doc.page_content)
        doc_tfidf       = tfidf_vectorizer.transform([source_question])
        tfidf_score     = float(cosine_similarity(query_tfidf, doc_tfidf)[0][0])
        final_score     = SEMANTIC_W * semantic_score + TFIDF_W * tfidf_score
        scored.append((final_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:TOP_K]]


# ── Metric helpers ─────────────────────────────────────────────────────────────
def get_cosine_similarity(text_a, text_b):
    vec_a = np.array(embedding_model.embed_query(text_a)).reshape(1, -1)
    vec_b = np.array(embedding_model.embed_query(text_b)).reshape(1, -1)
    return float(cosine_similarity(vec_a, vec_b)[0][0])


def get_label_accuracy(retrieved_docs, correct_label):
    retrieved_labels = [doc.metadata.get("label", "") for doc in retrieved_docs]
    return int(correct_label in retrieved_labels)


# ── Sample test questions ──────────────────────────────────────────────────────
per_label    = N_SAMPLES // df["Label"].nunique()
test_samples = pd.concat([
    group.sample(min(per_label, len(group)), random_state=42)
    for _, group in df.groupby("Label")
]).reset_index(drop=True)

print(f"Test set: {len(test_samples)} questions across {df['Label'].nunique()} labels")
print(test_samples["Label"].value_counts().to_string(), "\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — FAISS only vs Hybrid comparison
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  PART 1 — FAISS vs Hybrid Comparison")
print("=" * 60)

comparison_results = []

for i, row in test_samples.iterrows():
    question      = row["Questions"]
    ground_truth  = row["Answers"]
    correct_label = row["Label"]

    # FAISS only
    faiss_docs      = faiss_only_retrieve(question)
    faiss_text      = " ".join([doc.page_content for doc in faiss_docs])
    faiss_cosine    = get_cosine_similarity(faiss_text, ground_truth)
    faiss_label_acc = get_label_accuracy(faiss_docs, correct_label)

    # Hybrid
    hybrid_docs      = hybrid_retrieve(question)
    hybrid_text      = " ".join([doc.page_content for doc in hybrid_docs])
    hybrid_cosine    = get_cosine_similarity(hybrid_text, ground_truth)
    hybrid_label_acc = get_label_accuracy(hybrid_docs, correct_label)

    # Did hybrid improve over FAISS?
    cosine_delta = round(hybrid_cosine - faiss_cosine, 4)
    hybrid_won   = int(hybrid_cosine > faiss_cosine)

    comparison_results.append({
        "question":           question[:80],
        "correct_label":      correct_label,
        "faiss_cosine":       round(faiss_cosine, 4),
        "hybrid_cosine":      round(hybrid_cosine, 4),
        "cosine_delta":       cosine_delta,
        "hybrid_won":         hybrid_won,
        "faiss_label_acc":    faiss_label_acc,
        "hybrid_label_acc":   hybrid_label_acc,
    })

    count = len(comparison_results)
    if count % 10 == 0:
        subset        = comparison_results
        avg_faiss     = np.mean([r["faiss_cosine"] for r in subset])
        avg_hybrid    = np.mean([r["hybrid_cosine"] for r in subset])
        hybrid_wins   = np.mean([r["hybrid_won"] for r in subset]) * 100
        print(f"  {count:>3}/{len(test_samples)} | "
              f"FAISS cosine: {avg_faiss:.4f} | "
              f"Hybrid cosine: {avg_hybrid:.4f} | "
              f"Hybrid better: {hybrid_wins:.1f}%")

print("-" * 60)

# Save comparison CSV
comp_df = pd.DataFrame(comparison_results)
comp_df.to_csv(OUT_COMPARISON, index=False)
print(f"Comparison results saved → {OUT_COMPARISON}\n")

# Compute Part 1 summary
avg_faiss_cosine     = float(comp_df["faiss_cosine"].mean())
avg_hybrid_cosine    = float(comp_df["hybrid_cosine"].mean())
avg_faiss_label      = float(comp_df["faiss_label_acc"].mean() * 100)
avg_hybrid_label     = float(comp_df["hybrid_label_acc"].mean() * 100)
hybrid_wins_pct      = float(comp_df["hybrid_won"].mean() * 100)
avg_cosine_delta     = float(comp_df["cosine_delta"].mean())

print("  PART 1 RESULTS")
print(f"  {'Metric':<30} {'FAISS':>10} {'Hybrid':>10} {'Delta':>10}")
print(f"  {'-'*60}")
print(f"  {'Avg Cosine Similarity':<30} {avg_faiss_cosine:>10.4f} {avg_hybrid_cosine:>10.4f} {avg_cosine_delta:>+10.4f}")
print(f"  {'Label Accuracy %':<30} {avg_faiss_label:>10.1f} {avg_hybrid_label:>10.1f} {avg_hybrid_label - avg_faiss_label:>+10.1f}")
print(f"  {'Hybrid beats FAISS':<30} {'':>10} {hybrid_wins_pct:>10.1f}%")

# Per label delta
print("\n  Per-label cosine delta (Hybrid - FAISS):")
for label, group in comp_df.groupby("correct_label"):
    delta = group["cosine_delta"].mean()
    bar   = "▲" if delta > 0 else "▼"
    print(f"  {bar} {label:<28} {delta:+.4f}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — LLM as Judge
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  PART 2 — LLM-as-Judge Answer Quality Scoring")
print(f"  (20 questions sampled evenly across labels)")
print("=" * 60)

# Check server is running
try:
    r = requests.get(f"{SERVER_URL}/", timeout=5)
    assert r.status_code == 200
    print("Flask server reachable ✓\n", flush=True)
except Exception:
    print("ERROR: Flask server not reachable at localhost:7860.")
    print("Please run 'python app.py' in another terminal and try again.")
    exit(1)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Sample 20 questions evenly across labels for LLM judge
n_per_label_judge = N_LLM_JUDGE // df["Label"].nunique()
judge_samples     = pd.concat([
    group.sample(min(n_per_label_judge, len(group)), random_state=99)
    for _, group in test_samples.groupby("Label")
]).reset_index(drop=True)

JUDGE_PROMPT = """You are an expert evaluator assessing a Jesse Livermore trading chatbot.

You will be given:
- A question asked by a user
- The ground truth answer (correct answer from the dataset)
- The chatbot's generated answer

Score the chatbot's answer on THREE criteria, each from 0 to 10:

1. RELEVANCE (0-10): Does the answer directly address what was asked?
   0 = completely off-topic, 10 = perfectly on-point

2. FAITHFULNESS (0-10): Is the answer grounded in Livermore's actual philosophy?
   0 = hallucinated or contradicts Livermore, 10 = fully faithful to his principles

3. CLARITY (0-10): Is the answer clear, concise, and well-structured?
   0 = confusing or rambling, 10 = perfectly clear and direct

Respond ONLY with a JSON object in this exact format, nothing else:
{{"relevance": <score>, "faithfulness": <score>, "clarity": <score>, "reasoning": "<one sentence>"}}

Question: {question}
Ground Truth: {ground_truth}
Chatbot Answer: {chatbot_answer}"""

llm_judge_results = []

for i, row in judge_samples.iterrows():
    question     = row["Questions"]
    ground_truth = row["Answers"]
    label        = row["Label"]

    print(f"  [{len(llm_judge_results)+1:>2}/{len(judge_samples)}] {question[:70]}...")

    # Step 1 — get chatbot answer from Flask server
    try:
        resp = requests.post(
            f"{SERVER_URL}/ask",
            json={"query": question},
            timeout=30
        )
        chatbot_answer = resp.json().get("answer", "")
    except Exception as e:
        print(f"    ERROR getting answer: {e}")
        chatbot_answer = ""

    if not chatbot_answer:
        print("    Skipping — no answer returned.")
        continue

    # Step 2 — send to LLM judge
    try:
        judge_response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    ground_truth=ground_truth[:400],
                    chatbot_answer=chatbot_answer[:400]
                )
            }],
            max_tokens=200,
            temperature=0.0   # deterministic scoring
        )
        raw = judge_response.choices[0].message.content.strip()
        # strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(raw)
    except Exception as e:
        print(f"    ERROR parsing judge response: {e}")
        scores = {"relevance": None, "faithfulness": None, "clarity": None, "reasoning": "parse error"}

    overall = None
    if all(scores.get(k) is not None for k in ["relevance", "faithfulness", "clarity"]):
        overall = round((scores["relevance"] + scores["faithfulness"] + scores["clarity"]) / 3, 2)

    print(f"    Relevance: {scores.get('relevance')}  "
          f"Faithfulness: {scores.get('faithfulness')}  "
          f"Clarity: {scores.get('clarity')}  "
          f"Overall: {overall}")

    llm_judge_results.append({
        "question":        question[:80],
        "label":           label,
        "chatbot_answer":  chatbot_answer[:150],
        "relevance":       scores.get("relevance"),
        "faithfulness":    scores.get("faithfulness"),
        "clarity":         scores.get("clarity"),
        "overall":         overall,
        "reasoning":       scores.get("reasoning", ""),
    })

    time.sleep(0.5)   # small delay to avoid rate limiting

print("-" * 60)

# Save LLM judge CSV
judge_df = pd.DataFrame(llm_judge_results)
judge_df.to_csv(OUT_LLM_JUDGE, index=False)
print(f"LLM judge results saved → {OUT_LLM_JUDGE}\n")

# Compute Part 2 summary
valid_judge = judge_df.dropna(subset=["overall"])
avg_relevance    = float(valid_judge["relevance"].mean())
avg_faithfulness = float(valid_judge["faithfulness"].mean())
avg_clarity      = float(valid_judge["clarity"].mean())
avg_overall      = float(valid_judge["overall"].mean())

print("  PART 2 RESULTS")
print(f"  Questions scored      : {len(valid_judge)}/{len(judge_samples)}")
print(f"  Avg Relevance         : {avg_relevance:.2f} / 10")
print(f"  Avg Faithfulness      : {avg_faithfulness:.2f} / 10")
print(f"  Avg Clarity           : {avg_clarity:.2f} / 10")
print(f"  Avg Overall Score     : {avg_overall:.2f} / 10")

# Per label LLM judge
print("\n  Per-label overall score:")
for label, group in valid_judge.groupby("label"):
    print(f"  {label:<28} {group['overall'].mean():.2f} / 10")


# ══════════════════════════════════════════════════════════════════════════════
# Final Summary JSON
# ══════════════════════════════════════════════════════════════════════════════
summary = {
    "evaluation_mode": "FAISS vs Hybrid + LLM-as-Judge",
    "total_questions_part1": len(comp_df),
    "total_questions_part2": len(valid_judge),
    "part1_retrieval_comparison": {
        "faiss_only": {
            "avg_cosine_similarity": round(avg_faiss_cosine, 4),
            "label_accuracy_pct":   round(avg_faiss_label, 1),
        },
        "hybrid": {
            "avg_cosine_similarity": round(avg_hybrid_cosine, 4),
            "label_accuracy_pct":   round(avg_hybrid_label, 1),
        },
        "hybrid_beats_faiss_pct":   round(hybrid_wins_pct, 1),
        "avg_cosine_improvement":   round(avg_cosine_delta, 4),
        "per_label_cosine_delta": {
            label: round(float(group["cosine_delta"].mean()), 4)
            for label, group in comp_df.groupby("correct_label")
        }
    },
    "part2_llm_judge": {
        "avg_relevance":    round(avg_relevance, 2),
        "avg_faithfulness": round(avg_faithfulness, 2),
        "avg_clarity":      round(avg_clarity, 2),
        "avg_overall":      round(avg_overall, 2),
        "per_label_overall": {
            label: round(float(group["overall"].mean()), 2)
            for label, group in valid_judge.groupby("label")
        }
    }
}

with open(OUT_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nFull summary saved → {OUT_SUMMARY}")

print("\n" + "=" * 60)
print("  EVALUATION COMPLETE")
print("=" * 60)