# VeritasRAG 🔎

**A Hybrid Retrieval-Augmented Generation System with Citations and Evaluation**

VeritasRAG is a working implementation of the RAG pipeline described in the
accompanying research paper *"Retrieval-Augmented Generation Systems"*. It takes
your documents, indexes them, and answers questions **grounded in evidence**,
with inline `[n]` citations back to the exact source passage — so answers can be
verified instead of trusted blindly.

It implements the paper's recommended "best" configuration end-to-end:

> **BM25 (sparse) + Dense embeddings → Reciprocal Rank Fusion → Cross-encoder rerank → grounded LLM answer with citations.**

---

## ✨ Features

| Tab | What it does |
|-----|--------------|
| **💬 Ask** | Chat over your documents. Every answer carries `[n]` citations you can expand to see the source. A **RAG on/off toggle** demonstrates hallucination when grounding is removed. |
| **🔍 Retrieval Comparison** | Runs the same query through **BM25 vs Dense vs Hybrid vs Hybrid+Rerank** and charts their latency and results — a live version of Table 1 / Figure 4 of the paper. |
| **📊 Evaluation** | Computes **Ragas-style** metrics locally (context precision, faithfulness, answer relevance) over an evaluation set and plots a quality radar. |

## 🧱 Architecture (maps 1:1 to the research paper)

```
                 ┌── Ingestion (PDF/TXT) ──► Chunking ──► Embeddings ──► FAISS HNSW index
   Offline       └──────────────────────────────────────► BM25 index
   ─────────────────────────────────────────────────────────────────────────────────────
   Online   Query ─► [ Dense search + BM25 ] ─► RRF fusion ─► Cross-encoder rerank
                                                                     │
                                                   top-k passages ───┤
                                                                     ▼
                            Prompt builder (instruction-hardened, delimited)
                                                                     ▼
                              Generator: Gemini / Ollama / Extractive
                                                                     ▼
                                    Answer with [n] citations
```

| Phase | Paper § | Module |
|-------|---------|--------|
| 1 Ingestion | 4.3 | `src/ingestion.py` |
| 2 Indexing | 4.5 | `src/chunking.py`, `src/vector_store.py` (HNSW) |
| 3 Retrieval | 4.4 | `src/sparse.py`, `src/embeddings.py`, `src/fusion.py`, `src/reranker.py`, `src/retriever.py` |
| 4 Augmentation | 5.2 / 5.6 | `src/prompt_builder.py` |
| 5 Generation | 4.3 | `src/generator.py` |
| 6 Evaluation | 5.7 | `src/evaluation.py` |

## 🛠️ Tech Stack (free / local)

- **UI:** Streamlit
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local)
- **Vector index:** FAISS (HNSW graph)
- **Sparse retrieval:** `rank-bm25`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)
- **Generator:** Gemini 2.5 Flash *(default)* / Ollama *(local)* / Extractive *(no LLM)*

---

## 🚀 Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the generator (see .env)
#    Default backend is Gemini; a key is already in .env.
```

### `.env` configuration

```
GENERATOR_BACKEND=gemini        # gemini | ollama | extractive
GEMINI_API_KEY=...              # your Google AI Studio key
GEMINI_MODEL=gemini-2.5-flash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
```

### (Optional) Run fully offline with Ollama

1. Install Ollama: https://ollama.com/download
2. Pull a model: `ollama pull llama3.2:1b`
3. In the sidebar, switch the backend to **Ollama** — no API key, no internet.

---

## ▶️ Run

```bash
streamlit run app.py
```

Then in the sidebar: click **Load sample** (uses the bundled TechNova docs) or
upload your own PDFs, click **Build index**, and start asking questions.

### Try these (with the sample corpus)

- *"What is the refund policy for new StreamFlow customers?"*
- *"How many days of paid annual leave do employees get?"*
- Toggle **RAG off** and ask the same — watch it hallucinate. 😉

## ✅ Smoke test

```bash
python smoke_test.py
```

Verifies ingestion → retrieval (all 4 strategies) → generation → evaluation.

---

## 📁 Project structure

```
veritasrag/
├── app.py                  # Streamlit UI (Ask / Compare / Evaluate)
├── requirements.txt
├── .env                    # backend config + API key (git-ignored)
├── smoke_test.py
├── data/
│   ├── sample_docs/        # bundled demo corpus
│   └── eval_set.json       # evaluation questions
└── src/
    ├── config.py           # all tunable parameters
    ├── ingestion.py        # Phase 1
    ├── chunking.py         # Phase 2
    ├── embeddings.py       # dense vectors
    ├── vector_store.py     # FAISS HNSW
    ├── sparse.py           # BM25
    ├── fusion.py           # Reciprocal Rank Fusion
    ├── reranker.py         # cross-encoder
    ├── retriever.py        # orchestrates all strategies
    ├── prompt_builder.py   # Phase 4 (grounding + injection defense)
    ├── generator.py        # Phase 5 (Gemini / Ollama / extractive)
    └── evaluation.py       # Phase 6 (Ragas-style metrics)
```

---

*Built as the semester project accompanying the research paper "Retrieval-Augmented
Generation Systems" — MSc Information Technology.*
