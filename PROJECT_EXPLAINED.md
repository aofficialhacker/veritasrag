# 📖 VeritasRAG — Explained in Plain English

A simple guide to what this project is, what happens inside it, and what we
demonstrate. Useful for understanding the project and for viva preparation.

---

## The problem we're solving

A normal AI chatbot (like ChatGPT) answers from **memory**. That causes two problems:

1. It **makes things up** confidently — this is called *hallucination*.
2. It **doesn't know your private documents** (your college rules, a company's manuals, etc.).

Think of it like a student taking a **closed-book exam** — they answer from
whatever they remember, and sometimes they bluff.

---

## The idea: RAG = giving the AI an "open book"

**RAG (Retrieval-Augmented Generation)** turns it into an **open-book exam**:

> Before answering, the system **looks up** the most relevant pages from *your*
> documents, hands them to the AI, and says *"answer using ONLY these pages, and
> cite them."*

Now the AI can't bluff — every answer is backed by a real passage you can check.
That is the whole point: **verify, don't trust.**

---

## What our project actually does — step by step

When you upload documents and ask a question, this happens:

1. **📄 Ingest** — reads your PDF/text files.
2. **✂️ Chunk** — cuts them into small overlapping pieces (so we can fetch just
   the relevant bit, not a whole 50-page PDF).
3. **🧠 Index** — converts every piece into numbers (*embeddings*) so the computer
   can measure "meaning similarity," and also builds a keyword index. Two search
   engines in one.
4. **🔍 Retrieve** — when you ask something, it finds the best matching pieces
   using **two methods at once**:
   - **BM25** = keyword matching (good for exact words/names)
   - **Dense** = meaning matching (good for paraphrases)
5. **🔀 Fuse (RRF)** — merges both lists into one smarter ranking.
6. **🎯 Rerank** — a precise model double-checks the top results and reorders
   them, so the *very best* passages go to the AI.
7. **💬 Generate** — the AI (Gemini or your local Ollama) writes the answer
   **using only those passages**, adding `[1] [2]` citations.
8. **📊 Evaluate** — we score how good the answers are automatically.

That whole chain is exactly what the research paper describes — we **built** the
thing the paper only *talks about*.

---

## What each tab demonstrates

| Tab | What it proves | The "wow" moment |
|-----|----------------|------------------|
| **💬 Ask** | Grounded answers with clickable citations | Toggle **RAG OFF** → the AI hallucinates/gets vague. Toggle **ON** → correct + cited. Side-by-side proof RAG works. |
| **🔍 Retrieval Comparison** | Why *hybrid + rerank* wins | Same question through 4 methods, with live charts — the paper's Table 1 / Figure 4, running for real. |
| **📊 Evaluation** | The system is measurably good | Faithfulness / Precision / Relevance scores + a radar chart — looks like a real research dashboard. |

---

## Why it looks impressive (but is actually simple)

- It's really ~300 lines of Python glue around free, pre-built models.
- But to a viewer it shows: two search engines, fusion math, a reranking stage,
  a switchable AI backend (cloud **and** offline), citations, *and* an evaluation
  dashboard.

### Two strong talking points for the viva

1. *"It runs 100% offline with a local model — no API, no cost, no data leaves
   the machine."*
2. *"Every answer is verifiable — click any citation to see the exact source. It
   literally can't make things up without being caught."*

---

## Key terms cheat-sheet

| Term | Simple meaning |
|------|----------------|
| **Embedding** | Turning text into a list of numbers that captures its meaning. |
| **Vector index (FAISS / HNSW)** | A super-fast way to find the most similar pieces of text. |
| **BM25** | Classic keyword search (matches exact words). |
| **Dense retrieval** | Meaning-based search (matches ideas, not just words). |
| **Hybrid + RRF** | Combining keyword + meaning search into one better ranking. |
| **Reranker (cross-encoder)** | A careful second-pass model that picks the truly best passages. |
| **Grounding** | Forcing the AI to answer only from the retrieved text. |
| **Hallucination** | When the AI confidently makes something up. |
| **Ragas metrics** | Automatic scores for how faithful/relevant an answer is. |

---

*Companion document to the project **VeritasRAG** and the research paper
"Retrieval-Augmented Generation Systems," MSc Information Technology.*
