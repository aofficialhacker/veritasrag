"""
VeritasRAG - Hybrid Retrieval-Augmented Generation with Citations & Evaluation
==============================================================================
Streamlit front-end that ties the six pipeline phases together into a single
demo-friendly application:

    Chat              - grounded Q&A with inline citations and a with/without-RAG
                        toggle that demonstrates hallucination.
    Retrieval Compare - BM25 vs Dense vs Hybrid vs Hybrid+Rerank, side by side,
                        reproducing Table 1 / Figure 4 of the research paper.
    Evaluation        - Ragas-style faithfulness / relevance / precision metrics.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.chunking import chunk_pages
from src.config import AVAILABLE_BACKENDS, EVAL_SET_PATH, GENERATOR, RETRIEVAL, SAMPLE_DOCS_DIR
from src.evaluation import RagEvaluator, aggregate
from src.generator import generate, health_check
from src.ingestion import load_documents
from src.retriever import HybridRetriever

STRATEGY_LABELS = {
    "sparse": "BM25 (Sparse)",
    "dense": "Dense (HNSW)",
    "hybrid": "Hybrid (RRF)",
    "rerank": "Hybrid + Rerank",
}

st.set_page_config(page_title="VeritasRAG", page_icon="🔎", layout="wide")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---- global ---- */
.stApp { background: linear-gradient(180deg, #fbfbff 0%, #f3f2fd 100%); }
.block-container { padding-top: 1.4rem; }

/* ---- hero banner ---- */
.vr-hero {
    background: linear-gradient(120deg, #6C5CE7 0%, #8b5cf6 45%, #a855f7 100%);
    border-radius: 20px;
    padding: 26px 32px;
    color: #fff;
    box-shadow: 0 12px 30px rgba(108, 92, 231, 0.28);
    margin-bottom: 18px;
}
.vr-hero-title { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; margin: 0; }
.vr-hero-sub { font-size: 1.02rem; opacity: 0.95; margin-top: 6px; font-weight: 400; }
.vr-hero-tags { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.vr-hero-tags span {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    backdrop-filter: blur(4px);
}

/* ---- metric cards ---- */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #ebe9fb;
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 4px 14px rgba(108, 92, 231, 0.08);
}
[data-testid="stMetricValue"] { color: #6C5CE7; font-weight: 800; }

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 12px; padding: 8px 18px; font-weight: 600;
    background: #ffffff; border: 1px solid #ecebfa;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(120deg, #6C5CE7, #8b5cf6) !important;
    color: #ffffff !important; border: none !important;
}

/* ---- buttons ---- */
.stButton > button {
    border-radius: 12px; font-weight: 600; border: 1px solid #ddd9f7;
    transition: transform .05s ease, box-shadow .2s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(108,92,231,0.18); }

/* ---- expanders (citation sources) ---- */
[data-testid="stExpander"] {
    border: 1px solid #ecebfa; border-radius: 12px; background: #ffffff;
}

/* ---- chat bubbles ---- */
[data-testid="stChatMessage"] { border-radius: 14px; }

/* ---- pipeline chips (How it works) ---- */
.vr-flow { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 8px 0 18px; }
.vr-step {
    background: #fff; border: 1px solid #e6e3fb; border-radius: 12px;
    padding: 10px 14px; font-weight: 600; font-size: 0.86rem; color: #3b3560;
    box-shadow: 0 3px 10px rgba(108,92,231,0.07);
}
.vr-step small { display:block; font-weight: 500; color: #8a86a8; font-size: 0.72rem; }
.vr-arrow { color: #b3aee0; font-weight: 800; font-size: 1.1rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def _init_state():
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("build_stats", None)
    st.session_state.setdefault("backend", GENERATOR.backend)
    st.session_state.setdefault("chat_history", [])


_init_state()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_evaluator():
    return RagEvaluator()


def build_index_from_paths(paths):
    retriever = HybridRetriever()
    stats = retriever.build(chunk_pages(load_documents(paths)))
    st.session_state.retriever = retriever
    st.session_state.build_stats = stats
    return stats


def render_answer_with_citations(answer: str, passages):
    """Show the answer, then the numbered sources it can cite."""
    st.markdown(answer)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    if not passages:
        return
    st.markdown(
        f"<span style='background:#eef2ff;color:#6C5CE7;padding:3px 11px;"
        f"border-radius:999px;font-size:0.78rem;font-weight:700;'>"
        f"📎 {len(passages)} sources retrieved · {len(cited)} cited</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    for i, p in enumerate(passages, start=1):
        marker = "✅" if i in cited else "•"
        with st.expander(f"{marker} [{i}] {p.citation}  ·  score={p.score:.3f}"):
            st.write(p.chunk.text)


# --------------------------------------------------------------------------
# Sidebar - backend + corpus
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 VeritasRAG")
    st.caption("Hybrid RAG · Citations · Evaluation")

    st.subheader("1 · Generator backend")
    backend = st.radio(
        "Answer generator",
        AVAILABLE_BACKENDS,
        index=AVAILABLE_BACKENDS.index(st.session_state.backend),
        format_func=lambda b: {
            "gemini": "Gemini 2.5 Flash (cloud)",
            "ollama": "Ollama (local, free)",
            "extractive": "Extractive (no LLM)",
        }[b],
        help="Switch the model that writes the final answer. Retrieval is identical for all.",
    )
    st.session_state.backend = backend
    GENERATOR.backend = backend

    if st.button("Test backend connection"):
        with st.spinner("Pinging backend..."):
            ok, msg = health_check(backend)
        (st.success if ok else st.error)(msg)

    st.divider()
    st.subheader("2 · Knowledge base")

    uploads = st.file_uploader(
        "Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Build index", type="primary", width="stretch"):
            if uploads:
                tmp_paths = []
                tmpdir = Path(tempfile.mkdtemp())
                for uf in uploads:
                    p = tmpdir / uf.name
                    p.write_bytes(uf.getbuffer())
                    tmp_paths.append(p)
                with st.spinner("Ingesting, chunking, embedding..."):
                    build_index_from_paths(tmp_paths)
                st.success("Index built from uploads.")
            else:
                st.warning("Upload a file first, or load the sample corpus.")
    with col_b:
        if st.button("Load sample", width="stretch"):
            sample_paths = sorted(SAMPLE_DOCS_DIR.glob("*"))
            if sample_paths:
                with st.spinner("Building sample index..."):
                    build_index_from_paths(sample_paths)
                st.success(f"Loaded {len(sample_paths)} sample docs.")
            else:
                st.warning("No sample documents found in data/sample_docs/.")

    if st.session_state.build_stats:
        s = st.session_state.build_stats
        st.metric("Chunks indexed", s["num_chunks"])
        st.caption(
            f"Embedding: {s['embed_seconds']}s · BM25: {s['bm25_seconds']}s · "
            f"model: {RETRIEVAL.embedding_model.split('/')[-1]}"
        )


# --------------------------------------------------------------------------
# Main tabs
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="vr-hero">
        <div class="vr-hero-title">🔎 VeritasRAG</div>
        <div class="vr-hero-sub">
            Hybrid Retrieval-Augmented Generation — grounded answers with citations,
            built to verify instead of trust.
        </div>
        <div class="vr-hero-tags">
            <span>🧩 FAISS HNSW</span>
            <span>🔤 BM25</span>
            <span>🔀 RRF Fusion</span>
            <span>🎯 Cross-Encoder Rerank</span>
            <span>💬 Gemini · Ollama</span>
            <span>📊 Ragas Evaluation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

retriever: HybridRetriever | None = st.session_state.retriever
tab_chat, tab_cmp, tab_eval, tab_about = st.tabs(
    ["💬 Ask", "🔍 Retrieval Comparison", "📊 Evaluation", "ℹ️ How it works"]
)

# ---- Chat -----------------------------------------------------------------
with tab_chat:
    if not retriever or not retriever.is_ready:
        st.info("👈 Build an index or load the sample corpus to begin.")
    else:
        use_rag = st.toggle(
            "Ground answer in retrieved documents (RAG)",
            value=True,
            help="Turn OFF to see the model answer from memory alone — it will often "
            "hallucinate. Turn ON to force grounding in your documents.",
        )
        query = st.chat_input("Ask a question about your documents...")
        if query:
            st.session_state.chat_history.append(("user", query))

        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                if isinstance(content, tuple):  # assistant payload
                    answer, passages, meta = content
                    render_answer_with_citations(answer, passages)
                    st.caption(meta)
                else:
                    st.markdown(content)

        if query:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating..."):
                    passages = []
                    t0 = time.perf_counter()
                    if use_rag:
                        result = retriever.retrieve(query)
                        passages = result.passages
                        gen = generate(query, passages, backend=backend, grounded=True)
                        retrieval_ms = result.latency_ms
                    else:
                        gen = generate(query, [], backend=backend, grounded=False)
                        retrieval_ms = 0.0
                    total_ms = (time.perf_counter() - t0) * 1000.0

                mode = "RAG-grounded" if use_rag else "⚠️ No-RAG (memory only)"
                meta = (
                    f"{mode} · backend: {gen.backend_used} · "
                    f"retrieval {retrieval_ms:.0f} ms · total {total_ms:.0f} ms"
                )
                if gen.note:
                    st.warning(gen.note)
                render_answer_with_citations(gen.answer, passages)
                st.caption(meta)
                st.session_state.chat_history.append(
                    ("assistant", (gen.answer, passages, meta))
                )

        if st.session_state.chat_history and st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

# ---- Retrieval comparison -------------------------------------------------
with tab_cmp:
    st.subheader("Retrieval strategy comparison")
    st.caption(
        "Runs the same query through all four strategies from the research paper "
        "and shows which passages each one surfaces, plus its latency."
    )
    if not retriever or not retriever.is_ready:
        st.info("👈 Build an index first.")
    else:
        cmp_query = st.text_input("Query", key="cmp_query", placeholder="e.g. What is the refund policy?")
        if st.button("Compare strategies", type="primary") and cmp_query:
            rows = []
            passages_by_strategy = {}
            for strat in ["sparse", "dense", "hybrid", "rerank"]:
                res = retriever.run_strategy(strat, cmp_query, top_k=RETRIEVAL.top_k_final)
                passages_by_strategy[strat] = res
                for p in res.passages:
                    rows.append(
                        {
                            "Strategy": STRATEGY_LABELS[strat],
                            "Rank": p.rank,
                            "Score": round(p.score, 4),
                            "Source": p.citation,
                        }
                    )

            lat_df = pd.DataFrame(
                {
                    "Strategy": [STRATEGY_LABELS[s] for s in passages_by_strategy],
                    "Latency (ms)": [passages_by_strategy[s].latency_ms for s in passages_by_strategy],
                    "Top score": [
                        round(passages_by_strategy[s].passages[0].score, 4)
                        if passages_by_strategy[s].passages else 0.0
                        for s in passages_by_strategy
                    ],
                }
            )

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    lat_df, x="Strategy", y="Latency (ms)", color="Strategy",
                    title="Query latency by strategy", text="Latency (ms)",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")
            with c2:
                st.dataframe(lat_df, width="stretch", hide_index=True)

            st.markdown("#### Top passages per strategy")
            cols = st.columns(4)
            for col, strat in zip(cols, ["sparse", "dense", "hybrid", "rerank"]):
                with col:
                    st.markdown(f"**{STRATEGY_LABELS[strat]}**")
                    for p in passages_by_strategy[strat].passages:
                        st.caption(f"[{p.rank}] {p.citation} · {p.score:.3f}")
                        st.write(p.chunk.text[:160] + "...")

# ---- Evaluation -----------------------------------------------------------
with tab_eval:
    st.subheader("RAG evaluation dashboard")
    st.caption(
        "Ragas-style metrics computed locally (no paid API): context precision, "
        "faithfulness, answer relevance — Section 5.7 of the research paper."
    )
    if not retriever or not retriever.is_ready:
        st.info("👈 Build an index first.")
    else:
        default_eval = EVAL_SET_PATH.exists()
        st.write(
            "Evaluation questions are read from `data/eval_set.json`."
            if default_eval else
            "No eval_set.json found — using a couple of generic questions."
        )
        if st.button("Run evaluation", type="primary"):
            if default_eval:
                questions = [q["question"] for q in json.loads(EVAL_SET_PATH.read_text())]
            else:
                questions = ["What is this document about?", "Summarise the key points."]

            evaluator = _get_evaluator()
            scores = []
            progress = st.progress(0.0, text="Scoring questions...")
            for i, q in enumerate(questions, start=1):
                res = retriever.retrieve(q)
                gen = generate(q, res.passages, backend=backend, grounded=True)
                scores.append(evaluator.score_one(q, gen.answer, res.passages))
                progress.progress(i / len(questions), text=f"Scored {i}/{len(questions)}")
            progress.empty()

            agg = aggregate(scores)
            m1, m2, m3 = st.columns(3)
            m1.metric("Context Precision", f"{agg['context_precision']:.2f}")
            m2.metric("Faithfulness", f"{agg['faithfulness']:.2f}")
            m3.metric("Answer Relevance", f"{agg['answer_relevance']:.2f}")

            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=[agg["context_precision"], agg["faithfulness"], agg["answer_relevance"]],
                    theta=["Context Precision", "Faithfulness", "Answer Relevance"],
                    fill="toself",
                    name="VeritasRAG",
                )
            )
            radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Aggregate quality profile",
            )
            st.plotly_chart(radar, width="stretch")

            df = pd.DataFrame([s.as_dict() for s in scores])
            st.dataframe(df, width="stretch", hide_index=True)

# ---- About ----------------------------------------------------------------
with tab_about:
    st.subheader("The VeritasRAG pipeline")
    st.markdown(
        """
        <div class="vr-flow">
            <div class="vr-step">📄 Ingest<small>PDF / TXT → text</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step">✂️ Chunk<small>overlapping windows</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step">🧠 Embed + Index<small>FAISS HNSW + BM25</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step">🔀 Fuse (RRF)<small>dense + sparse</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step">🎯 Rerank<small>cross-encoder</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step">💬 Generate<small>cited answer</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step">📊 Evaluate<small>Ragas metrics</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("How it maps to the research paper")
    st.markdown(
        """
| Pipeline phase | Paper section | Where in this app |
|---|---|---|
| **1. Ingestion** | 4.3 | `src/ingestion.py` — PDF/TXT text + page numbers |
| **2. Indexing** | 4.5 | `src/chunking.py`, `src/vector_store.py` (FAISS **HNSW**) |
| **3. Retrieval** | 4.4 | `src/sparse.py` (BM25) + dense + `src/fusion.py` (**RRF**) + `src/reranker.py` |
| **4. Augmentation** | 5.2 / 5.6 | `src/prompt_builder.py` — instruction hardening + delimited context |
| **5. Generation** | 4.3 | `src/generator.py` — Gemini / Ollama / extractive |
| **6. Evaluation** | 5.7 | `src/evaluation.py` — Ragas-style metrics |

**Production retrieval path:** BM25 + Dense → Reciprocal Rank Fusion →
Cross-encoder rerank → top-k → grounded prompt → LLM answer with `[n]` citations.
This is the *"hybrid retrieval with a cross-encoder reranker"* configuration the
paper identifies as the most consistently strong.
        """
    )
    st.info(
        f"Chunk size ≈ {RETRIEVAL.chunk_size_words} words with "
        f"{RETRIEVAL.chunk_overlap_words}-word overlap · "
        f"embedding dim depends on model · HNSW M={RETRIEVAL.hnsw_m}."
    )
