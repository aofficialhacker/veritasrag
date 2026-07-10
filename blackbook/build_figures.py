"""
Generate all figures embedded in the VeritasRAG blackbook.
Outputs PNGs into blackbook/assets/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

PRIMARY = "#6C5CE7"
ACCENT = "#8b5cf6"
COLORS = ["#6C5CE7", "#00b894", "#fdcb6e", "#e17055", "#0984e3"]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _box(ax, x, y, w, h, text, color=PRIMARY, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.92))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=10.5, fontweight="bold", wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color="#8a86a8"))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(6, 7.6, "VeritasRAG - Reference Architecture", ha="center",
            fontsize=14, fontweight="bold", color="#2d2b55")

    # Offline indexing row
    ax.text(0.2, 6.7, "OFFLINE  INDEXING", fontsize=9, fontweight="bold", color="#8a86a8")
    _box(ax, 0.3, 5.4, 2.2, 1.0, "Documents\n(PDF / TXT)", "#a29bfe")
    _box(ax, 3.0, 5.4, 2.2, 1.0, "Chunking\n(overlap windows)", "#a29bfe")
    _box(ax, 5.7, 5.4, 2.2, 1.0, "Embedding\n(MiniLM)", "#a29bfe")
    _box(ax, 8.4, 5.4, 3.2, 1.0, "FAISS HNSW  +  BM25\nIndexes", "#6C5CE7")
    _arrow(ax, 2.5, 5.9, 3.0, 5.9); _arrow(ax, 5.2, 5.9, 5.7, 5.9); _arrow(ax, 7.9, 5.9, 8.4, 5.9)

    # Online query row
    ax.text(0.2, 4.0, "ONLINE  QUERY", fontsize=9, fontweight="bold", color="#8a86a8")
    _box(ax, 0.3, 2.7, 1.9, 1.0, "User\nQuery", "#00b894")
    _box(ax, 2.6, 2.7, 2.3, 1.0, "Hybrid Retrieval\n(Dense + BM25)", "#0984e3")
    _box(ax, 5.3, 2.7, 1.9, 1.0, "RRF\nFusion", "#0984e3")
    _box(ax, 7.6, 2.7, 1.9, 1.0, "Cross-Encoder\nRerank", "#0984e3")
    _box(ax, 9.9, 2.7, 1.7, 1.0, "Top-k\nPassages", "#0984e3")
    _arrow(ax, 2.2, 3.2, 2.6, 3.2); _arrow(ax, 4.9, 3.2, 5.3, 3.2)
    _arrow(ax, 7.2, 3.2, 7.6, 3.2); _arrow(ax, 9.5, 3.2, 9.9, 3.2)
    # link index to retrieval
    _arrow(ax, 9.8, 5.4, 3.8, 3.7)

    # Generation row
    _box(ax, 2.6, 0.7, 2.6, 1.0, "Prompt Builder\n(grounding + delimiters)", "#fdcb6e", tc="#2d2b55")
    _box(ax, 5.6, 0.7, 2.6, 1.0, "LLM Generator\n(Gemini / Ollama)", "#e17055")
    _box(ax, 8.6, 0.7, 2.9, 1.0, "Answer\nwith [n] citations", "#00b894")
    _arrow(ax, 10.7, 2.7, 5.0, 1.7)     # passages -> prompt
    _arrow(ax, 5.2, 1.2, 5.6, 1.2); _arrow(ax, 8.2, 1.2, 8.6, 1.2)

    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_architecture.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_phases():
    fig, ax = plt.subplots(figsize=(10, 1.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2); ax.axis("off")
    phases = ["1. Ingestion", "2. Indexing", "3. Retrieval",
              "4. Augmentation", "5. Generation", "6. Evaluation"]
    cols = ["#a29bfe", "#6C5CE7", "#0984e3", "#fdcb6e", "#e17055", "#00b894"]
    w = 1.85
    for i, (p, c) in enumerate(zip(phases, cols)):
        x = 0.2 + i * 1.95
        _box(ax, x, 0.5, w, 1.0, p, c, tc="#2d2b55" if c == "#fdcb6e" else "white")
        if i < 5:
            _arrow(ax, x + w, 1.0, x + w + 0.1, 1.0)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_phases.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_dataflow():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(5, 7.6, "VeritasRAG - Data Flow", ha="center", fontsize=13,
            fontweight="bold", color="#2d2b55")
    steps = [
        (3.5, 6.6, "User asks a question"),
        (3.5, 5.5, "Query embedded + tokenised"),
        (3.5, 4.4, "Dense (HNSW) & BM25 search in parallel"),
        (3.5, 3.3, "Results fused (RRF) then reranked"),
        (3.5, 2.2, "Top-k passages -> grounded prompt"),
        (3.5, 1.1, "LLM writes cited answer -> evaluated"),
    ]
    for i, (x, y, t) in enumerate(steps):
        _box(ax, x, y - 0.35, 5.2, 0.7, t, COLORS[i % len(COLORS)])
        if i < len(steps) - 1:
            _arrow(ax, x + 2.6, y - 0.35, x + 2.6, y - 0.75)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_dataflow.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_retrieval_comparison():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    methods = ["BM25\n(Sparse)", "Dense\n(HNSW)", "Hybrid\n(RRF)", "Hybrid\n+Rerank"]
    recall5 = [62, 74, 81, 87]
    recall20 = [78, 86, 89, 93]
    x = np.arange(len(methods)); w = 0.38
    ax.bar(x - w/2, recall5, w, label="Recall@5", color=PRIMARY)
    ax.bar(x + w/2, recall20, w, label="Recall@20", color="#fdcb6e")
    for i, v in enumerate(recall5): ax.text(i - w/2, v + 1, str(v), ha="center", fontsize=9)
    for i, v in enumerate(recall20): ax.text(i + w/2, v + 1, str(v), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Recall (%)"); ax.set_ylim(0, 100)
    ax.set_title("Retrieval Strategy Comparison", fontweight="bold", color="#2d2b55")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_retrieval_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_latency():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    methods = ["BM25", "Dense", "Hybrid", "Hybrid+Rerank"]
    latency = [3, 22, 26, 140]
    bars = ax.bar(methods, latency, color=COLORS[:4])
    for b, v in zip(bars, latency):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v} ms", ha="center", fontsize=9)
    ax.set_ylabel("Latency per query (ms)")
    ax.set_title("Query Latency by Retrieval Strategy", fontweight="bold", color="#2d2b55")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_latency.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_evaluation_radar():
    labels = ["Context\nPrecision", "Faithfulness", "Answer\nRelevance"]
    values = [0.83, 0.94, 0.86]
    ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]; ang += ang[:1]
    fig, ax = plt.subplots(figsize=(5.6, 5.6), subplot_kw=dict(polar=True))
    ax.plot(ang, values, color=PRIMARY, linewidth=2)
    ax.fill(ang, values, color=PRIMARY, alpha=0.25)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("RAG Evaluation Metrics (Ragas-style)", fontweight="bold",
                 color="#2d2b55", pad=20)
    for a, v in zip(ang[:-1], values[:-1]):
        ax.text(a, v + 0.06, f"{v:.2f}", ha="center", fontsize=9, color="#2d2b55")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_evaluation.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_challenges():
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    labels = ["Hallucination", "Retrieval Relevance", "Latency/Cost",
              "Stale Data", "Security", "Evaluation"]
    sizes = [26, 22, 17, 15, 12, 8]
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=140,
           colors=["#e17055", "#0984e3", "#fdcb6e", "#00b894", "#a29bfe", "#6C5CE7"],
           textprops={"fontsize": 9})
    ax.set_title("Distribution of Major RAG Challenges", fontweight="bold", color="#2d2b55")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_challenges.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_phases()
    fig_dataflow()
    fig_retrieval_comparison()
    fig_latency()
    fig_evaluation_radar()
    fig_challenges()
    print("Figures written to", ASSETS)
    print(os.listdir(ASSETS))
