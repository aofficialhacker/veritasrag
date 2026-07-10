"""Quick end-to-end smoke test (run: python smoke_test.py)."""
from pathlib import Path

from src.chunking import chunk_pages
from src.config import SAMPLE_DOCS_DIR
from src.evaluation import RagEvaluator
from src.generator import generate, health_check
from src.ingestion import load_documents
from src.retriever import HybridRetriever

print("1. Ingesting sample docs...")
paths = sorted(SAMPLE_DOCS_DIR.glob("*"))
pages = load_documents(paths)
chunks = chunk_pages(pages)
print(f"   {len(paths)} docs -> {len(pages)} pages -> {len(chunks)} chunks")

print("2. Building hybrid retriever (loads embedding model)...")
r = HybridRetriever()
stats = r.build(chunks)
print(f"   {stats}")

q = "What is the refund policy for new StreamFlow customers?"
print(f"3. Comparing strategies for: {q!r}")
for strat in ["sparse", "dense", "hybrid", "rerank"]:
    res = r.run_strategy(strat, q)
    top = res.passages[0] if res.passages else None
    print(f"   {strat:8s} {res.latency_ms:6.1f}ms  top={top.citation if top else 'none'}")

print("4. Backend health check (gemini)...")
ok, msg = health_check("gemini")
print(f"   gemini ok={ok}: {msg[:80]}")

print("5. Generating grounded answer...")
res = r.retrieve(q)
gen = generate(q, res.passages, backend="gemini", grounded=True)
print(f"   backend={gen.backend_used}")
print(f"   answer: {gen.answer[:300]}")

print("6. Evaluating...")
ev = RagEvaluator()
score = ev.score_one(q, gen.answer, res.passages)
print(f"   {score.as_dict()}")

print("\nSMOKE TEST PASSED")
