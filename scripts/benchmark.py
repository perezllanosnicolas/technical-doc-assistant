import os
import time
from src.retriever import compare_retrievers
from src.ingestion import load_vectorstore
from langchain_core.documents import Document

# ── 1. Charge vectorstore ─────────────────────────────────────────────
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
persist_dir = os.path.join(base_dir, "vectorstore")

print(f"Loading vectorstore from: {persist_dir}")
vs = load_vectorstore(persist_dir=persist_dir)
data = vs.get()

all_docs = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(data["documents"], data["metadatas"])
]
print(f"✅ {len(all_docs)} chunks loaded\n")

# ── 2. Queries of benchmark ───────────────────────────────────────────
queries = [
    "H-Net architecture advantages",
    "Hierarchical sequence modeling end-to-end",
    "Performance benchmark results",
    "Boundary prediction accuracy",
]

STRATEGIES = ["base_similarity", "mmr", "contextual", "ensemble"]

descriptions = {
    "base_similarity": "General conceptual queries",
    "mmr":             "Repetitive / redundant docs",
    "contextual":      "Multi-paragraph specifications",
    "ensemble":        "Technical IDs, codes, numbers",
}

# ── 3. Execute the benchmark ─────────────────────────────────────────────
summary: dict[str, list[float]] = {s: [] for s in STRATEGIES}

for q in queries:
    print("=" * 70)
    print(f"QUERY: {q}")
    print("=" * 70)

    t0 = time.perf_counter()
    results = compare_retrievers(vs, all_docs, query=q, k=3)
    elapsed = time.perf_counter() - t0

    for strategy in STRATEGIES:
        data_s = results.get(strategy, {})
        if "error" in data_s:
            print(f"  [{strategy}] ERROR: {data_s['error']}")
            continue

        num_chunks = data_s.get("num_chunks", 0)
        sources    = data_s.get("sources", [])

        docs_returned = data_s.get("docs", [])  

        if docs_returned:
            # Comparar por contenido completo del chunk
            unique_contents = len(set(d.page_content for d in docs_returned))
        else:

            previews = data_s.get("preview", [])
            unique_contents = len(set(previews))  

        content_diversity = unique_contents / num_chunks if num_chunks else 0
        summary[strategy].append(content_diversity)

        # Mostrar resultado por query
        unique_sources = len(set(sources))
        print(f"\n  ┌─ {strategy.upper()} ({elapsed:.2f}s total) ─")
        print(f"  │  chunks: {num_chunks}  |  unique content: {unique_contents}  |  diversity: {content_diversity:.2f}")
        for i, doc in enumerate(docs_returned[:3], 1):
            src  = doc.metadata.get("source", "unknown").split("/")[-1].split("\\")[-1]
            prev = doc.page_content[:120].replace("\n", " ")
            print(f"  │  [{i}] ({src}) {prev}...")
        print(f"  └{'─'*60}")


summary_chunks:  dict[str, list[int]]   = {s: [] for s in STRATEGIES}
summary_excl:    dict[str, list[int]]   = {s: [] for s in STRATEGIES}  # chunks exclusivos

all_content_sets = {
    s: set(d.page_content for d in results[s].get("docs", []))
    for s in STRATEGIES if "error" not in results.get(s, {})
}
union_all = set().union(*all_content_sets.values())

for strategy in STRATEGIES:
    data_s = results.get(strategy, {})
    if "error" in data_s:
        continue
    docs_returned = data_s.get("docs", [])
    content_set   = set(d.page_content for d in docs_returned)

    other_sets = set().union(*(
        v for k, v in all_content_sets.items() if k != strategy
    ))
    exclusive = content_set - other_sets

    summary_chunks[strategy].append(len(docs_returned))
    summary_excl[strategy].append(len(exclusive))

# ── 4. Summary table ─────────────────────────────────────────────────────
print("\n")
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)
print(f"  {'Strategy':<20} {'Avg chunks':>10}  {'Exclusive finds':>15}  {'Recommended for'}")
print(f"  {'-'*20} {'-'*10}  {'-'*15}  {'-'*30}")

scores = {}
for strategy in STRATEGIES:
    avg_k   = sum(summary_chunks[strategy]) / len(summary_chunks[strategy]) if summary_chunks[strategy] else 0
    avg_exc = sum(summary_excl[strategy])   / len(summary_excl[strategy])   if summary_excl[strategy]   else 0
    scores[strategy] = avg_exc  # "mejor" = más hallazgos únicos
    print(f"  {strategy:<20} {avg_k:>10.1f}  {avg_exc:>15.1f}  {descriptions.get(strategy, '')}")

best = max(scores, key=scores.get)
print(f"\n  Best overall: {best} (exclusive finds: {scores[best]:.1f} avg per query)")
print("=" * 70)