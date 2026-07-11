"""
Retrieval evaluation harness.

This is the piece that turns "I called a vector search API" into
"I measured whether my retrieval actually works." It runs every
question in eval_set.json through the live retrieval pipeline and
reports two metrics:

  - Source Hit Rate@K: did the correct source document appear anywhere
    in the top-K retrieved chunks?
  - Keyword Coverage: did the retrieved text actually contain the
    key facts we expect (a proxy for "would the LLM have enough to
    answer correctly")?

Run from the backend/ virtual environment, with the vector DB already
populated (ingest backend/data/*.txt first), e.g.:

    cd backend
    python ../evaluation/evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.retrieval import retrieve  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"


def run_evaluation(top_k: int = 5) -> None:
    cases = json.loads(EVAL_SET_PATH.read_text())

    source_hits = 0
    keyword_hits = 0
    rows = []

    for case in cases:
        results = retrieve(case["question"], top_k=top_k)
        retrieved_sources = {r["source"] for r in results}
        combined_text = " ".join(r["text"].lower() for r in results)

        source_hit = case["expected_source"] in retrieved_sources
        keyword_hit = all(kw.lower() in combined_text for kw in case["must_contain"])

        source_hits += int(source_hit)
        keyword_hits += int(keyword_hit)

        rows.append(
            {
                "question": case["question"],
                "source_hit": source_hit,
                "keyword_hit": keyword_hit,
                "top_score": round(results[0]["score"], 3) if results else None,
            }
        )

    n = len(cases)
    print(f"\nEvaluated {n} questions (top_k={top_k})\n")
    print(f"{'Question':<55} {'Source Hit':<12} {'Keyword Hit':<12} {'Top Score'}")
    print("-" * 95)
    for r in rows:
        q = r["question"][:52] + "..." if len(r["question"]) > 52 else r["question"]
        print(f"{q:<55} {str(r['source_hit']):<12} {str(r['keyword_hit']):<12} {r['top_score']}")

    print("\nSummary")
    print(f"  Source Hit Rate@{top_k}:   {source_hits}/{n}  ({100*source_hits/n:.1f}%)")
    print(f"  Keyword Coverage@{top_k}:  {keyword_hits}/{n}  ({100*keyword_hits/n:.1f}%)")


if __name__ == "__main__":
    run_evaluation()
