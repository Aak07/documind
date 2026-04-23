"""Quick test script — not a formal test, just to verify things work."""

from src.generation.graph import query

if __name__ == "__main__":
    # Make sure you've ingested documents first!
    result = query("Is the GAAP Operating Expense greater than 1 Million dollars?")

    print("=" * 60)
    print("QUESTION:", result["question"])
    print("=" * 60)
    print("ANSWER:", result["generation"])
    print("=" * 60)
    print("SOURCES:")
    for doc in result["documents"]:
        print(f"  - {doc['metadata'].get('source', 'Unknown')} "
              f"(score: {doc['score']:.3f})")
    print("=" * 60)
    print("LATENCY:")
    for stage, ms in result["latency_ms"].items():
        print(f"  {stage}: {ms:.0f}ms")
    print(f"  TOTAL: {sum(result['latency_ms'].values()):.0f}ms")
    print("=" * 60)
    print(f"Hallucination detected: {result['is_hallucination']}")
    print(f"Retry count: {result['retry_count']}")