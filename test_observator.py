"""Test that observability is wired into the pipeline."""

from src.generation.graph import query

if __name__ == "__main__":
    #Warming-UP the Embedder
    from src.ingestion.embedder import warmup
    print("Pre-warming embedding model...")
    warmup()
    
    result = query("Using the 'RECONCILIATION OF GAAP TO NON-GAAP FINANCIAL MEASURES' table in the fiscal 2026 report, what is the value for Non-GAAP gross profit for the three months ended October 26, 2025?")
    #Did Nvidia launch Borderlands 4, and if so, what specific DLSS version was used

    print("=" * 60)
    print("ANSWER:", result["generation"][:200])
    print("=" * 60)

    # Latency per stage
    print("\nLATENCY:")
    total_ms = 0
    for stage, ms in result.get("latency_ms", {}).items():
        print(f"  {stage}: {ms:.0f}ms")
        total_ms += ms
    print(f"  TOTAL: {total_ms:.0f}ms")

    # Cost
    print(f"\nTOTAL COST: ${result.get('cost_usd', 0):.6f}")

    # Token usage breakdown
    print("\nTOKEN USAGE BY STAGE:")
    for stage, summary in result.get("token_usage", {}).items():
        if isinstance(summary, dict):
            calls = summary.get("num_calls", 0)
            tokens_in = summary.get("total_input_tokens", 0)
            tokens_out = summary.get("total_output_tokens", 0)
            cost = summary.get("total_cost_usd", 0)
            print(f"  {stage}: {calls} calls, {tokens_in} in / {tokens_out} out, ${cost:.6f}")

    print(f"\nHallucination: {result.get('is_hallucination', False)}")
    print(f"Retries: {result.get('retry_count', 0)}")