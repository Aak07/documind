"""
Benchmark script that runs eval and saves timestamped results.
Usage: python -m src.evaluation.benchmark
"""

import json
import os
from datetime import datetime

from src.evaluation.ragas_eval import evaluate


def run_benchmark():
    """Run evaluation and save results."""
    print("Starting DocuMind benchmark...")

    results = evaluate()

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"eval/eval_results/benchmark_{timestamp}.json"

    os.makedirs("eval/eval_results", exist_ok=True)

    output = {
        "timestamp": timestamp,
        "metrics": {k: v for k, v in results.items() if isinstance(v, float)},
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output


if __name__ == "__main__":
    run_benchmark()