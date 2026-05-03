"""
RAGAS evaluation for the RAG pipeline.
Measures: faithfulness, answer relevancy, context precision, context recall.
"""

import json
from typing import List, Dict
from datasets import Dataset

from src.generation.graph import query as rag_query


def load_golden_dataset(path: str = "eval/golden_dataset.json") -> List[Dict]:
    """Load the golden Q&A dataset."""
    with open(path, "r") as f:
        return json.load(f)


def run_pipeline_on_dataset(dataset: List[Dict]) -> Dict[str, List]:
    """
    Run the RAG pipeline on each question in the dataset.
    Collects the data RAGAS needs: question, answer, contexts, ground_truth.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}: {item['question'][:50]}...")

        try:
            result = rag_query(item["question"])

            questions.append(item["question"])
            answers.append(result["generation"])
            contexts.append([doc["text"] for doc in result["documents"]])
            ground_truths.append(item["ground_truth"])

        except Exception as e:
            print(f"  ERROR: {e}")
            questions.append(item["question"])
            answers.append("ERROR: Failed to generate answer")
            contexts.append([])
            ground_truths.append(item["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def evaluate(dataset_path: str = "eval/golden_dataset.json") -> Dict:
    """
    Run RAGAS evaluation and return scores.

    Metrics:
    - faithfulness: Is the answer grounded in the retrieved context?
    - answer_relevancy: Does the answer address the question?
    - context_precision: Are the retrieved docs relevant?
    - context_recall: Did we retrieve the docs we needed?
    """
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    # Load golden dataset
    golden = load_golden_dataset(dataset_path)
    print(f"Loaded {len(golden)} evaluation questions")

    # Run pipeline
    print("\nRunning RAG pipeline on evaluation set...")
    data = run_pipeline_on_dataset(golden)

    # Create RAGAS dataset
    ragas_dataset = Dataset.from_dict(data)

    # Run evaluation
    print("\nRunning RAGAS evaluation...")
    results = ragas_evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # Print results
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in results.items():
        if isinstance(score, float):
            print(f"  {metric}: {score:.4f}")
    print("=" * 50)

    return dict(results)


if __name__ == "__main__":
    evaluate()