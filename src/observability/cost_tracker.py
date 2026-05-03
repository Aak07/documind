"""
Token counting and cost estimation per query.
Shows hiring managers you think about production costs — not just accuracy.
"""

import tiktoken
from typing import Dict, Optional

# Groq pricing (per million tokens) as of 2025
MODEL_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}


class CostTracker:
    """Track token usage and cost across a session."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = []

         # Approximate tokenizer (LLaMA not supported in tiktoken)
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = tiktoken.get_encoding("gpt2")

    def count_tokens(self, text: Optional[str]) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def record_call(
        self,
        model: str,
        input_text: str,
        output_text: str,
        stage: str = "",
    ) -> Dict:
        """Record a single LLM call's cost."""
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)

        # FIX: Fuzzy match model names — handle partial matches
        pricing = None
        for model_key, prices in MODEL_PRICING.items():
            if model_key in model or model in model_key:
                pricing = prices
                break
        if pricing is None:
            pricing = {"input": 1.0, "output": 1.0}  # Conservative fallback

        cost = (
            (pricing["input"] / 1_000_000) * input_tokens
            + (pricing["output"] / 1_000_000) * output_tokens
        )

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        call_record = {
            "stage": stage,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 8),
        }

        self.calls.append(call_record)
        return call_record

    def get_summary(self) -> Dict:
        """Get cost summary for the current session."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "num_calls": len(self.calls),
            "calls": self.calls,
        }

    def reset(self):
        """Reset tracker for a new query."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = []