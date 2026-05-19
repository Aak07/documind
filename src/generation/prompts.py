"""
All prompts in one place — versioned, documented, and easy to A/B test.
"""

RAG_PROMPT = """You are a precise document analyst. Answer the question using ONLY
information explicitly stated in the context below.

STRICT RULES:
1. Use only facts directly stated in the context — no inferences, no external knowledge.
2. If the context contains the answer, quote or closely paraphrase the relevant text.
3. If the context does not contain enough information, say exactly:
   "The provided documents do not contain sufficient information to answer this question."
4. Always cite which document and page your answer comes from.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (cite your source):"""


RELEVANCE_GRADER_PROMPT = """You are a relevance grader. Given a user question and a retrieved document chunk, determine if the document is relevant to answering the question.

Give a binary score: 'yes' or 'no'.
'yes' means the document contains information relevant to the question.

DOCUMENT:
{document}

QUESTION: {question}

Respond with ONLY 'yes' or 'no':"""


HALLUCINATION_GRADER_PROMPT = """You are a factual grounding checker. Your job is to verify if the ANSWER below is supported by the SOURCE DOCUMENTS.

An answer is "grounded" if the key claims in the answer can be found in or reasonably inferred from the source documents. Minor phrasing differences are acceptable — focus on factual accuracy.

SOURCE DOCUMENTS:
{documents}

ANSWER TO CHECK:
{generation}

Is this answer grounded in the source documents? Reply with exactly one word: yes or no"""


ANSWER_GRADER_PROMPT = """You are an answer quality grader. Given a question and an AI-generated answer, determine if the answer actually addresses the question.

Give a binary score: 'yes' or 'no'.
'yes' means the answer is useful and addresses the question.

QUESTION: {question}

ANSWER: {generation}

Does the answer address the question? Respond with ONLY 'yes' or 'no':"""