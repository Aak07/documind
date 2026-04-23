"""
All prompts in one place — versioned, documented, and easy to A/B test.
"""

RAG_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

RULES:
1. Answer ONLY based on the context provided below.
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question based on the provided documents."
3. Cite the source document and page number when possible.
4. Be concise and direct.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


RELEVANCE_GRADER_PROMPT = """You are a relevance grader. Given a user question and a retrieved document chunk, determine if the document is relevant to answering the question.

Give a binary score: 'yes' or 'no'.
'yes' means the document contains information relevant to the question.

DOCUMENT:
{document}

QUESTION: {question}

Respond with ONLY 'yes' or 'no':"""


HALLUCINATION_GRADER_PROMPT = """You are a hallucination grader. Given a set of source documents and an AI-generated answer, determine if the answer is grounded in the source documents.

Give a binary score: 'yes' or 'no'.
'yes' means the answer is grounded in the documents (no hallucination).
'no' means the answer contains information NOT found in the documents.

SOURCE DOCUMENTS:
{documents}

GENERATED ANSWER:
{generation}

Is the answer grounded in the documents? Respond with ONLY 'yes' or 'no':"""


ANSWER_GRADER_PROMPT = """You are an answer quality grader. Given a question and an AI-generated answer, determine if the answer actually addresses the question.

Give a binary score: 'yes' or 'no'.
'yes' means the answer is useful and addresses the question.

QUESTION: {question}

ANSWER: {generation}

Does the answer address the question? Respond with ONLY 'yes' or 'no':"""