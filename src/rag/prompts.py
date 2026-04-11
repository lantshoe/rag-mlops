"""
prompts.py
----------
Shared prompt templates used across all RAG pipelines.
Centralizing prompts ensures both pipelines behave consistently.
"""

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based only on the context provided below.
If the answer is not found in the context, say "I don't know based on the provided document."

Context:
{context_str}

Question: {query_str}

Answer:"""


RAG_SUMMARIZE_TEMPLATE = """Read the following document and write a comprehensive summary 
covering all main topics, key concepts, and important details.

Document:
{full_text}

Summary:"""