"""
Turns retrieved chunks + a question into a grounded answer.

Uses Groq's free, OpenAI-compatible chat completions endpoint. The
prompt explicitly instructs the model to answer only from the supplied
context and to say so when the context doesn't cover the question —
this is the main lever against hallucination in a RAG system.
"""
from typing import List, Dict, Any
from openai import OpenAI

from app.config import settings

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the "
    "provided context excerpts. If the context does not contain the "
    "answer, say clearly that the documents don't cover it — never "
    "invent information. Keep answers concise and cite which source "
    "each fact came from when relevant."
)


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    if not settings.groq_api_key:
        return (
            "No LLM API key configured (GROQ_API_KEY). Retrieval is working — "
            "here are the most relevant excerpts found, but I can't compose a "
            "natural-language answer without a configured model."
        )

    if not chunks:
        return "I couldn't find anything relevant to that question in the indexed documents."

    client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)

    context = _build_context(chunks)
    user_prompt = (
        f"Context excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content
