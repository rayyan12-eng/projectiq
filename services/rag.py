"""
ProjectIQ RAG (Retrieval-Augmented Generation)
------------------------------------------------
A distinct pathway from services/agent.py's tool-use loop. Where the
agent RETRIEVES DATA VIA A TOOL CALL and reasons over the tool result,
this module follows the classic RAG shape instead: retrieve first,
inject the retrieved context directly into the prompt, then generate
one grounded answer in a single call - no tool-use loop involved.

Retrieval reuses the same TF-IDF vector embeddings built by
ml/build_embeddings.py (see services/tools.py for the underlying
vector store). Only the FORMAT of grounding differs: prompt-injected
context here, vs. a tool result the model reasons over in agent.py.

Requires ANTHROPIC_API_KEY to be set in the environment to run
answer_with_rag(); build_rag_context() has no API dependency and is
fully testable offline.
"""
import os

from services import tools

SYSTEM_PROMPT = """You are ProjectIQ, a construction project risk advisor for a contracting company.

You will be given a question along with retrieved context describing similar past projects
(with their actual cost overrun % and schedule delay outcomes). Ground your answer in that
retrieved context - cite specific numbers from it. If the retrieved context doesn't cover the
question well, say so plainly rather than guessing. Keep the answer concise and structured."""


def build_rag_context(query_text: str, top_k: int = 5) -> str:
    """
    Retrieves the top_k most similar past projects via TF-IDF vector search
    and formats them as a context block ready for prompt injection.
    Pure function, no API calls - fully testable offline.
    """
    similar = tools.find_similar_projects_semantic(query_text, top_k=top_k)

    if not similar:
        return "No similar past projects were found in the historical dataset for this query."

    lines = [f"Retrieved {len(similar)} similar past project(s) from the historical dataset:\n"]
    for i, proj in enumerate(similar, 1):
        lines.append(
            f"{i}. [similarity {proj['similarity_score']}] {proj['description']} "
            f"Actual outcome: {proj['cost_overrun_pct']}% cost overrun, {proj['delay_days']} days delay."
        )
    return "\n".join(lines)


def answer_with_rag(query_text: str, top_k: int = 5, model: str = "claude-sonnet-4-6") -> str:
    """
    Full RAG call: retrieve context, inject it into the prompt, generate
    one grounded answer. Single API call, no tool-use loop.
    """
    from anthropic import Anthropic

    context = build_rag_context(query_text, top_k=top_k)

    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Retrieved context:\n{context}\n\nQuestion: {query_text}",
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or (
        "What kind of cost overrun and delay should we expect for a congested urban "
        "commercial site with unreliable suppliers and many design change orders?"
    )
    print(answer_with_rag(query))
