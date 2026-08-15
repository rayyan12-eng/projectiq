from services import rag


def test_build_rag_context_returns_formatted_string():
    context = rag.build_rag_context(
        "a congested urban commercial site with unreliable suppliers and many design change orders",
        top_k=3,
    )
    assert isinstance(context, str)
    assert "similar past project" in context.lower()
    assert "cost overrun" in context.lower()


def test_build_rag_context_includes_actual_outcomes():
    context = rag.build_rag_context("Residential project small budget", top_k=5)
    assert "Actual outcome:" in context


def test_build_rag_context_no_match_returns_explicit_message():
    context = rag.build_rag_context("zzqx flibbertigibbet nonexistent", top_k=5)
    assert "no similar past projects" in context.lower()


def test_build_rag_context_respects_top_k():
    context = rag.build_rag_context("Commercial project", top_k=2)
    assert "Retrieved 2 similar" in context or "Retrieved 1 similar" in context or "No similar" in context
