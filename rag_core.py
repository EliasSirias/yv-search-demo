import re
from typing import Any
from config import MAX_DISTANCE

max_distance = MAX_DISTANCE


def keyword_overlap_ok(question: str, kept_texts: list[str]) -> bool:
    # Extract simple keywords (>=3 chars). Keeps acronyms like SSO.
    q_terms = set(re.findall(r"[A-Za-z]{3,}", question.lower()))
    if not q_terms:
        return True

    ctx_terms = set(re.findall(r"[A-Za-z]{3,}", " ".join(kept_texts).lower()))

    # Require at least one meaningful shared term
    return len(q_terms.intersection(ctx_terms)) >= 1


def scope_coverage_ok(question: str, kept_texts: list[str]) -> bool:
    q = question.lower()
    ctx = " ".join(kept_texts).lower()

    # If the user asks tenant-scoped questions, require tenant language in the retrieved text.
    tenant_scoped = bool(
        re.search(r"\btenant\b|\bper[-\s]?tenant\b|\bspecific tenant\b", q)
    )
    if tenant_scoped:
        return bool(
            re.search(r"\btenant\b|\bper[-\s]?tenant\b|\btenant[-\s]?specific\b", ctx)
        )
    return True


def multi_intent_coverage_ok(question: str, kept_texts: list[str]) -> bool:
    q = question.lower()
    ctx = " ".join(kept_texts).lower()

    asks_sync = bool(re.search(r"\bsync\b|\bsynchroniz", q))
    asks_perms = bool(re.search(r"\bpermission\b|\bpermissions\b|\bgroup\b", q))

    if asks_sync and asks_perms:
        has_sync = bool(re.search(r"\bsync\b|\bsynchroniz", ctx))
        has_perms = bool(re.search(r"\bpermission\b|\bpermissions\b|\bgroup\b", ctx))
        return has_sync and has_perms

    return True


def search_docs(
    query: str,
    vectorstore: Any,
    knowledge_base_filter: str = "All",
    k: int = 2,
    min_hit_count: int = 1,
    max_distance: float = 1.25,
) -> dict:
    candidate_k = max(k * 4, 10)

    docs_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=candidate_k,
    )

    kept = []

    for document, score in docs_with_scores:
        page_content = document.page_content

        if isinstance(page_content, str):
            text = page_content.strip()
        else:
            text = str(page_content).strip()
        source = document.metadata.get("source", "Unknown Source")
        knowledge_base = document.metadata.get("knowledge_base", "Unknown")
        category = document.metadata.get("category", "Uncategorized")
        relative_path = document.metadata.get("relative_path", source)

        matches_selected_base = (
            knowledge_base_filter == "All" or knowledge_base == knowledge_base_filter
        )

        if text and score <= max_distance and matches_selected_base:
            kept.append(
                {
                    "text": text,
                    "score": float(score),
                    "source": source,
                    "knowledge_base": knowledge_base,
                    "category": category,
                    "relative_path": relative_path,
                }
            )

        if len(kept) >= k:
            break

    kept_texts = [item["text"] for item in kept]
    passes_keywords = keyword_overlap_ok(query, kept_texts)
    passes_scope = scope_coverage_ok(query, kept_texts)
    passes_multi = multi_intent_coverage_ok(query, kept_texts)

    if (
        len(kept) < min_hit_count
        or not passes_keywords
        or not passes_scope
        or not passes_multi
    ):
        if not passes_scope:
            message = (
                "I found documentation related to your topic, but it does not "
                "cover the tenant-specific part of your question."
            )
        elif not passes_multi:
            message = (
                "I found documentation related to part of your question, but it "
                "does not cover the full combination of topics."
            )
        else:
            message = "No relevant documentation found."

        return {
            "found": False,
            "query": query,
            "message": message,
            "results": [],
        }
    best_score = min(item["score"] for item in kept)
    if best_score <= 1.00:
        confidence_label = "High confidence"
        confidence_message = "Documentation closely matches your question."
    else:
        confidence_label = "Possible match"
        confidence_message = (
            "I found related documentation that may help, "
            "but it may not fully answer your question."
        )
    return {
        "found": True,
        "query": query,
        "best_score": min(item["score"] for item in kept),
        "confidence": {
            "label": confidence_label,
            "message": confidence_message,
        },
        "results": kept,
    }
