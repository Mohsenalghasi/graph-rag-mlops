# core/generate.py
import os
import re
from typing import Any, Dict, List

from openai import AzureOpenAI

from core.embedder import embed_query
from core.search import search_top_k

# Match any [something]
_ANY_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


def _aoai_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    if not endpoint or not api_key or not api_version:
        raise RuntimeError(
            "Missing env vars: AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_API_VERSION"
        )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def _format_context(hits: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """
    Each chunk becomes:
    [source|page]
    content
    """
    parts: List[str] = []
    used = 0

    for h in hits:
        src = (h.get("source_basename") or h.get("source") or "").strip()
        page = h.get("page")
        content = (h.get("content") or "").strip()
        if not src or page is None or not content:
            continue

        key = f"{src}|{int(page)}"
        block = f"[{key}]\n{content}\n"

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n".join(parts).strip()


def _allowed_keys(hits: List[Dict[str, Any]]) -> List[str]:
    keys: List[str] = []
    for h in hits:
        src = (h.get("source_basename") or h.get("source") or "").strip()
        page = h.get("page")
        if src and page is not None:
            keys.append(f"{src}|{int(page)}")
    return sorted(set(keys))


def _split_lines(text: str) -> List[str]:
    """
    We enforce ONE sentence per line.
    Each line must end with exactly one citation.
    """
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def answer_question(query: str, k: int = 5) -> Dict[str, Any]:
    refusal = "I don't have enough evidence in the retrieved documents to answer this."

    query = (query or "").strip()
    if not query:
        raise ValueError("Query is empty")

    # ----------------
    # Retrieval
    # ----------------
    qvec = embed_query(query)
    hits = search_top_k(qvec, k=k)

    allowed = _allowed_keys(hits)
    allowed_set = set(allowed)
    context = _format_context(hits)

    if not context:
        return {"answer": refusal, "citations": [], "k": k}

    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    if not chat_deployment:
        raise RuntimeError("Missing env var: AZURE_OPENAI_CHAT_DEPLOYMENT")

    # ----------------
    # Prompt (STRICT)
    # ----------------
    system = (
        "You are a careful assistant. Use ONLY the provided context.\n"
        "Rules:\n"
        f"1) If the answer is not directly supported by the context, output exactly:\n"
        f"   \"{refusal}\"\n"
        "2) Write the answer as BULLET LINES. One sentence per line.\n"
        "3) EACH line MUST end with exactly one citation.\n"
        "4) Format EXACTLY like this template:\n"
        "- <sentence> [FILENAME.pdf|PAGE]\n"
        "- <sentence> [FILENAME.pdf|PAGE]\n"
        "5) A citation MUST look like: [FILENAME.pdf|PAGE]\n"
        "6) You are ONLY allowed to copy-paste citations from this list:\n"
        f"{allowed}\n"
        "7) Do NOT invent citations. Do NOT use [6]. Do NOT use placeholders.\n"
        "8) Keep it short: 2 to 4 bullet lines maximum.\n"
    )

    user = f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n"

    client = _aoai_client()
    resp = client.chat.completions.create(
        model=chat_deployment,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    text = (resp.choices[0].message.content or "").strip()

    if text == refusal:
        return {"answer": refusal, "citations": [], "k": k}

    # ----------------
    # HARD VALIDATION
    # ----------------
    lines = _split_lines(text)
    if not lines:
        return {"answer": refusal, "citations": [], "k": k, "error": "no_output_lines"}

    used_citations: List[str] = []

    for line in lines:
        matches = list(_ANY_BRACKET_RE.finditer(line))

        # Exactly ONE citation per line
        if len(matches) != 1:
            return {
                "answer": refusal,
                "citations": [],
                "k": k,
                "error": f"invalid_citation_count_line: {line}",
            }

        key = matches[0].group(1).strip()
        if key not in allowed_set:
            return {
                "answer": refusal,
                "citations": [],
                "k": k,
                "error": f"invalid_citation_key: {key}",
            }

        used_citations.append(key)

    return {
        "answer": text,
        "citations": sorted(set(used_citations)),
        "k": k,
    }
