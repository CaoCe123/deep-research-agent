import re

from tavily import TavilyClient


def slugify(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe slug, preserving CJK and alphanumerics."""
    cleaned = re.sub(r"[^\w一-鿿]+", "-", text, flags=re.UNICODE).strip("-")
    return cleaned[:max_len] or "report"


def tavily_search(query: str, max_results: int = 3) -> list[dict]:
    """Search Tavily; on any failure log a warning and return an empty list."""
    try:
        client = TavilyClient()
        resp = client.search(query, max_results=max_results)
    except Exception as e:  # noqa: BLE001 — one failed query must not abort the round
        print(f"[warn] 检索失败 query={query!r}: {e}")
        return []
    return [
        {
            "query": query,
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]
