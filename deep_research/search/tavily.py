from tavily import TavilyClient

from .base import make_finding


def search(query: str, max_results: int = 3) -> list[dict]:
    """Search Tavily; on any failure log a warning and return an empty list."""
    try:
        client = TavilyClient()
        resp = client.search(query, max_results=max_results)
    except Exception as e:  # noqa: BLE001 — one failed query must not abort the round
        print(f"[warn] 检索失败 query={query!r}: {e}")
        return []
    return [
        make_finding(query=query, title=r.get("title", ""),
                     url=r.get("url", ""), content=r.get("content", ""))
        for r in resp.get("results", [])
    ]
