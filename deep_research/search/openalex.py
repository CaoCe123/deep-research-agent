import json
import re
import urllib.parse
import urllib.request

from .base import make_finding

OPENALEX_URL = "https://api.openalex.org/works"


def _clean_query(query: str) -> str:
    """OpenAlex's `search` param rejects punctuation like ()/?,&  with HTTP 400.
    Strip everything except word chars (incl. CJK), spaces and hyphens; collapse spaces."""
    stripped = re.sub(r"[^\w\s-]", " ", query, flags=re.UNICODE)
    return re.sub(r"\s+", " ", stripped).strip()


def _restore_abstract(inverted_index) -> str:
    """OpenAlex returns abstracts as a word->[positions] inverted index. Restore it."""
    if not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def search(query: str, max_results: int = 3) -> list[dict]:
    """Search OpenAlex (no API key), sorted by citation count desc."""
    params = {
        "search": _clean_query(query),
        "sort": "cited_by_count:desc",
        "per-page": max_results,
        "mailto": "deep-research-agent@example.com",
    }
    url = f"{OPENALEX_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — one failed query must not abort the round
        print(f"[warn] OpenAlex 检索失败 query={query!r}: {e}")
        return []

    findings = []
    for w in data.get("results", []):
        authors = [a["author"]["display_name"]
                   for a in w.get("authorships", []) if a.get("author")]
        primary = w.get("primary_location") or {}
        venue = (primary.get("source") or {}).get("display_name")
        doi = w.get("doi")
        findings.append(make_finding(
            query=query,
            title=w.get("title") or "",
            url=doi or primary.get("landing_page_url") or "",
            content=_restore_abstract(w.get("abstract_inverted_index")),
            authors=authors,
            year=w.get("publication_year"),
            doi=doi,
            venue=venue,
            cited_by=w.get("cited_by_count"),
        ))
    return findings
