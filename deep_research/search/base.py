def make_finding(query, title, url, content,
                 authors=None, year=None, doi=None, venue=None, cited_by=None) -> dict:
    """Unified search result. Academic fields stay empty/None for web sources."""
    return {
        "query": query,
        "title": title,
        "url": url,
        "content": content,
        "authors": authors or [],
        "year": year,
        "doi": doi,
        "venue": venue,
        "cited_by": cited_by,
    }
