from . import openalex, tavily

SOURCES = {
    "tavily": tavily.search,
    "openalex": openalex.search,
}


def get_search_fn(source: str):
    """Return the search() function for a source name, or raise ValueError."""
    if source not in SOURCES:
        raise ValueError(f"未知检索源: {source}；可选: {', '.join(SOURCES)}")
    return SOURCES[source]
