import re


def slugify(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe slug, preserving CJK and alphanumerics."""
    cleaned = re.sub(r"[^\w一-鿿]+", "-", text, flags=re.UNICODE).strip("-")
    return cleaned[:max_len] or "report"
