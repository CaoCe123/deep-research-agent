from deep_research import tools


def test_slugify_basic():
    assert tools.slugify("Hello World") == "Hello-World"


def test_slugify_keeps_cjk_and_strips_punctuation():
    out = tools.slugify("2026 年小型语言模型（SLM）")
    assert out == "2026-年小型语言模型-SLM"


def test_slugify_truncates_and_has_fallback():
    assert len(tools.slugify("a" * 200)) == 60
    assert tools.slugify("!!!") == "report"
