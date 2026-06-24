from contextlib import contextmanager

import pytest

import main


class _FakeApp:
    """Stand-in for a compiled graph: streams two node updates, then exposes a report."""
    def __init__(self, report):
        self._report = report

    def stream(self, inputs, cfg, stream_mode):
        yield {"plan": {}}
        yield {"write": {}}

    def get_state(self, cfg):
        return type("S", (), {"values": {"report": self._report}})()


class _FakeBuilder:
    def __init__(self, report):
        self._report = report

    def compile(self, checkpointer=None):
        return _FakeApp(self._report)


def _patch_graph(monkeypatch, report):
    """Patch main()'s dependencies so it runs without keys, network, or a real graph."""
    monkeypatch.setattr(main, "check_keys", lambda *a, **k: None)
    monkeypatch.setattr(main, "build_graph", lambda: _FakeBuilder(report))

    @contextmanager
    def fake_saver(conn_string):
        yield object()

    import langgraph.checkpoint.sqlite as sqlite_mod
    monkeypatch.setattr(sqlite_mod.SqliteSaver, "from_conn_string",
                        staticmethod(fake_saver))


def test_parse_args_defaults():
    args = main.parse_args(["my topic"])
    assert args.topic == "my topic"
    assert args.max_iters == 3
    assert args.out == "reports"
    assert args.thread_id is None
    assert args.sqlite == "research.sqlite"


def test_parse_args_overrides():
    args = main.parse_args(["t", "--max-iters", "1", "--out", "o", "--thread-id", "x"])
    assert args.max_iters == 1 and args.out == "o" and args.thread_id == "x"


def test_report_path_uses_slug(tmp_path):
    p = main.report_path(str(tmp_path), "Hello World")
    assert p.name == "Hello-World.md"
    assert p.parent == tmp_path


def test_check_keys_exits_when_missing(monkeypatch):
    monkeypatch.delenv("AGIBOT_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main.check_keys()
    assert exc.value.code != 0


def test_check_keys_passes_with_agibot_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AGIBOT_API_KEY", "a")
    monkeypatch.setenv("TAVILY_API_KEY", "b")
    main.check_keys()  # should not raise


def test_check_keys_passes_with_anthropic_fallback(monkeypatch):
    monkeypatch.delenv("AGIBOT_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("TAVILY_API_KEY", "b")
    main.check_keys()  # should not raise


def test_check_keys_openalex_does_not_require_tavily(monkeypatch):
    monkeypatch.setenv("AGIBOT_API_KEY", "a")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    main.check_keys("openalex")  # tavily key not needed for openalex source


def test_check_keys_openalex_still_requires_model_key(monkeypatch):
    monkeypatch.delenv("AGIBOT_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main.check_keys("openalex")
    assert exc.value.code != 0


def test_parse_args_source_defaults_to_tavily():
    args = main.parse_args(["t"])
    assert args.source == "tavily"


def test_parse_args_source_can_be_openalex():
    args = main.parse_args(["t", "--source", "openalex"])
    assert args.source == "openalex"


def test_parse_args_unknown_source_exits():
    with pytest.raises(SystemExit):
        main.parse_args(["t", "--source", "scopus"])


def test_main_writes_report_file(monkeypatch, tmp_path):
    _patch_graph(monkeypatch, report="# 报告\n[1] 来源")
    out_dir = tmp_path / "reports"
    main.main(["研究问题", "--out", str(out_dir), "--sqlite", str(tmp_path / "t.sqlite")])

    report_file = out_dir / "研究问题.md"
    assert report_file.read_text(encoding="utf-8") == "# 报告\n[1] 来源"


def test_main_exits_nonzero_on_empty_report(monkeypatch, tmp_path):
    _patch_graph(monkeypatch, report="   ")  # whitespace-only counts as empty
    with pytest.raises(SystemExit) as exc:
        main.main(["t", "--out", str(tmp_path), "--sqlite", str(tmp_path / "t.sqlite")])
    assert exc.value.code != 0
