import pytest

import main


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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main.check_keys()
    assert exc.value.code != 0


def test_check_keys_passes_when_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("TAVILY_API_KEY", "b")
    main.check_keys()  # should not raise
