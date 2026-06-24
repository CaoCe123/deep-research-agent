from deep_research import config


def test_model_name_constants():
    assert config.REASONING_MODEL == "claude-opus-4-8"
    assert config.SUMMARY_MODEL == "claude-haiku-4-5-20251001"


def test_factories_return_models_with_expected_names(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reasoning = config.get_reasoning_model()
    summary = config.get_summary_model()
    assert reasoning.model == config.REASONING_MODEL
    assert summary.model == config.SUMMARY_MODEL


def test_factories_apply_configured_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reasoning = config.get_reasoning_model()
    # providers.json points all models at the agibot endpoint
    assert str(reasoning.anthropic_api_url) == "https://lingzhi.agibot.com"
