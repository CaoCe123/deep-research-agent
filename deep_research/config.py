import json
import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic

REASONING_MODEL = "claude-opus-4-8"          # 规划 / 反思 / 写作
SUMMARY_MODEL = "claude-haiku-4-5-20251001"  # 检索结果摘要

# provider 配置（baseURL 等）从 providers.json 读取。
# API key 从 AGIBOT_API_KEY 读取（兼容回退到 ANTHROPIC_API_KEY），不写进任何文件。
_PROVIDERS_PATH = Path(__file__).resolve().parent.parent / "providers.json"
API_KEY_ENV = "AGIBOT_API_KEY"
API_KEY_ENV_FALLBACK = "ANTHROPIC_API_KEY"


def get_api_key() -> str | None:
    return os.getenv(API_KEY_ENV) or os.getenv(API_KEY_ENV_FALLBACK)


def _base_url(model: str) -> str | None:
    """Return the configured baseURL for a model, or None to use the default endpoint."""
    try:
        providers = json.loads(_PROVIDERS_PATH.read_text(encoding="utf-8"))["provider"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None
    return providers.get(model, {}).get("baseURL")


def _make_model(model: str) -> ChatAnthropic:
    # 注意：agibot/Bedrock 后端已弃用 temperature 参数（传入会 400），故不设置。
    kwargs = {"model": model}
    base_url = _base_url(model)
    if base_url:
        kwargs["base_url"] = base_url
    api_key = get_api_key()
    if api_key:
        kwargs["api_key"] = api_key
    return ChatAnthropic(**kwargs)


def get_reasoning_model() -> ChatAnthropic:
    return _make_model(REASONING_MODEL)


def get_summary_model() -> ChatAnthropic:
    return _make_model(SUMMARY_MODEL)
