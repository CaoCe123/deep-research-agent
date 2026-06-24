from langchain_anthropic import ChatAnthropic

REASONING_MODEL = "claude-opus-4-8"          # 规划 / 反思 / 写作
SUMMARY_MODEL = "claude-haiku-4-5-20251001"  # 检索结果摘要


def get_reasoning_model() -> ChatAnthropic:
    return ChatAnthropic(model=REASONING_MODEL, temperature=0)


def get_summary_model() -> ChatAnthropic:
    return ChatAnthropic(model=SUMMARY_MODEL, temperature=0)
