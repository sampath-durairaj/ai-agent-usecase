"""Shared helpers: call a real LLM (OpenAI or Anthropic) if its API key is set, else return a mock reply."""
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def ask_openaillm(user: str | list, system: str = "", max_tokens: int = 500, mock: str = "[MOCK] set OPENAI_API_KEY for real output") -> str:
    """`user` can be a single prompt string or a list of {"role", "content"} messages (chat history)."""
    from openai import OpenAI
    client = OpenAI()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if isinstance(user, list):
        messages.extend(user)
    else:
        messages.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=max_tokens,
        messages=messages,
    )
    return resp.choices[0].message.content


def ask_llm(user: str, system: str = "", max_tokens: int = 500, mock: str = "[MOCK] set ANTHROPIC_API_KEY for real output") -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return mock
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text
