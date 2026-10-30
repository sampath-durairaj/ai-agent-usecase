"""Tiny LLM wrapper. Uses a real model if an API key exists, else a rule-based mock
so every exercise runs offline."""
import os, json, re

def call_llm(prompt: str) -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            max_tokens=500, messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    return _mock(prompt)

def _mock(prompt: str) -> str:
    """Very dumb stand-in so the pipeline is testable without internet."""
    p = prompt.lower()
    if "classify" in p:
        text = p.split("ticket:")[-1]
        if any(w in text for w in ["refund", "charged", "invoice", "payment"]):
            cat, pri = "billing", "high"
        elif any(w in text for w in ["crash", "error", "bug", "not working"]):
            cat, pri = "technical", "high"
        elif any(w in text for w in ["password", "login", "account"]):
            cat, pri = "account", "medium"
        else:
            cat, pri = "general", "low"
        return json.dumps({"category": cat, "priority": pri,
                           "summary": re.sub(r"\s+", " ", text.strip())[:60]})
    if "context:" in p:
        ctx = prompt.split("Context:")[-1].split("Question:")[0].strip()
        return "(mock answer based on context) " + ctx.split("\n")[0]
    return "(mock) " + prompt[:80]
