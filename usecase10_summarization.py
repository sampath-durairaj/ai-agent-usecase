"""Use Case 1: Summarizer with a prompt template.
Learn: client setup, system prompt, templating, parameters (max_tokens, temperature)."""
import os
from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY
MODEL = os.getenv("MODEL", "claude-sonnet-5-5")

TEMPLATE = "Summarize the text below in {n} bullet points for a {audience}.\n\nTEXT:\n{text}"

def summarize(text: str, n: int = 3, audience: str = "beginner") -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        temperature=0.2,
        system="You are a concise technical writer.",
        messages=[{"role": "user", "content": TEMPLATE.format(n=n, audience=audience, text=text)}],
    )
    return resp.content[0].text

if __name__ == "__main__":
    sample = ("Retrieval-Augmented Generation (RAG) combines a retriever that finds relevant "
              "documents with a language model that writes answers grounded in them. This "
              "reduces hallucinations and lets models use private, up-to-date data without retraining.")
    print(summarize(sample, n=3, audience="manager"))