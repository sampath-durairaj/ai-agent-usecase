"""
Day 1 - Use Case 1 (Beginner): Email Tone Rewriter
Concepts: first LLM call, system prompt vs user prompt, parameterising a prompt.
Run: python day01/usecase1_tone_rewriter.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import ask_openaillm

SYSTEM = (
    "You are an expert business writing assistant. Rewrite the user's email in the "
    "requested tone. Keep the meaning, keep it under 120 words, output only the email."
)


def rewrite(email: str, tone: str) -> str:
    prompt = f"Tone: {tone}\n\nEmail:\n{email}"
    return ask_openaillm(prompt, system=SYSTEM, mock=f"[MOCK {tone}] {email}")


if __name__ == "__main__":
    draft = "hey, the report is late again. fix it today or we have a problem."
    for tone in ["polite", "friendly", "formal"]:
        print(f"--- {tone} ---")
        print(rewrite(draft, tone), "\n")