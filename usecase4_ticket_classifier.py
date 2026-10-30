"""
USE CASE 1 (Beginner): Support-ticket classifier with prompt template + structured (JSON) output
Concepts: prompt templates, structured output, validation, retry on bad JSON.
Run:  python usecase1_ticket_classifier.py
"""
import json
from llm import call_llm

PROMPT = """You are a support triage assistant.
Classify the ticket. Reply with ONLY valid JSON with keys:
category (billing|technical|account|general), priority (low|medium|high), summary (<=60 chars).

Ticket: {ticket}"""

VALID_CAT = {"billing", "technical", "account", "general"}
VALID_PRI = {"low", "medium", "high"}

def classify(ticket: str, retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        raw = call_llm(PROMPT.format(ticket=ticket))
        try:
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            assert data["category"] in VALID_CAT and data["priority"] in VALID_PRI
            return data
        except (ValueError, KeyError, AssertionError):
            print(f"  retry {attempt + 1}: bad output -> {raw[:50]!r}")
    return {"category": "general", "priority": "low", "summary": "unparsed"}

if __name__ == "__main__":
    tickets = [
        "I was charged twice for my invoice this month, please refund",
        "The app crashes with an error when I upload a photo",
        "I forgot my password and can't login to my account",
        "Do you have a student discount?",
    ]
    for t in tickets:
        print(t, "\n ->", classify(t))
