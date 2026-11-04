"""Day 1 - Use Case 1 (Beginner): Prompt template + structured JSON output.
Scenario: turn a messy customer email into a structured support ticket.
Concepts: LLM call, system prompt, prompt template, JSON parsing, validation.
"""
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM = (
    "You are a support triage assistant. Reply with ONLY valid JSON with keys: "
    "category (billing|bug|feature_request|other), priority (low|medium|high), "
    "summary (max 15 words), customer_sentiment (positive|neutral|negative)."
)
TEMPLATE = "Customer email:\n\"\"\"\n{email}\n\"\"\"\nReturn the ticket JSON."


def email_to_ticket(email: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        response_format={"type": "json_object"},  # forces the model to return valid JSON
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(email=email)},
        ],
    )
    text = resp.choices[0].message.content.strip()
    ticket = json.loads(text)  # raises if model returned non-JSON
    assert ticket["priority"] in {"low", "medium", "high"}
    return ticket


if __name__ == "__main__":
    email = ("Hi, I was charged twice for my subscription this month and nobody "
             "answers my chats. Please fix this today!!")
    print(json.dumps(email_to_ticket(email), indent=2))
