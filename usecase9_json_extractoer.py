"""Use Case 2: Extract structured JSON from messy text (e.g., support emails).
Learn: output formatting, JSON parsing, validation, retry on bad output."""
import os, json
#from anthropic import Anthropic
from openai import OpenaAI

client = OpenaAI()
MODEL = os.getenv("MODEL", "claude-sonnet-5-5")

PROMPT = """Extract fields from the email. Reply with ONLY valid JSON, keys:
customer_name (string), product (string), issue (string), urgency ("low"|"medium"|"high").

EMAIL:
{email}"""

def extract(email: str, retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        resp = client.messages.create(
            model=MODEL, max_tokens=300, temperature=0,
            messages=[{"role": "user", "content": PROMPT.format(email=email)}],
        )
        raw = resp.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            data = json.loads(raw)
            assert {"customer_name", "product", "issue", "urgency"} <= data.keys()
            return data
        except (json.JSONDecodeError, AssertionError):
            print(f"Bad output, retry {attempt + 1}")
    raise ValueError("Could not get valid JSON")

if __name__ == "__main__":
    mail = "Hi, I'm Priya. My Acme X200 router keeps rebooting every hour and I work from home - need help ASAP!"
    print(json.dumps(extract(mail), indent=2))