"""Day 1 - Use Case 2 (Beginner): Your first AI Agent, no framework.
Scenario: assistant that can do math and look up order status via tools.
Concepts: tool schema, tool_use -> run tool -> tool_result loop (the core of ALL agents).
"""

import json
from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ORDERS = {"A100": "shipped", "A101": "processing", "A102": "delivered"}


def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "error: invalid characters"
    return str(eval(expression))  # fine for a demo; never eval untrusted input in prod


def get_order_status(order_id: str) -> str:
    return ORDERS.get(order_id.upper(), "order not found")


TOOLS = [
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_order_status", "description": "Get status of an order by id.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
]
IMPL = {"calculator": calculator, "get_order_status": get_order_status}


def run_agent(question: str, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):  # agent loop with a safety limit
        resp = client.chat.completions.create(model=MODEL, max_tokens=500,
                                              tools=TOOLS, messages=messages)
        msg = resp.choices[0].message
        if not msg.tool_calls:  # no tool requested -> final answer
            return msg.content
        messages.append(msg)  # keep the assistant's tool request in history
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)  # arguments arrive as a JSON string
            print(f"  [tool] {call.function.name}({args})")
            out = IMPL[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
    return "Stopped: max steps reached"


if __name__ == "__main__":
    print(run_agent("What is the status of order A101, and what is 17*23 + 5?"))
