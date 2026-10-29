import os
import json
import requests
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_weather(latitude, longitude):
    """Public API that returns the current weather for a location."""
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["current"]


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

messages = [
    {"role": "system", "content": "You are a helpful weather assistant."},
    {"role": "user", "content": "What's the weather like in Paris today?"},
]

# Step 1: model decides whether to call the tool
completion = client.chat.completions.create(
    model="gpt-4o", messages=messages, tools=tools
)
msg = completion.choices[0].message


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    raise ValueError(f"Unknown function: {name}")


# Step 2: run any tool calls
if msg.tool_calls:
    messages.append(msg)  # append the assistant message once
    for tool_call in msg.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = call_function(tool_call.function.name, args)
        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
        )


# Step 3: get a structured final answer
class WeatherResponse(BaseModel):
    temperature: float = Field(description="The current temperature in celsius for the given location.")
    response: str = Field(description="A natural language response to the user's question.")


completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=WeatherResponse,
)

final_response = completion_2.choices[0].message.parsed
print(final_response.temperature)
print(final_response.response)