import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CalenderEvent(BaseModel):
    name:str
    date:str
    participants: list[str]

completion = client.chat.completions.parse(
    model="gpt-4o",
    messages=[
    {
        "role":"system",
        "content":"Extract the evet information"
    },
    {
        "role":"user",
        "content":"Geetha and Samitha are going to a birthday party on 09/11/2026 at 6pm"
    }
    ],
    response_format=CalenderEvent


)

event = completion.choices[0].message.parsed

print(event)
