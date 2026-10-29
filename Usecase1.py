import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role":"system",
            "content":"You are a helpful assistant"
        },
        {
            "role" :"user",
            "content":"Where is New Delhi located"
        }
    ]

)
response = completion.choices[0].message.content
print(response)