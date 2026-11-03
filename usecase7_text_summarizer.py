"""
USE CASE 1 (Beginner): Text summarizer with a reusable prompt template.
Concepts: API call, system prompt vs user prompt, prompt templates, parameters.
"""
from common import ask_openaillm

TEMPLATE = """Summarize the text below for a {audience}.
Rules: max {n} bullet points, plain language.

TEXT:
{text}"""

def summarize(text, audience="busy manager", n=3):
    prompt = TEMPLATE.format(audience=audience, n=n, text=text)
    return ask_openaillm(prompt, system="You are a precise summarization assistant.")

if __name__ == "__main__":
    article = ("Retrieval-Augmented Generation (RAG) combines a retriever that finds relevant "
               "documents with a language model that writes answers grounded in them. This reduces "
               "hallucinations and lets models use private, up-to-date data without retraining.")
    print(summarize(article))
    print("-----")
    print(summarize(article, audience="10-year-old", n=2))
