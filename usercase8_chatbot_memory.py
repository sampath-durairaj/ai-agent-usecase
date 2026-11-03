"""
USE CASE 2 (Beginner): Command-line chatbot that REMEMBERS the conversation.
Concepts: the LLM is stateless - YOU resend the history each turn (this is "memory").
Try: tell it your name, then ask "what's my name?". Then set MAX_TURNS=2 and see it forget.
"""
from common import ask_openaillm

MAX_TURNS = 10  # sliding window: keep only the last N user/assistant pairs

def chat():
    history = []
    print("Chatbot ready. Type 'quit' to exit.")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        history.append({"role": "user", "content": user})
        window = history[-MAX_TURNS * 2:]          # trim old messages
        reply = ask_openaillm(window, system="You are a friendly tutor. Keep answers short.")
        history.append({"role": "assistant", "content": reply})
        print("Bot:", reply)

if __name__ == "__main__":
    chat()
