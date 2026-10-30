"""
USE CASE 2 (Beginner): RAG from scratch - FAQ assistant
Pipeline: documents -> chunk -> vectorize (TF-IDF) -> retrieve top-k (cosine) -> augment prompt -> LLM.
Next days we swap TF-IDF for real embeddings + a vector DB. The PIPELINE stays the same.
Run:  python usecase2_rag_from_scratch.py
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm import call_llm

DOCS = """Refund policy: Customers can request a refund within 30 days of purchase. Refunds take 5-7 business days.
Shipping: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days and costs $15.
Password reset: Click 'Forgot password' on the login page. A reset link is emailed and expires in 1 hour.
Support hours: Our support team is available Monday to Friday, 9am to 6pm IST.
Premium plan: The premium plan costs $20 per month and includes priority support and 100GB storage."""

def chunk(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]  # 1 line = 1 chunk

class SimpleRAG:
    def __init__(self, docs: str):
        self.chunks = chunk(docs)
        self.vec = TfidfVectorizer(stop_words="english")
        self.matrix = self.vec.fit_transform(self.chunks)

    def retrieve(self, q: str, k: int = 2):
        sims = cosine_similarity(self.vec.transform([q]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in top]

    def ask(self, q: str, k: int = 2) -> str:
        hits = self.retrieve(q, k)
        context = "\n".join(c for c, _ in hits)
        prompt = f"Answer using ONLY the context.\nContext:\n{context}\n\nQuestion: {q}"
        print(f"  retrieved: {[(c[:25], round(s, 2)) for c, s in hits]}")
        return call_llm(prompt)

if __name__ == "__main__":
    rag = SimpleRAG(DOCS)
    for q in ["How long do refunds take?", "What does express shipping cost?",
              "When can I contact support?"]:
        print("Q:", q, "\nA:", rag.ask(q), "\n")
