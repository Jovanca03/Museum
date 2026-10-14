"""BGE-M3 embeddings served locally by Ollama."""
from __future__ import annotations
import os
import ollama
from langchain_core.embeddings import Embeddings

class BGE_M3_Embeddings(Embeddings):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
        self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embed(model=self.model_name, input=texts)
        vectors = response.get("embeddings", [])
        if len(vectors) != len(texts):
            raise RuntimeError(f"Jumlah embedding tidak sesuai: {len(vectors)} untuk {len(texts)} teks.")
        if any(len(v) != 1024 for v in vectors):
            raise RuntimeError("Dimensi embedding BGE-M3 harus 1024.")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
