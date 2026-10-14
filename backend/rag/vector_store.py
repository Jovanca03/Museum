"""Chroma store for the existing BGE-M3 museum index."""
from __future__ import annotations
import os
from pathlib import Path
from langchain_chroma import Chroma
from rag.embeddings import BGE_M3_Embeddings

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = Path(os.getenv("CHROMA_DB_DIR", str(ROOT / "data" / "chroma_db"))).resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "museum_cakraningrat_bge_m3")

def get_vector_store():
    if not DB_DIR.exists():
        raise FileNotFoundError(f"Database Chroma belum ditemukan: {DB_DIR}. Salin database yang sudah diindeks ke folder data/chroma_db.")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=BGE_M3_Embeddings(),
    )
