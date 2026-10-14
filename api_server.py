"""Local HTTP API for Unity AI Guide: Chroma semantic RAG (BGE-M3) + Ollama Qwen.

Place this file in the Python project root, beside rag/ and data/.
No external LLM API key is used.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA_PATH = Path(os.getenv("MUSEUM_CATALOG_PATH", str(ROOT / "data" / "cakra_data.json")))
CHROMA_PATH = Path(os.getenv("CHROMA_DB_DIR", str(ROOT / "data" / "chroma_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "museum_cakraningrat_bge_m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = "bge-m3"  # matches rag/embeddings.py in the existing project
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:1.7b")
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

app = FastAPI(
    title="Museum AI Guide API",
    version="1.0.0",
    description="Local Unity bridge using BGE-M3 semantic retrieval and Qwen via Ollama.",
)


class HistoryTurn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    artifact_id: str = Field(min_length=1, max_length=128)
    persona: str = "Budayawan"
    gender: str = "female"
    museum_id: str = "MUSEUM_CAKRANINGRAT"
    conversation_history: list[HistoryTurn] = Field(default_factory=list, max_length=10)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float | None = None
    source: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    artifact_id: str
    persona: str
    retrieval_mode: str
    query_used: str
    retrieved_chunks: list[RetrievedChunk]


def _load_catalog() -> list[dict[str, Any]]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Katalog tidak ditemukan: {DATA_PATH}. Letakkan data/cakra_data.json di folder data/ project."
        )
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        artifacts = payload
    elif isinstance(payload, dict) and isinstance(payload.get("selected_artifacts"), list):
        artifacts = payload["selected_artifacts"]
    elif isinstance(payload, dict) and isinstance(payload.get("artifacts"), list):
        artifacts = payload["artifacts"]
    elif isinstance(payload, dict) and payload.get("artifact_id"):
        artifacts = [payload]
    else:
        raise ValueError("Format katalog JSON tidak dikenali.")
    return [item for item in artifacts if isinstance(item, dict)]


@lru_cache(maxsize=1)
def _get_vector_store():
    """Use the project's configured BGE-M3 embedding adapter and collection."""
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(f"Database Chroma tidak ditemukan: {CHROMA_PATH}")
    # This package includes rag/ and uses the same BGE-M3 collection as the indexer.
    from rag.vector_store import get_vector_store
    return get_vector_store()


def _get_ollama_client():
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("Paket Python 'ollama' belum terpasang. Jalankan pip install -r requirements-unity.txt.") from exc
    return ollama.Client(host=OLLAMA_HOST)


def _persona_guidance(persona: str) -> str:
    value = persona.lower()
    if any(term in value for term in ("arkeolog", "archaeolog")):
        return "Gunakan sudut pandang arkeologi jika didukung konteks: material, bentuk, fungsi, dan bukti artefak."
    if any(term in value for term in ("histor", "sejarawan", "sejarah")):
        return "Gunakan sudut pandang sejarah, kronologi, dan konteks masa jika didukung konteks."
    if any(term in value for term in ("budaya", "budayawan", "cultural")):
        return "Gunakan sudut pandang budaya dan praktik sosial hanya sejauh tercantum dalam konteks."
    if any(term in value for term in ("pedagog", "pendidik", "educator")):
        return "Jelaskan secara edukatif dan mudah dipahami, tanpa menambahkan fakta di luar konteks."
    return "Gunakan pendekatan pemandu museum yang informatif dan mudah dipahami."



def _retrieve(question: str, artifact_id: str, artifact_name: str, top_k: int):
    query = (
        f"Artefak museum: {artifact_name}. "
        f"Pertanyaan pengunjung: {question}"
    ).strip()

    store = _get_vector_store()
    results_by_id = {}

    def add_results(pairs):
        for doc, score in pairs:
            metadata = getattr(doc, "metadata", {}) or {}
            text = str(getattr(doc, "page_content", "")).strip()

            # Jangan menerima chunk artefak lain atau teks kosong.
            if metadata.get("artifact_id") != artifact_id or not text:
                continue

            chunk_id = str(
                metadata.get("chunk_id")
                or metadata.get("id")
                or getattr(doc, "id", None)
                or text[:100]
            )

            item = {
                "chunk_id": chunk_id,
                "score": float(score) if score is not None else None,
                "source": str(
                    metadata.get("source")
                    or metadata.get("source_path")
                    or metadata.get("source_file")
                    or "Sumber koleksi museum"
                ),
                "text": text,
            }

            # Hindari chunk yang sama masuk lebih dari sekali.
            if chunk_id not in results_by_id:
                results_by_id[chunk_id] = item

    try:
        # Pencarian utama berdasarkan pertanyaan pengunjung.
        pairs = store.similarity_search_with_score(
            query,
            k=max(top_k, 8),
            filter={"artifact_id": artifact_id},
        )
        add_results(pairs)

        # Pencarian tambahan untuk fakta penting yang mungkin
        # tidak terambil oleh pencarian semantik utama.
        supplemental_queries = [
            f"{artifact_name} kronologi kehilangan jumlah komponen "
            "hilang jenis material tanggal wawancara staf museum",
            f"{artifact_name} alur penjualan penadah Bangkalan Surabaya "
            "nilai transaksi Rp3.000.000 barang curian bersama artefak lain",
            f"{artifact_name} status terkini keberadaan barang "
            "dugaan peleburan konfirmasi resmi kepolisian",
            f"{artifact_name} sumber informasi catatan wawancara "
            "inventaris jumlah bilah instrumen saron",
        ]

        for extra_query in supplemental_queries:
            pairs = store.similarity_search_with_score(
                extra_query,
                k=4,
                filter={"artifact_id": artifact_id},
            )
            add_results(pairs)

    except Exception as exc:
        raise RuntimeError(
            f"Pencarian semantic Chroma/BGE-M3 gagal: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    results = list(results_by_id.values())

    # Batasi konteks agar tidak terlalu panjang, tetapi pertahankan
    # seluruh chunk unik yang ditemukan oleh pencarian tambahan.
    return query, results

def _generate_answer(request: ChatRequest, artifact_name: str, retrieved: list[dict[str, Any]]) -> str:
    if not retrieved or not any(item["text"] for item in retrieved):
        return (
            "Sorry, I could not find enough information in the collection sources "
            "to answer that question."
            if _question_language(request.question) == "English"
            else "Maaf, saya belum menemukan informasi yang cukup dalam sumber koleksi untuk menjawab pertanyaan itu."
        )

    language = _question_language(request.question)
    history = "\\n\\n".join(
        f"Visitor: {turn.question}\\nAI Guide: {turn.answer}"
        for turn in request.conversation_history[-5:]
    ) or "No previous conversation."

    context = "\\n\\n---\\n\\n".join(
        f"[Source chunk: {item['chunk_id']} | Source: {item['source']}]\\n{item['text']}"
        for item in retrieved
    )

    if language == "English":
        system_message = (
            "You are a museum AI guide. You MUST answer entirely in English because "
            "the visitor's current question is in English. Do not answer in Indonesian. "
            "Use only facts explicitly supported by the supplied context. Do not invent facts. "
            "If the context does not contain the requested information, say so clearly."
        )
        prompt = f"""SESSION
Museum ID: {request.museum_id}
Active artifact: {artifact_name} (ID: {request.artifact_id})
Guide persona: {request.persona}
Persona gender: {request.gender}
Persona guidance: {_persona_guidance(request.persona)}

ATURAN FAKTA — WAJIB DIPATUHI

1. Jawab menggunakan bahasa yang sama dengan pertanyaan pengunjung.
2. Gunakan hanya fakta yang didukung oleh konteks hasil retrieval.
3. Gunakan seluruh potongan relevan yang tersedia, bukan hanya potongan pertama.
4. Jangan mencampurkan fakta dari artefak lain. Artefak aktif adalah {request.artifact_id}.
5. Jika relevan dan tersedia, jelaskan jumlah serta jenis komponen yang hilang, tanggal laporan atau diketahuinya kehilangan, sumber informasi, catatan inventaris, alur penjualan, nilai transaksi, dan status terkini.
6. Bedakan dengan jelas:
   - Fakta yang tercatat dalam inventaris.
   - Keterangan hasil wawancara dengan pihak museum.
   - Informasi dari catatan kasus atau pemberitaan.
   - Informasi yang belum dikonfirmasi secara resmi.
7. Jika kehilangan komponen dinyatakan terkonfirmasi melalui wawancara, sampaikan sebagai kehilangan yang dilaporkan berdasarkan wawancara. Jangan menyebut jumlah atau identitas komponennya sekadar dugaan jika konteks menyatakannya telah terkonfirmasi.
8. Jika nilai Rp3.000.000 disebut untuk barang curian bersama artefak lain, jelaskan cakupan nilai tersebut. Jangan menyatakannya sebagai harga khusus artefak aktif.
9. Jangan menyatakan barang telah dilebur apabila konteks hanya menyebut dugaan tersebut belum terkonfirmasi secara resmi.
10. Jangan menyatakan seluruh perangkat gamelan hilang jika konteks hanya menyebut komponen tertentu.
11. Jangan mengarang harga khusus artefak, identitas pelaku, waktu pasti pencurian, atau hasil penyelidikan.
12. Jika informasi tidak ditemukan dalam konteks, nyatakan keterbatasannya secara spesifik.
13. Berikan jawaban lengkap, terstruktur, dan mudah dipahami. Hindari jawaban terlalu umum jika konteks memuat fakta pendukung yang relevan.
14. Jangan menyebut skor retrieval, nama model, chunk ID, atau metadata teknis kepada pengunjung.
15. Jawab sesuai fokus pertanyaan. Jika pengunjung bertanya tentang kronologi kehilangan dan penjualan, prioritaskan urutan kejadian, jumlah komponen yang hilang, sumber informasi, jalur penjualan, nilai transaksi, dan status terakhir barang.
16. Jangan menambahkan informasi tentang nilai dokumenter, inskripsi, dimensi, pelestarian, atau rekonstruksi digital jika tidak relevan dengan pertanyaan.
17. Bedakan status kehilangan komponen dari status dugaan peleburannya. Jika kehilangan didukung wawancara staf museum, jangan menyebut kehilangan itu belum terkonfirmasi hanya karena belum ada konfirmasi resmi kepolisian mengenai dugaan peleburannya.
18. Jelaskan bahwa Rp3.000.000 merupakan nilai barang curian bersama artefak lainnya, bukan harga khusus komponen Gamelan Ratna Dumilah.
19. Jangan mengarang tanggal atau waktu kejadian jika sumber hanya mencatat waktu kehilangan diketahui atau dilaporkan.

CONVERSATION HISTORY
{history}

VISITOR'S QUESTION
{request.question}

RETRIEVED CONTEXT
{context}

Write a clear, concise museum-guide answer in English."""
    else:
        system_message = (
            "Anda adalah pemandu museum AI. Anda WAJIB menjawab seluruhnya dalam "
            "Bahasa Indonesia karena pertanyaan pengunjung saat ini menggunakan Bahasa Indonesia. "
            "Gunakan hanya fakta yang didukung konteks. Jangan mengarang fakta."
        )
        prompt = f"""IDENTITAS SESI
Museum ID: {request.museum_id}
Artefak aktif: {artifact_name} (ID: {request.artifact_id})
Persona pemandu: {request.persona}
Gender persona: {request.gender}
Panduan persona: {_persona_guidance(request.persona)}

ATURAN FAKTA
1. Jawab seluruhnya dalam Bahasa Indonesia.
2. Gunakan hanya fakta yang secara eksplisit didukung oleh konteks hasil pencarian di bawah.
3. Jangan menambahkan tafsir, simbolisme, filosofi, kronologi, atau klaim umum yang tidak disebutkan dalam konteks.
4. Jangan menyatukan informasi dari artefak lain; artefak aktif adalah {request.artifact_id}.
5. Jika fakta yang diminta tidak ditemukan, katakan bahwa informasi tersebut belum tersedia dalam sumber yang ditemukan.
6. Riwayat percakapan hanya untuk memahami rujukan, bukan sumber fakta.
7. Jangan menyebut skor retrieval, nama model, chunk ID, atau metadata teknis.

RIWAYAT PERCAKAPAN
{history}

PERTANYAAN PENGUNJUNG
{request.question}

KONTEN HASIL PENCARIAN SEMANTIK BGE-M3
{context}

Tulis jawaban pemandu museum yang jelas dan ringkas dalam Bahasa Indonesia."""

    client = _get_ollama_client()
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat(
            model=CHAT_MODEL,
            messages=messages,
            think=False,
        )
    except TypeError:
        response = client.chat(
            model=CHAT_MODEL,
            messages=messages,
        )

    answer = getattr(getattr(response, "message", None), "content", None)
    if not answer and isinstance(response, dict):
        message_data = response.get("message")
        if isinstance(message_data, dict):
            answer = message_data.get("content")

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"Model {CHAT_MODEL} tidak mengembalikan teks jawaban.")

    return answer.strip()


def _question_language(question: str) -> str:
    """Detect English vs Indonesian for the visitor's current question."""
    import re

    text = question.lower()
    english_markers = {
        "what", "why", "when", "where", "who", "which", "how",
        "is", "are", "was", "were", "does", "do", "did", "can",
        "could", "would", "should", "the", "and", "of", "for",
        "with", "in", "on", "this", "that", "tell", "explain",
        "describe", "please", "history", "historical", "cultural",
        "significance", "meaning",
    }
    indonesian_markers = {
        "apa", "siapa", "kapan", "dimana", "di", "ke", "dari",
        "yang", "dan", "atau", "untuk", "dengan", "ini", "itu",
        "bagaimana", "mengapa", "jelaskan", "ceritakan", "sejarah",
        "budaya", "makna", "adalah", "tolong",
    }
    words = set(re.findall(r"[a-z]+", text))
    english_score = len(words & english_markers)
    indonesian_score = len(words & indonesian_markers)
    return "English" if english_score > indonesian_score else "Indonesian"


@app.get("/health")
def health():
    result: dict[str, Any] = {
        "api": "ok",
        "catalog_path": str(DATA_PATH),
        "catalog_exists": DATA_PATH.is_file(),
        "chroma_path": str(CHROMA_PATH),
        "chroma_path_exists": CHROMA_PATH.exists(),
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "ollama_host": OLLAMA_HOST,
    }
    try:
        store = _get_vector_store()
        result["chroma_count"] = int(store._collection.count())
        result["chroma"] = "ok"
    except Exception as exc:
        result["chroma"] = "error"
        result["chroma_error"] = f"{type(exc).__name__}: {exc}"
    try:
        client = _get_ollama_client()
        listed = client.list()
        models = listed.get("models", []) if isinstance(listed, dict) else getattr(listed, "models", [])
        names = []
        for model in models or []:
            if isinstance(model, dict):
                names.append(model.get("model") or model.get("name") or "")
            else:
                names.append(getattr(model, "model", None) or getattr(model, "name", None) or "")
        result["installed_models"] = names
        result["ollama"] = "ok"
        result["embedding_model_available"] = any(str(n).split(":")[0] == EMBED_MODEL for n in names)
        result["chat_model_available"] = CHAT_MODEL in names
    except Exception as exc:
        result["ollama"] = "error"
        result["ollama_error"] = f"{type(exc).__name__}: {exc}"
    result["ready"] = (
        result.get("chroma") == "ok"
        and result.get("ollama") == "ok"
        and result.get("chroma_count", 0) > 0
        and result.get("embedding_model_available") is True
        and result.get("chat_model_available") is True
    )
    return result


@app.get("/api/artifacts")
def list_artifacts():
    try:
        artifacts = _load_catalog()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "artifacts": [
            {
                "artifact_id": str(item.get("artifact_id", "")),
                "name": str(item.get("name") or item.get("vr_display_name") or item.get("artifact_id") or "Artefak"),
                "museum_id": str(item.get("museum_id") or "MUSEUM_CAKRANINGRAT"),
            }
            for item in artifacts
        ]
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Pertanyaan tidak boleh kosong.")
    try:
        artifacts = _load_catalog()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    artifact = next((item for item in artifacts if item.get("artifact_id") == request.artifact_id), None)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artefak aktif {request.artifact_id!r} tidak ada di katalog.")
    artifact_name = str(artifact.get("name") or artifact.get("vr_display_name") or request.artifact_id)

    try:
        query_used, retrieved = _retrieve(question, request.artifact_id, artifact_name, request.top_k)
        answer = _generate_answer(request, artifact_name, retrieved)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"AI Guide belum siap: {type(exc).__name__}: {exc}") from exc

    return ChatResponse(
        answer=answer,
        artifact_id=request.artifact_id,
        persona=request.persona,
        retrieval_mode="chroma-semantic-bge-m3",
        query_used=query_used,
        retrieved_chunks=[RetrievedChunk(**item) for item in retrieved],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
