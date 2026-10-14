"""
Demo AI Guide Museum Cakraningrat
RAG + Gemini + Persona + Context Manager

Prototype terminal untuk demonstrasi:
- Pemilihan persona
- Pemilihan artefak
- Percakapan berbasis RAG
- Context-aware conversation
- Ganti artefak
- Ganti persona
- Reset conversation history saat context berubah
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from context_manager import ConversationContext
from rag.catalog import find_artifact, load_catalog
from rag.chunking import build_chunks

# Import baru untuk arsitektur Vector Store ChromaDB
from rag.vector_store import get_vector_store

# ============================================================
# KONFIGURASI
# ============================================================

load_dotenv()

# Jalur data sudah mengarah ke hasil scraping terbaru
CATALOG_PATH = "data/cakra_data.json"
PERSONAS_PATH = "persona/personas.json"
CHROMA_DB_DIR = "data/chroma_db"

CHAT_MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    "gemini-3.6-flash",
)

GEMINI_MAX_RETRIES_503 = int(os.getenv("GEMINI_MAX_RETRIES_503", "0"))
GEMINI_RETRY_DELAY_503 = float(os.getenv("GEMINI_RETRY_DELAY_503", "0"))

# ============================================================
# LOAD PERSONA
# ============================================================

def load_personas(path: str = PERSONAS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

# ============================================================
# TOKENIZER & LEXICAL RETRIEVAL (FALLBACK)
# ============================================================

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

def lexical_retrieval(
    question: str,
    artifacts: list[dict[str, Any]],
    artifact_id: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    
    candidates = [
        artifact for artifact in artifacts
        if (not artifact_id or artifact.get("artifact_id") == artifact_id)
    ]

    query_tokens = _tokens(question)
    scored: list[dict[str, Any]] = []

    for artifact in candidates:
        for chunk in build_chunks(artifact):
            words = _tokens(chunk["text"])
            score = len(query_tokens & words) / max(len(query_tokens | words), 1)

            if score > 0:
                scored.append({
                    **chunk,
                    "score": score,
                    "retrieval_mode": "local-keyword",
                })

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

# ============================================================
# RETRIEVAL (CHROMA DB)
# ============================================================


def retrieve(
    question: str,
    artifacts: list[dict[str, Any]],
    artifact_id: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    try:
        # Menggunakan koleksi Chroma dan embedding BGE-M3 lokal.
        vector_store = get_vector_store()

        # Prioritaskan pencarian dalam artefak yang sedang aktif.
        if artifact_id:
            docs = vector_store.similarity_search(
                question,
                k=top_k,
                filter={"artifact_id": artifact_id},
            )

            # Jika tidak ada hasil dengan filter, coba pencarian
            # semantik dengan nama artefak sebagai konteks.
            if not docs:
                active_artifact = next(
                    (
                        artifact
                        for artifact in artifacts
                        if artifact.get("artifact_id") == artifact_id
                    ),
                    {},
                )
                artifact_name = active_artifact.get(
                    "name",
                    active_artifact.get("vr_display_name", ""),
                )
                query = f"{artifact_name}. {question}".strip()

                docs = vector_store.similarity_search(
                    query,
                    k=top_k,
                )
        else:
            docs = vector_store.similarity_search(
                question,
                k=top_k,
            )

        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "chunk_id": getattr(doc, "id", None),
                "retrieval_mode": "chroma-semantic-bge-m3",
            })

        if results:
            return results

    except Exception as exc:
        print(
            f"\n[Retrieval Error] Pencarian BGE-M3 gagal: "
            f"{type(exc).__name__}: {exc}"
        )

    # Fallback lokal jika pencarian semantik gagal atau kosong.
    print("[Retrieval Fallback] Menggunakan local keyword retrieval.")
    return lexical_retrieval(
        question,
        artifacts,
        artifact_id,
        top_k,
    )

# ============================================================
# FORMAT HASIL RAG & CONVERSATION HISTORY
# ============================================================

def context_from(results: list[dict[str, Any]]) -> str:
    if not results:
        return "Tidak ada konten yang berhasil diambil."
    formatted = []
    for item in results:
        metadata = item.get("metadata", {})
        chunk_id = item.get("chunk_id", "unknown")
        artifact_id = metadata.get("artifact_id", "unknown")
        source = metadata.get("source", "unknown")
        text = item.get("text", "")
        formatted.append(f"[chunk_id={chunk_id}; artifact_id={artifact_id}; source={source}]\n{text}")
    return "\n\n---\n\n".join(formatted)

def conversation_from_context(context: ConversationContext) -> str:
    if not context.conversation_history:
        return "Belum ada percakapan sebelumnya."
    history = []
    for turn in context.conversation_history[-5:]:
        history.append(f"Pengunjung: {turn['question']}\nAI Guide: {turn['answer']}")
    return "\n\n".join(history)

# ============================================================
# JAWABAN LOKAL & GENERATE ANSWER GEMINI
# ============================================================

def safe_local_answer(persona: dict[str, Any], results: list[dict[str, Any]]) -> str:
    if not results:
        return (f"Maaf, layanan AI sedang tidak tersedia sementara. Sebagai {persona['name']}, "
                "saya belum dapat memberikan jawaban karena tidak ada informasi yang relevan pada knowledge base.")
    
    snippets = []
    for item in results[:3]:
        text = item.get("text", "").strip()
        if not text:
            continue
        metadata = item.get("metadata", {})
        source = metadata.get("source", "sumber museum")
        artifact_id = metadata.get("artifact_id", "unknown")
        snippets.append(f"[artifact_id={artifact_id}; source={source}]\n{text}")
        
    return ("Maaf, layanan AI sedang mengalami gangguan sementara. Namun, informasi yang berhasil "
            "ditemukan pada knowledge base adalah:\n\n" + "\n\n---\n\n".join(snippets) + 
            "\n\nInformasi di atas hanya menggunakan data yang berhasil ditemukan pada knowledge base.")

def generate_answer(
    question: str,
    persona: dict[str, Any],
    results: list[dict[str, Any]],
    context: ConversationContext,
) -> str:
    import ollama

    conversation_history = conversation_from_context(context)
    retrieved_context = context_from(results)

    if not results:
        return (
            "Maaf, saya belum menemukan informasi yang cukup "
            "dalam knowledge base untuk menjawab pertanyaan tersebut."
        )

    prompt = f"""
Anda adalah {persona['name']}, pemandu digital di Museum Cakraningrat.

DESKRIPSI PERSONA:
{persona['description']}

GAYA JAWABAN:
{persona['answer_style']}

KONTEKS SESI:
Museum: {context.museum_id}
Artefak aktif: {context.artifact_id}
Persona: {context.persona}
Gender: {context.gender}

RIWAYAT PERCAKAPAN:
{conversation_history}


ATURAN UTAMA:
1. KONTEN TERAMBIL DARI KNOWLEDGE BASE adalah satu-satunya sumber fakta.
2. Jawab inti pertanyaan pada kalimat pertama.
3. Jika pertanyaan menanyakan fungsi, kegunaan, atau penggunaan artefak,
   cari dan sampaikan secara langsung informasi penggunaan yang tertulis
   dalam konteks.
4. Jika pertanyaan menanyakan asal, jawab berdasarkan lokasi asal yang
   tercantum dalam sumber. Jangan menggantinya dengan lokasi penyimpanan.
5. Jika pertanyaan menanyakan bahan, ukuran, periode, atau kondisi,
   sampaikan hanya data yang sesuai dengan pertanyaan.
6. Jangan mendahului jawaban faktual dengan simbolisme, filosofi,
   estetika, atau interpretasi persona.
7. Jangan membuat klaim historis, makna simbolis, atau fungsi tambahan
   yang tidak dinyatakan dalam konteks.
8. Jangan mengubah fakta, nama, tanggal, atau tingkat kepastian sumber.
9. Jangan mencampurkan informasi dari artefak lain.
10. Gunakan riwayat percakapan hanya untuk memahami rujukan seperti
    "benda ini" atau pertanyaan lanjutan. Jangan menganggap jawaban AI
    sebelumnya sebagai fakta jika tidak didukung konteks sumber.
11. Persona hanya menentukan sudut penyampaian dan gaya bahasa.
    Persona tidak boleh mengubah fakta atau menambahkan informasi.
12. Jika sumber tidak memuat jawaban, katakan bahwa informasi tersebut
    belum tersedia dalam data museum.
13. Jawab dalam Bahasa Indonesia secara ringkas, jelas, dan langsung.
14. Jangan mengulang sapaan.
15. Jangan menampilkan ID artefak atau metadata teknis kepada pengunjung.

PERTANYAAN PENGUNJUNG:
{question}

KONTEN TERAMBIL DARI KNOWLEDGE BASE:
{retrieved_context}
"""

    model_name = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:1.7b")

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Anda adalah AI Guide Museum Cakraningrat. "
                        "Ikuti instruksi dan batasan sumber dalam prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            think=False,
        )

        answer = response.message.content
        if answer and answer.strip():
            return answer.strip()

        print("\n[Ollama] Model tidak mengembalikan jawaban.")
        return safe_local_answer(persona, results)

    except Exception as exc:
        print(
            f"\n[Ollama Error] Gagal menghasilkan jawaban: "
            f"{type(exc).__name__}: {exc}"
        )
        return safe_local_answer(persona, results)
    
    
print("\n========== DEBUG RAG ==========")

for i, item in enumerate(results, start=1):
    metadata = item.get("metadata", {})
    print(f"\n--- Chunk {i} ---")
    print("Artifact ID:", metadata.get("artifact_id"))
    print("Source:", metadata.get("source"))
    print("Isi chunk:")
    print(item.get("text", ""))

print("\n======== AKHIR DEBUG RAG ========")

# ============================================================
# ANTARMUKA MENU (PERSONA, ARTEFAK, NAVIGASI)
# ============================================================
def choose_persona(personas: dict[str, Any]) -> str:
    print("\n" + "=" * 30)
    print("PILIH PERSONA AI GUIDE")
    print("=" * 30)
    persona_keys = list(personas.keys())
    for number, key in enumerate(persona_keys, start=1):
        print(f"{number}. {personas[key]['name']}")
    while True:
        choice = input("\nPilihan persona: ").strip()
        try:
            index = int(choice) - 1
            if 0 <= index < len(persona_keys):
                return persona_keys[index]
        except ValueError:
            pass
        print("Pilihan tidak valid. Masukkan nomor sesuai daftar.")

def choose_artifact(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    print("\n" + "=" * 30)
    print("PILIH ARTEFAK")
    print("=" * 30)
    for number, artifact in enumerate(artifacts, start=1):
        print(f"{number}. {artifact['name']}")
    while True:
        choice = input("\nPilihan artefak: ").strip()
        try:
            index = int(choice) - 1
            if 0 <= index < len(artifacts):
                return artifacts[index]
        except ValueError:
            pass
        print("Pilihan tidak valid. Masukkan nomor sesuai daftar.")

def persona_greeting(persona: dict[str, Any]) -> str:
    name = persona["name"]
    greetings = {
        "Arkeolog": "Selamat datang di Museum Cakraningrat. Saya pemandu dengan perspektif arkeologis...",
        "Historian": "Selamat datang di Museum Cakraningrat. Saya pemandu dengan perspektif sejarah...",
        "Budayawan": "Selamat datang di Museum Cakraningrat. Saya pemandu dengan perspektif budaya..."
    }
    return greetings.get(name, f"Selamat datang di Museum Cakraningrat. Saya adalah {name}, pemandu Anda. Silakan pilih artefak.")

def artifact_greeting(persona: dict[str, Any], artifact: dict[str, Any]) -> str:
    return (f"\nBaik, sekarang kita akan mengeksplorasi {artifact['name']}.\n\n"
            f"Saya akan membantu Anda memahami artefak ini dari perspektif {persona['name']}.\n\n"
            "Silakan tanyakan apa pun yang ingin Anda ketahui mengenai artefak ini.")

def conversation_hint() -> None:
    print("\n[Petunjuk] Ketik 'kembali' untuk ke menu awal, atau 'keluar' untuk mengakhiri.")


def show_navigation() -> str:
    print("\n" + "=" * 30)
    print("MENU NAVIGASI")
    print("=" * 30)
    print("1. Ganti artefak")
    print("2. Ganti museum")
    print("3. Ganti persona AI")
    print("4. Kembali ke menu utama")
    print("5. Akhiri program")

    while True:
        choice = input("\nPilih aksi (1-5): ").strip()

        if choice in {"1", "2", "3", "4", "5"}:
            return choice

        print("Pilihan tidak valid. Masukkan nomor 1-5.")

def closing_message() -> str:
    return "SESI AI GUIDE SELESAI\nTerima kasih telah mengunjungi Museum Cakraningrat."

# ============================================================
# ALUR PERCAKAPAN
# ============================================================

def answer_question(question: str, artifacts: list[dict[str, Any]], persona: dict[str, Any], context: ConversationContext) -> None:
    print("\nMencari informasi artefak...")
    results = retrieve(question, artifacts, context.artifact_id, top_k=4)
    retrieval_mode = results[0]["retrieval_mode"] if results else "tidak ada hasil"
    
    print(f"Mode pencarian: {retrieval_mode}")
    print("AI Guide sedang menjawab...")
    
    answer = generate_answer(question, persona, results, context)
    context.add_turn(question, answer)
    
    print("\nAI Guide:")
    print(answer)
    print(f"\n[Context] artifact={context.artifact_id} | persona={context.persona} | turns={len(context.conversation_history)}")


def conversation_loop(
    artifacts: list[dict[str, Any]],
    persona: dict[str, Any],
    context: ConversationContext,
) -> str:
    while True:
        conversation_hint()
        question = input("\nAnda: ").strip()
        command = question.lower()

        if command in {"kembali", "keluar", "exit"}:
            return "menu"

        if not question:
            print("Pertanyaan tidak boleh kosong.")
            continue

        answer_question(question, artifacts, persona, context)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    load_dotenv()
    artifacts = load_catalog(CATALOG_PATH)
    personas = load_personas(PERSONAS_PATH)
    
    print("\n" + "=" * 30)
    print("AI GUIDE AVATAR - MUSEUM CAKRANINGRAT")
    print("=" * 30)
    print(f"Jumlah artefak : {len(artifacts)}")
    print(f"Jumlah persona : {len(personas)}")

    persona_key = choose_persona(personas)
    persona = personas[persona_key]
    
    print("\n" + "=" * 30)
    print(f"AI GUIDE: {persona['name'].upper()}")
    print("=" * 30)
    print("\n" + persona_greeting(persona))

    while True:
        artifact = choose_artifact(artifacts)
        if not find_artifact(artifacts, artifact["artifact_id"]):
            print("Artefak tidak ditemukan.")
            continue

        context = ConversationContext(
            museum_id=artifact.get("museum_id", "MUSEUM_CAKRANINGRAT"),
            artifact_id=artifact["artifact_id"],
            persona=persona_key,
            gender="female",
        )

        print(artifact_greeting(persona, artifact))
        conversation_result = conversation_loop(artifacts, persona, context)

        while True:
            choice = show_navigation()
            if choice == "1":
                print("\nBaik, kita kembali ke daftar artefak.")
                break
            elif choice == "2":
                new_persona_key = choose_persona(personas)
                persona_key = new_persona_key
                persona = personas[persona_key]
                
                context = ConversationContext(
                    museum_id=context.museum_id,
                    artifact_id=context.artifact_id,
                    persona=persona_key,
                    gender=context.gender,
                )
                
                print("\n" + "=" * 30)
                print("PERSONA BERHASIL DIGANTI")
                print("=" * 30)
                print(artifact_greeting(persona, artifact))
                
                cconversation_result = conversation_loop(artifacts, persona, context)
                continue
            elif choice == "3":
                print(closing_message())
                return

if __name__ == "__main__":
    main()