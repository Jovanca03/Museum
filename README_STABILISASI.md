# Stabilisasi AI + RAG Museum Cakraningrat (sebelum Unity)

Paket ini mempertahankan satu jalur jawaban: **BGE-M3 → semantic retrieval ChromaDB dengan filter `artifact_id` → Qwen3:1.7B via Ollama**. Tidak ada `catalog-structured-direct`, keyword fallback, Gemini, atau API LLM eksternal. Fine-tuning tidak dilakukan pada tahap ini.

## Status penting

- Uji kontrak API dengan mock: `pytest -q` (memastikan kode mengirim pertanyaan ke semantic retrieval dan meneruskan chunk ke generator).
- Uji runtime nyata belum dapat dinyatakan lulus sampai dijalankan pada laptop proyek yang memiliki Ollama dan `data/chroma_db` yang benar. Database Chroma tidak termasuk dalam paket yang tersedia untuk pemeriksaan ini; jangan mengindeks ulang sebelum memastikan folder database yang dipakai sekarang.

## Prasyarat

- Ollama berjalan di `http://127.0.0.1:11434`
- Model `bge-m3` dan `qwen3:1.7b` sudah tersedia (`ollama list`)
- Database Chroma lama berada di `data/chroma_db/` dan nama collection `museum_cakraningrat_bge_m3`
- Gunakan virtual environment Python proyek yang sama dengan saat membuat indeks Chroma

## Pasang dan uji bertahap

1. Ekstrak isi paket ke folder proyek AI/RAG (bukan ke folder Unity). Pastikan `api_server.py`, `rag/`, dan `data/` berada sejajar.
2. Pertahankan database yang ada; jangan hapus `data/chroma_db`.
3. Pasang dependensi: `python -m pip install -r requirements.txt`.
4. Pastikan model lokal: `ollama list`.
5. Jalankan pemeriksaan semantic retrieval nyata: `python test_live_rag.py`. Ketiga kasus harus PASS dan menampilkan chunk yang relevan. Bila FAIL, periksa filter metadata dan isi chunk sebelum lanjut.
6. Jalankan `pytest -q` untuk tes kontrak kode.
7. Jalankan backend: `python -m uvicorn api_server:app --host 127.0.0.1 --port 8000`.
8. Buka `http://127.0.0.1:8000/health`. Semua komponen harus `ok`, `chroma_count > 0`, model embedding/chat tersedia, `ready: true`.
9. Uji `POST /api/chat` dari `http://127.0.0.1:8000/docs`, pakai `artifact_id=CAKRA_ART013` dan pertanyaan fungsi. Periksa `retrieval_mode=chroma-semantic-bge-m3`, `retrieved_chunks`, dan jawaban.

**Unity belum menjadi bagian dari tahap ini.** Integrasi Unity baru dimulai setelah tes live retrieval, tes generasi Qwen, dan pertanyaan di luar cakupan semuanya dievaluasi.
