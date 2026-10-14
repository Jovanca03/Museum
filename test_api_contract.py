"""Offline contract test; mocks only Ollama/Chroma, not production retrieval logic."""
from fastapi.testclient import TestClient
import api_server


class FakeDoc:
    page_content = "Digunakan untuk pagelaran seni dan budaya di Pendopo Keraton Bangkalan serta untuk menyambut tamu-tamu penting."
    metadata = {
        "artifact_id": "CAKRA_ART013",
        "chunk_id": "CAKRA_ART013_sejarah_dan_nilai_budaya_fungsi_budaya",
        "source": "catatan koleksi museum",
    }
    id = metadata["chunk_id"]


class FakeStore:
    def similarity_search_with_score(self, query, k, filter):
        assert filter == {"artifact_id": "CAKRA_ART013"}
        assert "Gamelan Se Lajing" in query
        return [(FakeDoc(), 0.18)]


class FakeMessage:
    content = "Gamelan Se Lajing digunakan untuk pagelaran seni dan budaya di Pendopo Keraton Bangkalan serta untuk menyambut tamu-tamu penting."


class FakeChatResponse:
    message = FakeMessage()


class FakeOllamaClient:
    def chat(self, **kwargs):
        assert kwargs["model"] == "qwen3:1.7b"
        assert "KONTEN HASIL PENCARIAN SEMANTIK BGE-M3" in kwargs["messages"][1]["content"]
        return FakeChatResponse()


def test_chat_uses_semantic_retrieval_contract(monkeypatch):
    monkeypatch.setattr(api_server, "_get_vector_store", lambda: FakeStore())
    monkeypatch.setattr(api_server, "_get_ollama_client", lambda: FakeOllamaClient())
    client = TestClient(api_server.app)
    request = {
        "question": "Apa fungsi dari artefak ini?",
        "artifact_id": "CAKRA_ART013",
        "persona": "Budayawan",
        "gender": "female",
        "museum_id": "MUSEUM_CAKRANINGRAT",
        "conversation_history": [],
        "top_k": 5,
    }
    response = client.post("/api/chat", json=request)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["retrieval_mode"] == "chroma-semantic-bge-m3"
    assert len(payload["retrieved_chunks"]) == 1
    assert "Pendopo Keraton Bangkalan" in payload["answer"]


if __name__ == "__main__":
    test_chat_uses_semantic_retrieval_contract(None)
    print("PASS: API contract returns semantic BGE-M3 mode and grounded chunk response.")
