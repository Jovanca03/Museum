
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

from rag.chunking import load_and_chunk_data
from rag.embeddings import BGE_M3_Embeddings


COLLECTION_NAME = "museum_cakraningrat_bge_m3"

DB_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "chroma_db"
)


def index_documents(json_path: Path):
    print(f"\nMemproses: {json_path}")

    chunks = load_and_chunk_data(json_path)

    if not chunks:
        print("Tidak ada chunk. File dilewati.")
        return

    embedding_function = BGE_M3_Embeddings()

    documents = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"],
            )
        )

        # Gunakan ID asli jika tersedia.
        ids.append(str(chunk.get("id", f"{json_path.stem}_{i}")))

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=embedding_function,
    )

    # Memproses dalam batch agar penggunaan memori lebih terkontrol.
    batch_size = 16

    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start:start + batch_size]
        batch_ids = ids[start:start + batch_size]

        vector_store.add_documents(
            documents=batch_docs,
            ids=batch_ids,
        )

    print(f"Jumlah chunk: {len(documents)}")
    print(f"Collection: {COLLECTION_NAME}")
    print("Indeksasi selesai.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent

    data_files = [
        base_dir / "data" / "cakra_data.json",
    ]

    for data_path in data_files:
        if data_path.exists():
            index_documents(data_path)
        else:
            print(f"File tidak ditemukan: {data_path}")
