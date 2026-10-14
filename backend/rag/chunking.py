
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _flatten_to_text(value, prefix=""):
    """Mengubah data terstruktur menjadi teks yang dapat diindeks."""
    lines = []

    if isinstance(value, dict):
        for key, item in value.items():
            label = f"{prefix}{key.replace('_', ' ')}"
            lines.extend(_flatten_to_text(item, f"{label}: "))

    elif isinstance(value, list):
        for item in value:
            lines.extend(_flatten_to_text(item, prefix))

    elif value is not None and str(value).strip():
        lines.append(f"{prefix}{value}")

    return lines


def load_and_chunk_data(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Mendukung JSON lama berupa list dan JSON terstruktur baru.
    if isinstance(data, list):
        artifacts = data
    elif isinstance(data, dict):
        artifacts = data.get("selected_artifacts", [])
    else:
        raise ValueError(
            f"Format JSON tidak didukung: {json_path.name}"
        )

    if not isinstance(artifacts, list):
        raise ValueError(
            f"Data artefak harus berupa list: {json_path.name}"
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    all_chunks = []

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue

        # Filter URL hanya diterapkan pada format data lama.
        if "content" in artifact:
            if artifact.get("status") != "Hilang/Pencurian":
                source_url = artifact.get("source_url", "")
                if (
                    not source_url
                    or source_url == "https://www.museumcakraningrat.web.id/"
                ):
                    continue

        # Ubah seluruh data artefak menjadi teks.
        lines = _flatten_to_text(artifact)
        main_text = "\n".join(lines).strip()

        if len(main_text) < 10:
            continue

        chunks = text_splitter.split_text(main_text)
        artifact_id = artifact.get("artifact_id", "unknown")

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{json_path.stem}_{artifact_id}_chunk_{i}",
                "artifact_id": artifact_id,
                "text": chunk,
                "metadata": {
                    "name": artifact.get(
                        "name",
                        artifact.get("vr_display_name", "")
                    ),
                    "source": artifact.get("source", ""),
                    "artifact_id": artifact_id,
                    "source_file": json_path.name,
                },
            })

    print(
        f"{json_path.name}: {len(artifacts)} artefak dibaca, "
        f"{len(all_chunks)} chunk dibuat."
    )

    return all_chunks


def build_chunks(artifact: dict):
    """Membuat chunk untuk satu artefak terstruktur."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    main_text = "\n".join(_flatten_to_text(artifact)).strip()

    if len(main_text) < 10:
        return []

    artifact_id = artifact.get("artifact_id", "unknown")
    chunks = text_splitter.split_text(main_text)

    return [
        {
            "id": f"{artifact_id}_chunk_{i}",
            "artifact_id": artifact_id,
            "text": chunk,
            "metadata": {
                "name": artifact.get(
                    "name",
                    artifact.get("vr_display_name", "")
                ),
                "source": artifact.get("source", ""),
                "artifact_id": artifact_id,
            },
        }
        for i, chunk in enumerate(chunks)
    ]
    
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "cakra_data.json"

    if data_path.exists():
        chunks = load_and_chunk_data(data_path)

        print(f"Total chunks: {len(chunks)}")

        for chunk in chunks:
            if chunk["artifact_id"] == "CAKRA_ART011":
                print(chunk["id"])
                print(chunk["metadata"])
                print(chunk["text"])
                print("-" * 50)
    else:
        print(f"File tidak ditemukan: {data_path}")