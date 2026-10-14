
"""Membaca katalog artefak JSON sambil mempertahankan metadata aslinya."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_catalog(path: str | Path) -> list[dict[str, Any]]:
    """Membaca daftar artefak dari beberapa format JSON yang didukung."""
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        artifacts = payload
        defaults: dict[str, Any] = {}

    elif isinstance(payload, dict) and isinstance(payload.get("artifacts"), list):
        artifacts = payload["artifacts"]
        defaults = {
            key: value for key, value in payload.items()
            if key != "artifacts"
        }

    elif isinstance(payload, dict) and isinstance(payload.get("selected_artifacts"), list):
        artifacts = payload["selected_artifacts"]
        defaults = {
            key: value for key, value in payload.items()
            if key != "selected_artifacts"
        }

    elif isinstance(payload, dict) and "artifact_id" in payload:
        artifacts = [payload]
        defaults = {}

    else:
        raise ValueError(
            "Format katalog tidak dikenali. Gunakan daftar artefak, "
            "objek dengan kunci 'artifacts' atau 'selected_artifacts', "
            "atau satu objek artefak."
        )

    normalized: list[dict[str, Any]] = []

    for position, item in enumerate(artifacts, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Artefak urutan {position} harus berupa objek JSON."
            )

        record = {**defaults, **item}

        artifact_id = str(record.get("artifact_id", "")).strip()
        if not artifact_id:
            raise ValueError(
                f"Artefak urutan {position} tidak memiliki artifact_id."
            )

        record["artifact_id"] = artifact_id
        record["museum_id"] = str(
            record.get("museum_id")
            or (
                "MUSEUM_CAKRANINGRAT"
                if record.get("museum") == "Museum Cakraningrat"
                else "MUSEUM_NASIONAL_INDONESIA"
                if record.get("museum") == "Museum Nasional Indonesia"
                else defaults.get("museum_id", "")
            )
        ).strip()

        record["source"] = record.get("source") or (
            ", ".join(record["sources"])
            if isinstance(record.get("sources"), list)
            else "Sumber belum dicatat."
        )

        normalized.append(record)

    return normalized


def find_artifact(
    artifacts: list[dict[str, Any]],
    artifact_id: str,
) -> dict[str, Any] | None:
    return next(
        (item for item in artifacts if item["artifact_id"] == artifact_id),
        None,
    )
