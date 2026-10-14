from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationContext:
    museum_id: str
    artifact_id: str
    persona: str
    gender: str = "female"
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        self.conversation_history.append({
            "question": question,
            "answer": answer,
        })

    def change_artifact(self, artifact_id: str) -> None:
        self.artifact_id = artifact_id

    def change_museum(self, museum_id: str, artifact_id: str) -> None:
        self.museum_id = museum_id
        self.artifact_id = artifact_id

    def summary(self) -> dict:
        return {
            "museum_id": self.museum_id,
            "artifact_id": self.artifact_id,
            "persona": self.persona,
            "gender": self.gender,
            "conversation_turns": len(self.conversation_history),
        }