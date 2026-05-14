from dataclasses import dataclass, field
from enum import Enum
import uuid


class EventType(str, Enum):
    DEATH = "death"
    VIOLENCE_RECEIVED = "violence_received"
    VIOLENCE_WITNESSED = "violence_witnessed"
    MAJOR_RESOURCE_LOSS = "major_resource_loss"
    MAJOR_RESOURCE_GAIN = "major_resource_gain"
    BETRAYAL = "betrayal"
    ALLIANCE_FORMED = "alliance_formed"
    CONVERSATION = "conversation"
    TRADE = "trade"
    WORLD_EVENT = "world_event"
    OBSERVATION = "observation"
    REFLECTION = "reflection"


@dataclass
class MemoryEntry:
    tick: int
    content: str
    event_type: EventType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    participants: list[str] = field(default_factory=list)
    importance: float = 0.0
    emotional_valence: float = 0.0  # -1.0 (terrible) to 1.0 (wonderful)
    tags: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict:
        return {
            "tick": self.tick,
            "event_type": self.event_type.value,
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "participants": ",".join(self.participants),
            "tags": ",".join(self.tags),
        }

    @classmethod
    def from_chroma(cls, id: str, document: str, metadata: dict) -> "MemoryEntry":
        return cls(
            id=id,
            tick=metadata["tick"],
            content=document,
            event_type=EventType(metadata["event_type"]),
            importance=metadata["importance"],
            emotional_valence=metadata["emotional_valence"],
            participants=metadata["participants"].split(",") if metadata["participants"] else [],
            tags=metadata["tags"].split(",") if metadata["tags"] else [],
        )

    def __str__(self) -> str:
        sign = "+" if self.emotional_valence >= 0 else ""
        return f"[tick={self.tick} imp={self.importance:.1f} val={sign}{self.emotional_valence:.1f}] {self.content}"
