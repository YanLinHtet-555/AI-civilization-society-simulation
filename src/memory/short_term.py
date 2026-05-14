from collections import deque
from .models import MemoryEntry


class ShortTermMemory:
    """Rolling buffer of the most recent observations. Always available in context."""

    def __init__(self, capacity: int = 20):
        self._buffer: deque[MemoryEntry] = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, entry: MemoryEntry) -> None:
        self._buffer.append(entry)

    def get_all(self) -> list[MemoryEntry]:
        return list(self._buffer)

    def get_recent(self, n: int) -> list[MemoryEntry]:
        entries = list(self._buffer)
        return entries[-n:] if len(entries) >= n else entries

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def format_for_prompt(self) -> str:
        if not self._buffer:
            return "(no recent memories)"
        lines = [f"- {entry}" for entry in self._buffer]
        return "\n".join(lines)
