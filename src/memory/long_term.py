import chromadb
from chromadb.config import Settings

from .models import MemoryEntry


class LongTermMemory:
    """ChromaDB-backed persistent store. Only stores memories above MIN_IMPORTANCE."""

    MIN_IMPORTANCE = 5.0

    def __init__(self, agent_id: str, persist_dir: str = "./data/chroma"):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=f"agent_{agent_id.replace('-', '_')}",
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, entry: MemoryEntry) -> None:
        if entry.importance < self.MIN_IMPORTANCE:
            return
        self._collection.upsert(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[entry.to_metadata()],
        )

    def retrieve(self, query: str, n: int = 5) -> list[MemoryEntry]:
        total = self._collection.count()
        if total == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n, total),
        )
        entries = []
        for doc, meta, id_ in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0],
        ):
            entries.append(MemoryEntry.from_chroma(id_, doc, meta))
        return entries

    def retrieve_by_participant(self, agent_name: str, n: int = 10) -> list[MemoryEntry]:
        total = self._collection.count()
        if total == 0:
            return []
        results = self._collection.get(
            where={"participants": {"$contains": agent_name}},  # type: ignore[dict-item]
            limit=n,
        )
        entries = []
        for doc, meta, id_ in zip(
            results["documents"],
            results["metadatas"],
            results["ids"],
        ):
            entries.append(MemoryEntry.from_chroma(id_, doc, meta))
        return entries

    def count(self) -> int:
        return self._collection.count()

    def format_for_prompt(self, query: str, n: int = 5) -> str:
        entries = self.retrieve(query, n)
        if not entries:
            return "(no relevant long-term memories)"
        lines = [f"- {entry}" for entry in entries]
        return "\n".join(lines)
