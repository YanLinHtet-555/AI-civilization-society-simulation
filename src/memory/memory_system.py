import logging
from typing import TYPE_CHECKING

from .models import EventType, MemoryEntry
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .semantic import SemanticMemory
from .reflection import ReflectionEngine
from .importance import score_importance, score_emotional_valence

if TYPE_CHECKING:
    from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class MemorySystem:
    """
    Facade over all three memory layers.

    Usage:
        memory = MemorySystem(agent_id="agent-1", agent_name="Aren")
        memory.observe(tick=5, content="John stole bread from the market", event_type=EventType.OBSERVATION)
        context = memory.get_context_string(query="food and trust")
        await memory.maybe_reflect(llm_client, current_tick=10)
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        short_term_capacity: int = 20,
        long_term_min_importance: float = 5.0,
        reflection_interval: int = 10,
        chroma_persist_dir: str = "./data/chroma",
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(agent_id=agent_id, persist_dir=chroma_persist_dir)
        self.long_term.MIN_IMPORTANCE = long_term_min_importance
        self.semantic = SemanticMemory()
        self._reflection_engine = ReflectionEngine(interval_ticks=reflection_interval)
        self._last_reflection_tick: int = 0

    def observe(
        self,
        tick: int,
        content: str,
        event_type: EventType,
        participants: list[str] | None = None,
        involves_self: bool = True,
        importance: float | None = None,
        emotional_valence: float | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """
        Record a new observation. Automatically scores importance and valence,
        stores in short-term always, and in long-term if importance >= threshold.
        """
        imp = importance if importance is not None else score_importance(event_type, content, involves_self)
        val = emotional_valence if emotional_valence is not None else score_emotional_valence(event_type, content)

        entry = MemoryEntry(
            tick=tick,
            content=content,
            event_type=event_type,
            participants=participants or [],
            importance=imp,
            emotional_valence=val,
            tags=tags or [],
        )

        self.short_term.add(entry)
        self.long_term.store(entry)

        logger.debug(
            "%s memory: imp=%.1f val=%+.1f [%s] %s",
            self.agent_name, imp, val, event_type.value, content,
        )
        return entry

    async def maybe_reflect(self, llm: "LLMClient", current_tick: int) -> list[MemoryEntry]:
        """Trigger reflection if interval has elapsed. Returns new insights."""
        if not self._reflection_engine.should_reflect(current_tick, self._last_reflection_tick):
            return []
        insights = await self._reflection_engine.reflect(
            agent_name=self.agent_name,
            memory=self,
            llm=llm,
            current_tick=current_tick,
        )
        self._last_reflection_tick = current_tick
        return insights

    def retrieve_relevant(self, query: str, n: int = 5) -> list[MemoryEntry]:
        """Retrieve semantically relevant long-term memories for a given situation."""
        return self.long_term.retrieve(query, n)

    def get_context_string(self, query: str = "", n_long_term: int = 5) -> str:
        """
        Build a formatted memory context block for injection into LLM prompts.
        Includes short-term buffer + semantically retrieved long-term memories.
        """
        sections = [
            f"=== {self.agent_name}'s Memory ===",
            "\n[Recent experiences]",
            self.short_term.format_for_prompt(),
        ]

        if self.long_term.count() > 0 and query:
            sections += [
                "\n[Relevant past experiences]",
                self.long_term.format_for_prompt(query, n_long_term),
            ]

        if self.semantic:
            sections += [
                "\n[Core beliefs]",
                self.semantic.format_for_prompt(),
            ]

        return "\n".join(sections)

    def clear_short_term(self) -> None:
        self.short_term.clear()

    def summary(self) -> dict:
        return {
            "agent": self.agent_name,
            "short_term_count": len(self.short_term),
            "long_term_count": self.long_term.count(),
            "belief_count": len(self.semantic),
            "last_reflection_tick": self._last_reflection_tick,
        }
