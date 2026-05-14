import logging
from typing import TYPE_CHECKING

from .models import EventType, MemoryEntry
from .importance import score_importance, score_emotional_valence

if TYPE_CHECKING:
    from ..llm.client import LLMClient
    from .memory_system import MemorySystem

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """
    Periodically reads short-term memories, calls the LLM to derive insights,
    and stores them as high-importance entries in long-term + semantic memory.

    Implements the pipeline from the Generative Agents paper:
      Event → Importance Score → Reflection → Belief Update
    """

    def __init__(self, interval_ticks: int = 10, insights_per_reflection: int = 3):
        self.interval_ticks = interval_ticks
        self.insights_per_reflection = insights_per_reflection

    def should_reflect(self, current_tick: int, last_reflection_tick: int) -> bool:
        return (current_tick - last_reflection_tick) >= self.interval_ticks

    async def reflect(
        self,
        agent_name: str,
        memory: "MemorySystem",
        llm: "LLMClient",
        current_tick: int,
    ) -> list[MemoryEntry]:
        """
        Run one reflection cycle. Returns the new insight MemoryEntries created.
        Falls back gracefully if LLM is unavailable.
        """
        from ..llm.prompts import REFLECTION_SYSTEM, REFLECTION_USER

        recent = memory.short_term.get_recent(self.interval_ticks)
        if not recent:
            return []

        system = REFLECTION_SYSTEM.format(
            agent_name=agent_name,
            current_beliefs=memory.semantic.format_for_prompt(),
        )
        user = REFLECTION_USER.format(
            agent_name=agent_name,
            recent_memories="\n".join(f"- {e}" for e in recent),
            n=self.insights_per_reflection,
        )

        raw = await llm.ask(system, user, max_tokens=256, cache_system=True)
        if not raw.strip():
            logger.warning("%s: reflection returned empty response", agent_name)
            return []

        insights = [line.strip() for line in raw.strip().splitlines() if line.strip()]

        new_entries: list[MemoryEntry] = []
        for insight in insights[: self.insights_per_reflection]:
            entry = MemoryEntry(
                tick=current_tick,
                content=insight,
                event_type=EventType.REFLECTION,
                importance=score_importance(EventType.REFLECTION, insight, involves_self=True),
                emotional_valence=score_emotional_valence(EventType.REFLECTION, insight),
            )
            memory.long_term.store(entry)
            memory.semantic.add_belief(insight)
            new_entries.append(entry)
            logger.debug("%s reflection insight: %s", agent_name, insight)

        return new_entries
