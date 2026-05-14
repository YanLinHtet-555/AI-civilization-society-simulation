"""
Tests for the three-layer memory system.
Run with:  pytest tests/test_memory.py -v
LLM calls are mocked — no API key required.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.memory.models import EventType, MemoryEntry
from src.memory.short_term import ShortTermMemory
from src.memory.semantic import SemanticMemory
from src.memory.importance import score_importance, score_emotional_valence
from src.memory.memory_system import MemorySystem


# ── Short-term memory ────────────────────────────────────────────────────────

class TestShortTermMemory:
    def _entry(self, tick: int, content: str) -> MemoryEntry:
        return MemoryEntry(tick=tick, content=content, event_type=EventType.OBSERVATION)

    def test_capacity_respected(self):
        mem = ShortTermMemory(capacity=5)
        for i in range(8):
            mem.add(self._entry(i, f"event {i}"))
        assert len(mem) == 5

    def test_oldest_dropped_when_full(self):
        mem = ShortTermMemory(capacity=3)
        for i in range(5):
            mem.add(self._entry(i, f"event {i}"))
        contents = [e.content for e in mem.get_all()]
        assert "event 0" not in contents
        assert "event 4" in contents

    def test_get_recent_returns_last_n(self):
        mem = ShortTermMemory(capacity=10)
        for i in range(7):
            mem.add(self._entry(i, f"event {i}"))
        recent = mem.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].content == "event 6"

    def test_clear(self):
        mem = ShortTermMemory(capacity=5)
        mem.add(self._entry(0, "something"))
        mem.clear()
        assert len(mem) == 0

    def test_format_for_prompt_empty(self):
        mem = ShortTermMemory()
        assert "no recent" in mem.format_for_prompt()

    def test_format_for_prompt_lists_entries(self):
        mem = ShortTermMemory()
        mem.add(self._entry(1, "John stole bread"))
        output = mem.format_for_prompt()
        assert "John stole bread" in output


# ── Importance scoring ───────────────────────────────────────────────────────

class TestImportanceScoring:
    def test_death_scores_maximum(self):
        score = score_importance(EventType.DEATH, "elder died in the village")
        assert score == 10.0

    def test_observation_scores_low(self):
        score = score_importance(EventType.OBSERVATION, "a bird flew past")
        assert score < 4.0

    def test_betrayal_scores_high(self):
        score = score_importance(EventType.BETRAYAL, "my ally betrayed me")
        assert score >= 8.0

    def test_keyword_bonus_applied(self):
        base = score_importance(EventType.OBSERVATION, "a bird flew past")
        with_kw = score_importance(EventType.OBSERVATION, "someone was murdered nearby")
        assert with_kw > base

    def test_not_involves_self_reduces_score(self):
        self_score = score_importance(EventType.VIOLENCE_WITNESSED, "fight nearby", involves_self=True)
        other_score = score_importance(EventType.VIOLENCE_WITNESSED, "fight nearby", involves_self=False)
        assert other_score < self_score

    def test_score_never_exceeds_10(self):
        score = score_importance(EventType.DEATH, "kill murder dead betray fight exile banish hate")
        assert score <= 10.0


class TestEmotionalValence:
    def test_death_negative(self):
        v = score_emotional_valence(EventType.DEATH, "the elder died")
        assert v < 0

    def test_alliance_positive(self):
        v = score_emotional_valence(EventType.ALLIANCE_FORMED, "we became friends and allies")
        assert v > 0

    def test_valence_bounded(self):
        v = score_emotional_valence(EventType.OBSERVATION, "neutral event")
        assert -1.0 <= v <= 1.0


# ── Semantic memory ──────────────────────────────────────────────────────────

class TestSemanticMemory:
    def test_add_and_retrieve(self):
        mem = SemanticMemory()
        mem.add_belief("I trust no one easily")
        assert "I trust no one easily" in mem.get_all()

    def test_dedup_prevents_near_duplicates(self):
        mem = SemanticMemory()
        mem.add_belief("I trust no one easily")
        mem.add_belief("I trust no one easily")
        assert len(mem) == 1

    def test_max_beliefs_respected(self):
        mem = SemanticMemory(max_beliefs=3)
        for i in range(5):
            mem.add_belief(f"unique belief number {i} about the world")
        assert len(mem) <= 3

    def test_remove_belief(self):
        mem = SemanticMemory()
        mem.add_belief("Power comes from fear")
        mem.remove_belief("Power comes from fear")
        assert "Power comes from fear" not in mem.get_all()

    def test_format_empty(self):
        mem = SemanticMemory()
        assert "no established" in mem.format_for_prompt()


# ── MemorySystem facade ──────────────────────────────────────────────────────

class _FakeLongTermMemory:
    """Stands in for LongTermMemory so tests never touch ChromaDB."""

    MIN_IMPORTANCE = 5.0

    def __init__(self):
        self._store: list[MemoryEntry] = []

    def store(self, entry: MemoryEntry) -> None:
        if entry.importance >= self.MIN_IMPORTANCE:
            self._store.append(entry)

    def retrieve(self, query: str, n: int = 5) -> list[MemoryEntry]:
        return self._store[:n]

    def count(self) -> int:
        return len(self._store)

    def format_for_prompt(self, query: str, n: int = 5) -> str:
        entries = self.retrieve(query, n)
        if not entries:
            return "(no relevant long-term memories)"
        return "\n".join(f"- {e}" for e in entries)


class TestMemorySystem:
    def _make_system(self) -> MemorySystem:
        mem = MemorySystem.__new__(MemorySystem)
        mem.agent_id = "test-agent-01"
        mem.agent_name = "Aren"
        mem.short_term = ShortTermMemory(capacity=10)
        mem.long_term = _FakeLongTermMemory()  # type: ignore[assignment]
        mem.semantic = SemanticMemory()
        from src.memory.reflection import ReflectionEngine
        mem._reflection_engine = ReflectionEngine(interval_ticks=5, insights_per_reflection=3)
        mem._last_reflection_tick = 0
        return mem

    def test_observe_stores_in_short_term(self):
        mem = self._make_system()
        mem.observe(tick=1, content="I saw smoke rising", event_type=EventType.OBSERVATION)
        assert len(mem.short_term) == 1

    def test_high_importance_stored_in_long_term(self):
        mem = self._make_system()
        mem.observe(
            tick=1,
            content="John was murdered by the river",
            event_type=EventType.DEATH,
        )
        assert mem.long_term.count() == 1

    def test_low_importance_not_in_long_term(self):
        mem = self._make_system()
        mem.observe(
            tick=1,
            content="a leaf fell",
            event_type=EventType.OBSERVATION,
            importance=1.0,
        )
        assert mem.long_term.count() == 0

    def test_summary_returns_counts(self):
        mem = self._make_system()
        mem.observe(tick=1, content="John was murdered", event_type=EventType.DEATH)
        s = mem.summary()
        assert s["short_term_count"] == 1
        assert s["long_term_count"] == 1

    def test_get_context_string_includes_sections(self):
        mem = self._make_system()
        mem.observe(tick=1, content="John stole my food", event_type=EventType.BETRAYAL)
        ctx = mem.get_context_string(query="food theft")
        assert "Aren" in ctx
        assert "John stole my food" in ctx

    @pytest.mark.asyncio
    async def test_maybe_reflect_calls_llm_at_interval(self):
        mem = self._make_system()
        for i in range(6):
            mem.observe(tick=i, content=f"event {i}", event_type=EventType.OBSERVATION)

        mock_llm = MagicMock()
        mock_llm.ask = AsyncMock(return_value="I learned that events repeat.\nI understand danger is near.\nI now believe nothing is safe.")

        insights = await mem.maybe_reflect(mock_llm, current_tick=5)
        assert len(insights) > 0
        mock_llm.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_maybe_reflect_skips_before_interval(self):
        mem = self._make_system()
        mem.observe(tick=1, content="something happened", event_type=EventType.OBSERVATION)

        mock_llm = MagicMock()
        mock_llm.ask = AsyncMock(return_value="I learned something.")

        insights = await mem.maybe_reflect(mock_llm, current_tick=3)
        assert insights == []
        mock_llm.ask.assert_not_called()

    @pytest.mark.asyncio
    async def test_reflection_insights_stored_in_semantic(self):
        mem = self._make_system()
        for i in range(6):
            mem.observe(tick=i, content=f"event {i}", event_type=EventType.OBSERVATION)

        mock_llm = MagicMock()
        mock_llm.ask = AsyncMock(
            return_value="I now believe that trust must be earned slowly.\nI learned that power favors the bold."
        )

        await mem.maybe_reflect(mock_llm, current_tick=5)
        beliefs = mem.semantic.get_all()
        assert any("trust" in b.lower() for b in beliefs)
