# AI Civilization Society Simulation

## Project Vision

A living digital society where autonomous AI agents evolve socially, economically, and politically over time. The objective is **emergent behavior** — not scripted outcomes.

Target emergent phenomena:
- Alliances forming naturally through repeated cooperation
- Trade networks crystallizing around resource scarcity
- Religious movements spreading via social influence
- Rebellions triggered by sustained inequality
- Social classes developing from accumulated advantage
- Agents lying, manipulating, and defecting when incentivized
- Governments stabilizing or collapsing based on legitimacy

Primary research reference: [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)

---

## Architecture

```
World Engine
    ↓
Event System
    ↓
Agent Cognition Layer
    ↓
Memory System          ← most critical component
    ↓
Social Graph
    ↓
Economy + Resource System
    ↓
LLM Decision Layer
```

---

## Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.11+ | asyncio support, AI ecosystem |
| Simulation loop | asyncio | Non-blocking concurrent agents |
| LLM | Anthropic Claude API (`claude-sonnet-4-6`) | Best reasoning/cost balance |
| Vector memory | ChromaDB | Local, no infra required |
| Structured state | SQLite via `aiosqlite` | Cheap, zero-ops, good for relationships/economy |
| API layer | FastAPI | Inspection and control endpoints |
| Config | Pydantic v2 + `.env` | Type-safe settings |

**Do NOT use LLMs for:** pathfinding, economy math, relationship score arithmetic, event resolution math. LLMs are expensive and slow — reserve them for reflection, conversation, strategic planning, and social interpretation.

---

## Project Structure

```
AI-civilization-society-simulation/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── main.py                          # simulation entry point
│
├── src/
│   ├── agents/
│   │   ├── agent.py                 # Agent dataclass + identity
│   │   ├── cognition.py             # perceive → goal → action pipeline
│   │   └── personality.py          # Big-5 style trait model
│   │
│   ├── memory/
│   │   ├── memory_system.py         # Three-layer memory facade
│   │   ├── short_term.py            # Recent events buffer (last N ticks)
│   │   ├── long_term.py             # ChromaDB-backed persistent store
│   │   ├── semantic.py              # World knowledge / beliefs
│   │   └── reflection.py           # Reflection + importance scoring (LLM)
│   │
│   ├── world/
│   │   ├── world_engine.py          # World state container
│   │   ├── event_engine.py          # Random + triggered event generation
│   │   └── map.py                   # Spatial grid (optional in MVP)
│   │
│   ├── social/
│   │   ├── social_graph.py          # Relationship weights between agents
│   │   └── gossip.py               # Belief propagation via conversation
│   │
│   ├── economy/
│   │   ├── resources.py             # Resource types, scarcity, production
│   │   ├── trade.py                 # Negotiation + exchange logic
│   │   └── market.py               # Price discovery (Phase 2)
│   │
│   ├── government/                  # Phase 3
│   │   ├── government.py
│   │   └── law.py
│   │
│   ├── religion/                    # Phase 3
│   │   └── ideology.py
│   │
│   ├── llm/
│   │   ├── client.py                # Anthropic SDK wrapper with caching
│   │   └── prompts.py              # Prompt templates
│   │
│   └── api/
│       └── server.py                # FastAPI inspection endpoints
│
├── data/
│   ├── agent_templates/             # Starter personality configs
│   └── world_configs/              # World presets (village, city, etc.)
│
└── tests/
    ├── test_memory.py
    ├── test_social_graph.py
    └── test_economy.py
```

---

## Core Systems

### 1. Agent Identity

```python
@dataclass
class Agent:
    id: str
    name: str
    age: int
    personality: Personality      # aggression, kindness, curiosity, ambition, honesty
    goals: list[str]              # current prioritized goals
    beliefs: list[str]            # semantic world model
    skills: dict[str, float]      # skill name → level 0.0–1.0
    needs: Needs                  # hunger, energy, safety, belonging
    relationships: dict[str, Relationship]  # agent_id → Relationship
    inventory: dict[str, float]   # resource → quantity
    memory: MemorySystem
```

Personality uses five traits (0.0–1.0):
- `aggression` — likelihood to use force or threats
- `kindness` — tendency toward cooperation and sharing
- `curiosity` — drives exploration and information-seeking
- `ambition` — pursuit of power and status
- `honesty` — resistance to deception

### 2. Memory System (Most Important)

Three layers, each with a distinct purpose:

**Short-term memory** — rolling buffer of the last 20 events/observations. Always in context. Cleared each day-cycle.

**Long-term memory** — ChromaDB vector store. Events are scored for importance (0–10) before storage. Only high-importance events are stored permanently. Retrieved via semantic similarity to current situation.

**Semantic memory** — the agent's world model: facts, beliefs, generalizations derived from reflection. Stored as embeddings. Updated by the reflection process.

**Reflection pipeline:**
```
Every N ticks:
  1. Pull recent short-term memories
  2. Ask LLM: "What are the 3 most important insights from these events?"
  3. Score each insight for importance
  4. Store high-importance insights in long-term + semantic stores
  5. Update agent beliefs list
```

This is what creates emergent worldviews:
```
repeated betrayals by neighbors
  → reflection: "people take advantage of weakness"
    → belief stored: "trust must be earned slowly"
      → future behavior: lower initial trust, higher defensiveness
```

**Importance scoring heuristics (no LLM needed):**
- Death of known agent: 10
- Physical violence involving self: 9
- Major resource gain/loss (>50% of inventory): 8
- Betrayal by trusted agent (trust > 0.7): 8
- New alliance formed: 7
- Significant conversation: 5
- Routine trade: 3
- Passing observation: 1–2

### 3. Social Relationship Graph

Each directed edge (A → B) stores:

```python
@dataclass
class Relationship:
    trust: float          # 0.0–1.0
    fear: float           # 0.0–1.0
    respect: float        # 0.0–1.0
    affection: float      # 0.0–1.0
    debt: float           # negative = A owes B, positive = B owes A
    history: list[str]    # last 10 significant interactions
```

Relationships decay toward neutral over time without interaction. Events shift scores:
- Shared meal: trust +0.05, affection +0.03
- Betrayal: trust −0.4, fear +0.1 (if aggressor is stronger)
- Defended in conflict: trust +0.15, respect +0.1
- Stolen from: trust −0.5, fear +0.2

Emergent structures:
- **Tribes**: clusters of high mutual trust
- **Hierarchies**: high-fear directed graphs pointing up
- **Alliances**: reciprocal high-trust + low-fear edges between tribes

### 4. Economy System

Resources: `food`, `wood`, `stone`, `coin`, `land` (Phase 2: `labor`, `knowledge`)

Each agent has production capacity based on skills and location. Resources are finite. Scarcity drives all economic behavior.

Agent economic decisions (rule-based, no LLM):
1. If need > threshold → attempt acquisition (trade, forage, steal)
2. If surplus > threshold → offer trade
3. Trade value = personal scarcity weight × quantity
4. Theft risk = (need urgency) × (1 − fear_of_target) × (1 − detection_probability)

### 5. Event Engine

World events fire on a probability schedule modified by world state:

| Event | Base probability/tick | Trigger conditions |
|---|---|---|
| Storm | 0.02 | — |
| Good harvest | 0.05 | spring season |
| Famine | 0.01 | food stock < 20% global |
| Disease outbreak | 0.01 | population density high |
| Resource discovery | 0.03 | — |
| Crime | dynamic | inequality index high |

Events are broadcast to all agents within range. Each agent decides independently how to respond based on personality, goals, and relationships.

### 6. Simulation Loop

```python
async def simulation_tick(world: World):
    events = world.event_engine.generate_events()

    async with asyncio.TaskGroup() as tg:
        for agent in world.agents:
            tg.create_task(agent_tick(agent, world, events))

    world.economy.update()
    world.social_graph.decay()
    world.tick += 1

async def agent_tick(agent: Agent, world: World, events: list[Event]):
    observations = agent.perceive(world, events)
    memories = agent.memory.retrieve(observations)
    agent.update_emotions(observations, memories)
    goal = agent.choose_goal()
    action = await agent.decide_action(goal, observations, memories)
    await agent.execute_action(action, world)
    await agent.reflect_and_store(observations)
```

LLM is called only in `decide_action` (for complex decisions) and `reflect_and_store` (for reflection). Routine decisions use rule-based logic.

---

## LLM Usage Policy

| Decision type | Method |
|---|---|
| Reflection / insight generation | LLM (claude-sonnet-4-6) |
| Agent conversation | LLM |
| Strategic planning (goals) | LLM, called infrequently |
| Theory of mind ("what does X want?") | LLM |
| Trade negotiation math | Rule-based |
| Emotion updates | Rule-based |
| Pathfinding | Rule-based |
| Importance scoring | Heuristic |
| Relationship score updates | Rule-based |

All LLM calls use prompt caching (`cache_control: ephemeral`) on shared system context (world state, agent roster) to minimize cost.

---

## Build Phases

### Phase 1 — MVP Village (current target)
- [ ] Agent identity + personality model
- [ ] Three-layer memory system
- [ ] Basic needs (hunger, energy)
- [ ] Simulation loop (sync first, async later)
- [ ] Simple conversation + gossip
- [ ] Social graph with trust/fear
- [ ] Console output / event log

Goal: 10–20 agents in a village, running for 100 ticks, producing a readable history of what happened and why.

### Phase 2 — Economy + Factions
- [ ] Full resource system with scarcity
- [ ] Trade negotiation
- [ ] Theft and conflict resolution
- [ ] Faction/tribe detection from social graph
- [ ] Map / spatial relationships
- [ ] FastAPI inspection layer

### Phase 3 — Governance + Ideology
- [ ] Government formation (triggered by faction size)
- [ ] Dynamic law creation
- [ ] Religious/ideological movements
- [ ] Propaganda and influence
- [ ] Elections, coups, rebellions

### Phase 4 — Civilization Scale
- [ ] Generational inheritance (beliefs, wealth, culture)
- [ ] Language evolution (agent slang/dialects)
- [ ] Historical archive (agents reference past events)
- [ ] Multiple villages / cities interacting

---

## Key Design Constraints

1. **Emergent over scripted** — never hard-code outcomes. Let systems interact.
2. **LLM is expensive** — every LLM call should produce durable value (stored memory, changed belief, logged decision). No throwaway calls.
3. **Observability first** — every significant agent decision must be logged with reasoning so the simulation is interpretable.
4. **Determinism toggle** — support a fixed random seed for reproducible runs during development.
5. **Fail gracefully** — LLM failures fall back to rule-based defaults. Simulation never halts on API errors.

---

## Environment Variables

```
ANTHROPIC_API_KEY=
SIMULATION_SEED=42
TICK_DELAY_MS=100
MAX_AGENTS=20
LOG_LEVEL=INFO
CHROMA_PERSIST_DIR=./data/chroma
```
