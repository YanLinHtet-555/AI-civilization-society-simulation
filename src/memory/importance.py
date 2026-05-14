from .models import EventType

_BASE_SCORES: dict[EventType, float] = {
    EventType.DEATH: 10.0,
    EventType.VIOLENCE_RECEIVED: 9.0,
    EventType.BETRAYAL: 8.0,
    EventType.MAJOR_RESOURCE_LOSS: 8.0,
    EventType.VIOLENCE_WITNESSED: 7.0,
    EventType.ALLIANCE_FORMED: 7.0,
    EventType.MAJOR_RESOURCE_GAIN: 7.0,
    EventType.REFLECTION: 6.0,
    EventType.CONVERSATION: 5.0,
    EventType.WORLD_EVENT: 4.0,
    EventType.TRADE: 3.0,
    EventType.OBSERVATION: 2.0,
}

# Each keyword adds to importance. Capped at +3.0 total.
_KEYWORD_WEIGHTS: dict[str, float] = {
    "kill": 3.0,
    "murder": 3.0,
    "dead": 2.0,
    "death": 2.0,
    "steal": 2.0,
    "stolen": 2.0,
    "betray": 2.0,
    "betrayed": 2.0,
    "attack": 2.0,
    "fight": 1.5,
    "ally": 1.5,
    "alliance": 1.5,
    "friend": 1.0,
    "trust": 1.0,
    "starve": 2.0,
    "hunger": 1.0,
    "exile": 2.0,
    "banish": 2.0,
    "love": 1.0,
    "hate": 1.5,
    "power": 1.0,
    "leader": 1.0,
}


def score_importance(
    event_type: EventType,
    content: str,
    involves_self: bool = True,
) -> float:
    base = _BASE_SCORES.get(event_type, 2.0)

    content_lower = content.lower()
    keyword_bonus = sum(w for kw, w in _KEYWORD_WEIGHTS.items() if kw in content_lower)
    keyword_bonus = min(keyword_bonus, 3.0)

    multiplier = 1.0 if involves_self else 0.7

    return min((base + keyword_bonus) * multiplier, 10.0)


def score_emotional_valence(event_type: EventType, content: str) -> float:
    """Returns -1.0 (very negative) to 1.0 (very positive)."""
    positive_kw = {"gift", "ally", "friend", "trust", "harvest", "found", "love", "help", "heal", "peace"}
    negative_kw = {"kill", "murder", "dead", "steal", "betray", "attack", "fight", "starve", "exile", "hate", "fear"}

    content_lower = content.lower()
    pos = sum(1 for kw in positive_kw if kw in content_lower)
    neg = sum(1 for kw in negative_kw if kw in content_lower)

    # Event-type baseline
    if event_type in (EventType.DEATH, EventType.VIOLENCE_RECEIVED, EventType.BETRAYAL, EventType.MAJOR_RESOURCE_LOSS):
        baseline = -0.6
    elif event_type in (EventType.ALLIANCE_FORMED, EventType.MAJOR_RESOURCE_GAIN):
        baseline = 0.5
    else:
        baseline = 0.0

    keyword_signal = (pos - neg) * 0.15
    return max(-1.0, min(1.0, baseline + keyword_signal))
