class SemanticMemory:
    """
    The agent's world model — generalizations, beliefs, and facts derived from reflection.
    Stored as plain strings; updated by the reflection engine.
    """

    def __init__(self, max_beliefs: int = 50):
        self._beliefs: list[str] = []
        self.max_beliefs = max_beliefs

    def add_belief(self, belief: str) -> None:
        belief = belief.strip()
        if not belief:
            return
        normalized = belief.lower()
        # Avoid near-duplicates (simple substring check)
        for existing in self._beliefs:
            if normalized in existing.lower() or existing.lower() in normalized:
                return
        if len(self._beliefs) >= self.max_beliefs:
            self._beliefs.pop(0)  # drop oldest
        self._beliefs.append(belief)

    def remove_belief(self, belief: str) -> None:
        self._beliefs = [b for b in self._beliefs if b.lower() != belief.lower()]

    def get_all(self) -> list[str]:
        return list(self._beliefs)

    def __len__(self) -> int:
        return len(self._beliefs)

    def format_for_prompt(self) -> str:
        if not self._beliefs:
            return "(no established beliefs)"
        lines = [f"- {b}" for b in self._beliefs]
        return "\n".join(lines)
