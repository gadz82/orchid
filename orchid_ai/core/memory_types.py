"""
Structured conversation summary models and prompt constants.

These dataclasses live in ``core/`` so the graph and agent layers
can use them without importing Pydantic.  They are serialised to
plain dicts for JSON storage in the chat persistence backend.

All fields have defaults so partial LLM output never causes a crash.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── JSON schema (shared with the LLM for structured extraction) ──

SUMMARY_JSON_SCHEMA = """{
  "topics": ["topic1", "topic2"],
  "entities": [
    {"name": "entity_name", "type": "person|product|concept|other", "details": "key information"}
  ],
  "actions_taken": ["action1", "action2"],
  "decisions": ["decision1"],
  "open_questions": ["question1"],
  "user_preferences": ["preference1"],
  "narrative": "A brief prose summary of the conversation flow (2-3 sentences)",
  "covered_turns": 5
}"""

# ── Structured prompt defaults ──────────────────────────────────

DEFAULT_STRUCTURED_SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer that produces structured summaries. "
    "Output ONLY valid JSON with this schema:\n"
    f"{SUMMARY_JSON_SCHEMA}\n\n"
    "Be factual and concise. Extract all entities, topics, and decisions mentioned."
)

DEFAULT_STRUCTURED_SUMMARY_USER_PROMPT = (
    "Summarise the following conversation excerpt in structured JSON format. "
    "Focus on: (1) the key topics discussed, (2) any entities or "
    "names mentioned, (3) actions taken or decisions made, (4) any "
    "outstanding questions. Be factual and concise.\n\n"
    "{transcript}"
)

DEFAULT_STRUCTURED_EXTENSION_SYSTEM_PROMPT = (
    "You are a conversation summarizer that produces structured summaries. "
    "You have an existing summary and new messages to incorporate. "
    "Update the summary to reflect new information, remove contradicted facts, "
    "and merge duplicate entities.\n\n"
    "Output ONLY valid JSON with this schema:\n"
    f"{SUMMARY_JSON_SCHEMA}"
)

DEFAULT_STRUCTURED_EXTENSION_USER_PROMPT = (
    "Given the existing summary below and the new conversation messages, "
    "produce an updated summary that incorporates all new information.\n\n"
    "Existing summary:\n{existing_summary}\n\n"
    "New messages:\n{new_messages}"
)

DEFAULT_NARRATIVE_FALLBACK_PROMPT = (
    "Summarise the following conversation excerpt in 2-4 sentences. "
    "Focus on: (1) the key topics discussed, (2) any entities or "
    "names mentioned, (3) actions taken or decisions made, (4) any "
    "outstanding questions. Be factual and concise.\n\n"
    "{transcript}"
)


# ── Structured summary models ───────────────────────────────────


@dataclass
class OrchidSummaryEntity:
    """A named entity extracted from the conversation."""

    name: str
    type: str = "other"
    details: str = ""


@dataclass
class OrchidConversationSummary:
    """Structured conversation summary with entity tracking.

    All fields have defaults so partial LLM output never crashes.
    Serialise via ``to_dict()`` / ``from_dict()`` for JSON storage.
    """

    topics: list[str] = field(default_factory=list)
    entities: list[OrchidSummaryEntity] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    user_preferences: list[str] = field(default_factory=list)
    narrative: str = ""
    covered_turns: int = 0

    def to_context_string(self) -> str:
        """Render as a human-readable context block for LLM injection."""
        parts: list[str] = []
        if self.topics:
            parts.append(f"Topics: {', '.join(self.topics)}")
        if self.entities:
            entity_lines = [f"  - {e.name} ({e.type}): {e.details}" for e in self.entities]
            parts.append("Entities:\n" + "\n".join(entity_lines))
        if self.actions_taken:
            parts.append(f"Actions taken: {'; '.join(self.actions_taken)}")
        if self.decisions:
            parts.append(f"Decisions: {'; '.join(self.decisions)}")
        if self.open_questions:
            parts.append(f"Open questions: {'; '.join(self.open_questions)}")
        if self.user_preferences:
            parts.append(f"User preferences: {'; '.join(self.user_preferences)}")
        if self.narrative:
            parts.append(f"Summary: {self.narrative}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return {
            "topics": list(self.topics),
            "entities": [{"name": e.name, "type": e.type, "details": e.details} for e in self.entities],
            "actions_taken": list(self.actions_taken),
            "decisions": list(self.decisions),
            "open_questions": list(self.open_questions),
            "user_preferences": list(self.user_preferences),
            "narrative": self.narrative,
            "covered_turns": self.covered_turns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OrchidConversationSummary:
        """Deserialize from a plain dict (e.g., from JSON storage)."""
        entities = [OrchidSummaryEntity(**e) for e in data.get("entities", [])]
        return cls(
            topics=data.get("topics", []),
            entities=entities,
            actions_taken=data.get("actions_taken", []),
            decisions=data.get("decisions", []),
            open_questions=data.get("open_questions", []),
            user_preferences=data.get("user_preferences", []),
            narrative=data.get("narrative", ""),
            covered_turns=data.get("covered_turns", 0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> OrchidConversationSummary | None:
        """Parse a structured summary from a JSON string.

        Returns ``None`` if the string is not valid JSON or does not
        contain the expected structure.
        """
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        return cls.from_dict(data)

    @classmethod
    def from_string_or_json(cls, text: str) -> OrchidConversationSummary:
        """Parse from either structured JSON or a flat narrative string.

        When ``text`` is valid JSON it is parsed as a structured
        summary.  Otherwise a minimal summary with only the narrative
        field set is returned.
        """
        structured = cls.from_json(text)
        if structured is not None:
            return structured
        return cls(narrative=text.strip())

    @classmethod
    def merge(cls, existing: OrchidConversationSummary, new_data: dict) -> OrchidConversationSummary:
        """Merge new structured data into an existing summary.

        Lists are appended (deduplicating where possible).  The
        narrative is overwritten.  ``covered_turns`` is incremented.
        """
        merged = OrchidConversationSummary(
            topics=_deduplicate_list(existing.topics, new_data.get("topics", [])),
            entities=_merge_entities(existing.entities, new_data.get("entities", [])),
            actions_taken=_deduplicate_list(existing.actions_taken, new_data.get("actions_taken", [])),
            decisions=_deduplicate_list(existing.decisions, new_data.get("decisions", [])),
            open_questions=_deduplicate_list(existing.open_questions, new_data.get("open_questions", [])),
            user_preferences=_deduplicate_list(existing.user_preferences, new_data.get("user_preferences", [])),
            narrative=new_data.get("narrative", existing.narrative),
            covered_turns=existing.covered_turns + new_data.get("covered_turns", 1),
        )
        return merged


def _deduplicate_list(existing: list[str], new: list[str]) -> list[str]:
    seen = set(existing)
    result = list(existing)
    for item in new:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _merge_entities(
    existing: list[OrchidSummaryEntity],
    new: list[dict],
) -> list[OrchidSummaryEntity]:
    seen: dict[str, OrchidSummaryEntity] = {}
    result: list[OrchidSummaryEntity] = []
    for e in existing:
        copy = OrchidSummaryEntity(name=e.name, type=e.type, details=e.details)
        seen[copy.name.lower()] = copy
        result.append(copy)
    for e_dict in new:
        name = (e_dict.get("name") or "").lower()
        if name in seen:
            existing_entity = seen[name]
            existing_entity.type = e_dict.get("type", existing_entity.type)
            new_details = e_dict.get("details", "")
            if new_details and new_details not in existing_entity.details:
                existing_entity.details += "; " + new_details
        else:
            entity = OrchidSummaryEntity(
                name=e_dict.get("name", ""),
                type=e_dict.get("type", "other"),
                details=e_dict.get("details", ""),
            )
            result.append(entity)
            seen[name] = entity
    return result
