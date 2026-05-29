from __future__ import annotations

import json

from orchid_ai.core.memory_types import (
    OrchidConversationSummary,
    OrchidSummaryEntity,
    _deduplicate_list,
    _merge_entities,
)


class TestOrchidSummaryEntity:
    def test_default_type(self):
        e = OrchidSummaryEntity(name="test")
        assert e.type == "other"
        assert e.details == ""

    def test_full_entity(self):
        e = OrchidSummaryEntity(name="LeBron", type="person", details="NBA player")
        assert e.name == "LeBron"
        assert e.type == "person"
        assert e.details == "NBA player"


class TestOrchidConversationSummary:
    def test_defaults(self):
        s = OrchidConversationSummary()
        assert s.topics == []
        assert s.entities == []
        assert s.actions_taken == []
        assert s.decisions == []
        assert s.open_questions == []
        assert s.user_preferences == []
        assert s.narrative == ""
        assert s.covered_turns == 0

    def test_to_context_string_empty(self):
        s = OrchidConversationSummary()
        assert s.to_context_string() == ""

    def test_to_context_string_with_topics(self):
        s = OrchidConversationSummary(topics=["basketball", "NBA"])
        result = s.to_context_string()
        assert "Topics: basketball, NBA" in result

    def test_to_context_string_with_entities(self):
        s = OrchidConversationSummary(
            entities=[OrchidSummaryEntity(name="LeBron", type="person", details="NBA player")]
        )
        result = s.to_context_string()
        assert "Entities" in result
        assert "LeBron" in result
        assert "person" in result
        assert "NBA player" in result

    def test_to_context_string_with_actions(self):
        s = OrchidConversationSummary(actions_taken=["searched database", "sent notification"])
        result = s.to_context_string()
        assert "Actions taken" in result
        assert "searched database" in result

    def test_to_context_string_with_decisions(self):
        s = OrchidConversationSummary(decisions=["approved request"])
        result = s.to_context_string()
        assert "Decisions" in result
        assert "approved request" in result

    def test_to_context_string_with_questions(self):
        s = OrchidConversationSummary(open_questions=["when is deadline?"])
        result = s.to_context_string()
        assert "Open questions" in result
        assert "when is deadline?" in result

    def test_to_context_string_with_preferences(self):
        s = OrchidConversationSummary(user_preferences=["likes short answers"])
        result = s.to_context_string()
        assert "User preferences" in result
        assert "likes short answers" in result

    def test_to_context_string_with_narrative(self):
        s = OrchidConversationSummary(narrative="User asked about basketball.")
        result = s.to_context_string()
        assert "Summary: User asked about basketball." in result

    def test_to_context_string_all_fields(self):
        s = OrchidConversationSummary(
            topics=["sports"],
            entities=[OrchidSummaryEntity(name="LeBron", type="person", details="NBA")],
            actions_taken=["search"],
            decisions=["approved"],
            open_questions=["next?"],
            user_preferences=["verbose"],
            narrative="Full summary.",
        )
        result = s.to_context_string()
        assert "Topics" in result
        assert "Entities" in result
        assert "Actions taken" in result
        assert "Decisions" in result
        assert "Open questions" in result
        assert "User preferences" in result
        assert "Summary" in result

    def test_to_dict(self):
        s = OrchidConversationSummary(
            topics=["sports"],
            entities=[OrchidSummaryEntity(name="LeBron", type="person")],
            covered_turns=3,
        )
        d = s.to_dict()
        assert d["topics"] == ["sports"]
        assert d["entities"] == [{"name": "LeBron", "type": "person", "details": ""}]
        assert d["covered_turns"] == 3
        assert d["narrative"] == ""

    def test_from_dict(self):
        data = {
            "topics": ["sports"],
            "entities": [{"name": "LeBron", "type": "person", "details": "NBA"}],
            "actions_taken": ["search"],
            "decisions": [],
            "open_questions": [],
            "user_preferences": [],
            "narrative": "Summary text",
            "covered_turns": 5,
        }
        s = OrchidConversationSummary.from_dict(data)
        assert s.topics == ["sports"]
        assert len(s.entities) == 1
        assert s.entities[0].name == "LeBron"
        assert s.entities[0].details == "NBA"
        assert s.actions_taken == ["search"]
        assert s.narrative == "Summary text"
        assert s.covered_turns == 5

    def test_from_dict_empty(self):
        s = OrchidConversationSummary.from_dict({})
        assert s.topics == []
        assert s.entities == []
        assert s.narrative == ""
        assert s.covered_turns == 0

    def test_to_dict_from_dict_round_trip(self):
        s = OrchidConversationSummary(
            topics=["sports", "NBA"],
            entities=[OrchidSummaryEntity(name="Kobe", type="person", details="Lakers")],
            actions_taken=["scored"],
            decisions=["mvp"],
            open_questions=["retire?"],
            user_preferences=["highlights"],
            narrative="Great game",
            covered_turns=2,
        )
        d = s.to_dict()
        s2 = OrchidConversationSummary.from_dict(d)
        assert s2.topics == s.topics
        assert s2.entities[0].name == s.entities[0].name
        assert s2.narrative == s.narrative
        assert s2.covered_turns == s.covered_turns

    def test_from_json_valid(self):
        data = {
            "topics": ["sports"],
            "entities": [{"name": "LeBron", "type": "person", "details": "NBA"}],
            "actions_taken": [],
            "decisions": [],
            "open_questions": [],
            "user_preferences": [],
            "narrative": "test",
            "covered_turns": 1,
        }
        s = OrchidConversationSummary.from_json(json.dumps(data))
        assert s is not None
        assert s.topics == ["sports"]
        assert s.narrative == "test"

    def test_from_json_invalid(self):
        s = OrchidConversationSummary.from_json("not valid json")
        assert s is None

    def test_from_json_not_dict(self):
        s = OrchidConversationSummary.from_json(json.dumps(["list", "not", "dict"]))
        assert s is None

    def test_from_json_type_error(self):
        s = OrchidConversationSummary.from_json(None)  # type: ignore
        assert s is None

    def test_from_string_or_json_structured(self):
        data = {
            "topics": ["sports"],
            "entities": [],
            "actions_taken": [],
            "decisions": [],
            "open_questions": [],
            "user_preferences": [],
            "narrative": "structured",
            "covered_turns": 1,
        }
        s = OrchidConversationSummary.from_string_or_json(json.dumps(data))
        assert s.topics == ["sports"]

    def test_from_string_or_json_narrative(self):
        s = OrchidConversationSummary.from_string_or_json("plain narrative text")
        assert s.narrative == "plain narrative text"
        assert s.topics == []

    def test_from_string_or_json_empty(self):
        s = OrchidConversationSummary.from_string_or_json("")
        assert s.narrative == ""
        assert s.topics == []

    def test_merge(self):
        existing = OrchidConversationSummary(
            topics=["sports"],
            entities=[OrchidSummaryEntity(name="LeBron", type="person")],
            actions_taken=["scored"],
            covered_turns=2,
        )
        new_data = {
            "topics": ["NBA", "sports"],
            "entities": [{"name": "Kobe", "type": "person", "details": "Lakers"}],
            "actions_taken": ["assisted", "scored"],
            "decisions": ["winner"],
            "open_questions": [],
            "user_preferences": [],
            "narrative": "Updated summary",
            "covered_turns": 1,
        }
        merged = OrchidConversationSummary.merge(existing, new_data)
        assert "sports" in merged.topics
        assert "NBA" in merged.topics
        assert len(merged.topics) == 2
        assert merged.covered_turns == 3
        assert merged.narrative == "Updated summary"
        assert "scored" in merged.actions_taken
        assert "assisted" in merged.actions_taken
        assert len(merged.entities) == 2
        assert merged.decisions == ["winner"]

    def test_merge_entity_dedup(self):
        existing = OrchidConversationSummary(
            entities=[OrchidSummaryEntity(name="LeBron", type="person", details="NBA player")]
        )
        new_data = {
            "topics": [],
            "entities": [{"name": "LeBron", "type": "person", "details": "also known as King James"}],
            "actions_taken": [],
            "decisions": [],
            "open_questions": [],
            "user_preferences": [],
            "narrative": "",
            "covered_turns": 0,
        }
        merged = OrchidConversationSummary.merge(existing, new_data)
        assert len(merged.entities) == 1
        assert "NBA player" in merged.entities[0].details
        assert "King James" in merged.entities[0].details

    def test_merge_without_entities_key(self):
        existing = OrchidConversationSummary(narrative="old")
        new_data = {
            "narrative": "new",
            "covered_turns": 1,
        }
        merged = OrchidConversationSummary.merge(existing, new_data)
        assert merged.narrative == "new"


class TestDeduplicateList:
    def test_no_duplicates(self):
        result = _deduplicate_list(["a", "b"], ["c", "d"])
        assert result == ["a", "b", "c", "d"]

    def test_with_duplicates(self):
        result = _deduplicate_list(["a", "b"], ["b", "c"])
        assert result == ["a", "b", "c"]

    def test_all_duplicates(self):
        result = _deduplicate_list(["a", "b"], ["a", "b"])
        assert result == ["a", "b"]

    def test_empty_new(self):
        result = _deduplicate_list(["a", "b"], [])
        assert result == ["a", "b"]

    def test_empty_existing(self):
        result = _deduplicate_list([], ["a", "b"])
        assert result == ["a", "b"]


class TestMergeEntities:
    def test_empty_existing(self):
        result = _merge_entities([], [{"name": "LeBron", "type": "person"}])
        assert len(result) == 1
        assert result[0].name == "LeBron"

    def test_deduplicate_by_name(self):
        existing = [OrchidSummaryEntity(name="LeBron", type="person")]
        result = _merge_entities(existing, [{"name": "LeBron", "type": "person", "details": "NBA"}])
        assert len(result) == 1
        assert "NBA" in result[0].details

    def test_case_insensitive_dedup(self):
        existing = [OrchidSummaryEntity(name="LeBron", type="person")]
        result = _merge_entities(existing, [{"name": "lebron", "type": "person", "details": "NBA"}])
        assert len(result) == 1

    def test_missing_name_in_new(self):
        existing = [OrchidSummaryEntity(name="LeBron", type="person")]
        result = _merge_entities(existing, [{"type": "person"}])
        assert len(result) == 2

    def test_existing_preserved(self):
        existing = [OrchidSummaryEntity(name="LeBron", type="person", details="original")]
        result = _merge_entities(existing, [])
        assert len(result) == 1
        assert result[0].details == "original"
