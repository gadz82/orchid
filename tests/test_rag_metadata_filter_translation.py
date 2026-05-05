"""Tests for the metadata-filter mini-language → Qdrant translation."""

from __future__ import annotations

import pytest
from qdrant_client.models import DatetimeRange, FieldCondition, MatchAny, MatchValue, Range

from orchid_ai.rag.backends.qdrant import (
    build_metadata_filter_clauses,
    infer_payload_index_types,
)


class TestBuildMetadataFilterClauses:
    def test_scalar_exact_match(self):
        must, must_not = build_metadata_filter_clauses({"status": "published"})
        assert must_not == []
        assert len(must) == 1
        clause = must[0]
        assert isinstance(clause, FieldCondition)
        assert clause.key == "status"
        assert clause.match == MatchValue(value="published")

    def test_match_any_list(self):
        must, must_not = build_metadata_filter_clauses({"language": ["en", "fr"]})
        assert must_not == []
        assert len(must) == 1
        assert must[0].match == MatchAny(any=["en", "fr"])

    def test_range_with_all_bounds(self):
        must, must_not = build_metadata_filter_clauses({"view_count": {"gte": 100, "lte": 1000, "gt": 50, "lt": 9999}})
        assert must_not == []
        assert must[0].range == Range(gte=100, lte=1000, gt=50, lt=9999)

    def test_range_partial_bounds_numeric(self):
        must, _ = build_metadata_filter_clauses({"price": {"gte": 5.0}})
        assert must[0].range == Range(gte=5.0)

    def test_range_with_iso_datetime_uses_datetime_range(self):
        """ISO-8601 string bounds switch to ``DatetimeRange`` so the
        Qdrant client doesn't try to parse the strings as floats."""
        must, _ = build_metadata_filter_clauses({"published_at": {"gte": "2026-01-01"}})
        # ``FieldCondition.range`` accepts both ``Range`` (numeric) and
        # ``DatetimeRange`` (ISO-8601 strings); the translator picks the
        # right model based on the operand type.
        assert isinstance(must[0].range, DatetimeRange)
        assert must[0].range == DatetimeRange(gte="2026-01-01")

    def test_contains_operator(self):
        must, _ = build_metadata_filter_clauses({"tags": {"contains": "release-notes"}})
        assert must[0].match == MatchValue(value="release-notes")

    def test_not_operator_lands_in_must_not(self):
        must, must_not = build_metadata_filter_clauses({"deprecated": {"not": True}})
        assert must == []
        assert len(must_not) == 1
        assert must_not[0].match == MatchValue(value=True)

    def test_backend_namespaced_keys_skipped(self):
        must, must_not = build_metadata_filter_clauses(
            {
                "status": "published",
                "_qdrant": {"any-extra": "ignored-here"},
                "_opensearch": {"knn": "ignored-here"},
            }
        )
        assert len(must) == 1
        assert must[0].key == "status"
        assert must_not == []

    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown metadata filter operator"):
            build_metadata_filter_clauses({"foo": {"weird": "value"}})

    def test_combined_filters_translate_independently(self):
        must, must_not = build_metadata_filter_clauses(
            {
                "status": "published",
                "language": ["en"],
                "view_count": {"gte": 100},
                "deprecated": {"not": True},
            }
        )
        assert len(must) == 3
        assert len(must_not) == 1
        keys_in_must = {c.key for c in must}
        assert keys_in_must == {"status", "language", "view_count"}

    def test_empty_filters_empty_clauses(self):
        must, must_not = build_metadata_filter_clauses({})
        assert must == []
        assert must_not == []


class TestInferPayloadIndexTypes:
    def test_string_inferred_as_keyword(self):
        assert infer_payload_index_types({"status": "published"}) == {"status": "keyword"}

    def test_integer_inferred(self):
        assert infer_payload_index_types({"count": 42}) == {"count": "integer"}

    def test_float_inferred(self):
        assert infer_payload_index_types({"weight": 1.5}) == {"weight": "float"}

    def test_bool_inferred_before_int(self):
        # Python bool is a subclass of int; the inference must short-circuit.
        assert infer_payload_index_types({"active": True}) == {"active": "bool"}

    def test_list_inferred_from_first_non_none(self):
        assert infer_payload_index_types({"languages": [None, "en", "fr"]}) == {"languages": "keyword"}

    def test_range_operand_drives_inference(self):
        assert infer_payload_index_types({"view_count": {"gte": 100}}) == {"view_count": "integer"}

    def test_iso_date_inferred_as_datetime(self):
        assert infer_payload_index_types({"published_at": {"gte": "2026-01-01"}}) == {"published_at": "datetime"}

    def test_contains_operand_drives_inference(self):
        assert infer_payload_index_types({"tags": {"contains": "release"}}) == {"tags": "keyword"}

    def test_not_operand_drives_inference(self):
        assert infer_payload_index_types({"deprecated": {"not": True}}) == {"deprecated": "bool"}

    def test_backend_namespaced_keys_skipped(self):
        assert infer_payload_index_types({"_qdrant": {"x": 1}, "status": "x"}) == {"status": "keyword"}

    def test_empty_filters_empty_inference(self):
        assert infer_payload_index_types({}) == {}
