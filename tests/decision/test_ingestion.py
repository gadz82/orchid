"""Tests for decision-backed document ingestion.

The ``ingestion`` point may select a registered parser before extraction and a
registered strategy after extraction. It may never invent a name; every
unknown, refused, low-confidence or failed answer falls back to the extension /
configured default. A disabled point makes zero provider calls and keeps the
legacy path. Tests use in-process providers and parser/strategy fakes — no
network and no real PDF/vision parsing.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel

import orchid_ai.documents.pipeline as pipeline_module
from orchid_ai.agents.rag_pipeline import RagPipeline
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
)
from orchid_ai.core.ingestion import OrchidChunk, OrchidIngestionStrategy
from orchid_ai.decision.registry import get_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.documents import parsers as parsers_module
from orchid_ai.documents.decision import (
    MAX_SNAPSHOT_CHARS,
    DocumentDescriptor,
    select_document_parser,
    select_ingestion_strategy,
)
from orchid_ai.documents.parsers import DocumentParser, get_parser_by_name, list_parser_names
from orchid_ai.documents.pipeline import extract_text, ingest_document
from orchid_ai.documents.strategies import INGESTION_REGISTRY
from orchid_ai.graph.decision_bindings import DecisionBinding, resolve_decision_binding
from orchid_ai.graph.graph import build_graph
from orchid_ai.rag.scopes import OrchidRAGScope
from orchid_ai.runtime import OrchidRuntime


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t", user_id="u", chat_id="c", agent_id="a")


def _binding(
    provider: Any,
    *,
    threshold: float = 0.75,
    options: dict[str, Any] | None = None,
    max_candidates: int | None = None,
) -> DecisionBinding:
    return DecisionBinding(
        spec=get_decision_point("ingestion"),
        config=OrchidDecisionPointConfig(
            model="stub",
            confidence_threshold=threshold,
            options=options or {},
            max_candidates=max_candidates,
        ),
        model=provider,
    )


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]


class _ChoiceProvider:
    """Return a fixed choice and record exactly what the gate sent."""

    def __init__(self, value: str, *, confidence: float = 0.95) -> None:
        self.value = value
        self.confidence = confidence
        self.calls = 0
        self.questions: list[OrchidDecisionQuestion] = []
        self.states: list[OrchidDecisionState] = []

    async def decide(self, state, questions, *, timeout=None):
        self.calls += 1
        self.states.append(state)
        self.questions.extend(questions)
        return OrchidDecisionResult(
            answers={
                questions[0].name: OrchidDecisionAnswer(
                    name=questions[0].name,
                    kind="choice",
                    value=self.value,
                    confidence=self.confidence,
                )
            },
            provider="recorder",
            model="recorder",
        )


class _QuestionRoutingProvider:
    """Answer parser and strategy questions with different fixed names."""

    def __init__(self, *, parser: str, strategy: str) -> None:
        self.parser = parser
        self.strategy = strategy
        self.questions: list[str] = []

    async def decide(self, state, questions, *, timeout=None):
        question = questions[0]
        self.questions.append(question.name)
        value = self.parser if question.name == "parser" else self.strategy
        return OrchidDecisionResult(
            answers={
                question.name: OrchidDecisionAnswer(
                    name=question.name,
                    kind="choice",
                    value=value,
                    confidence=0.95,
                )
            },
            provider="routing",
            model="routing",
        )


class _ErrorProvider:
    def __init__(self, code: str) -> None:
        self.code = code

    async def decide(self, state, questions, *, timeout=None):
        raise OrchidDecisionError("provider failed", code=self.code, provider="error-provider")


class _TimeoutProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise TimeoutError("deadline exceeded")


class _RaisingProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise RuntimeError("unexpected provider failure")


class _EmptyProvider:
    async def decide(self, state, questions, *, timeout=None):
        return OrchidDecisionResult(answers={}, provider="empty", model="empty")


class _RefusalProvider:
    async def decide(self, state, questions, *, timeout=None):
        question = questions[0]
        return OrchidDecisionResult(
            answers={
                question.name: OrchidDecisionAnswer(
                    name=question.name,
                    kind="choice",
                    value=None,
                    confidence=0.9,
                    refusal="not enough metadata",
                )
            },
            provider="refusal",
            model="refusal",
        )


class _FakeWriter:
    def __init__(self) -> None:
        self.batches: list[tuple[list[Any], str]] = []

    async def upsert(self, documents: list[Any], namespace: str) -> None:
        self.batches.append((documents, namespace))


class _MarkerStrategy(OrchidIngestionStrategy):
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.calls: list[dict[str, Any]] = []

    async def ingest(self, *, text, filename, scope, doc_store=None, embeddings=None):
        self.calls.append({"text": text, "filename": filename, "scope": scope})
        return [OrchidChunk(text=text, metadata={"marker": self.marker})]


@pytest.fixture
def parser_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the heavy PDF/image parsers with recording fakes.

    ``.txt`` is replaced too so the parse-once test can count actual parse
    calls. CSV keeps the real stdlib parser.
    """

    class FakePDFParser(DocumentParser):
        calls: ClassVar[list[tuple[bytes, str]]] = []

        async def parse(self, file_bytes: bytes, filename: str) -> str:
            FakePDFParser.calls.append((file_bytes, filename))
            return "pdf text"

    class FakeImageParser(DocumentParser):
        calls: ClassVar[list[tuple[bytes, str]]] = []

        async def parse(self, file_bytes: bytes, filename: str) -> str:
            FakeImageParser.calls.append((file_bytes, filename))
            return "image text"

    class FakeTextParser(DocumentParser):
        calls: ClassVar[list[tuple[bytes, str]]] = []

        async def parse(self, file_bytes: bytes, filename: str) -> str:
            FakeTextParser.calls.append((file_bytes, filename))
            return file_bytes.decode("utf-8", errors="replace")

    monkeypatch.setitem(parsers_module.PARSER_REGISTRY, ".pdf", FakePDFParser)
    for ext in (".png", ".jpg", ".jpeg"):
        monkeypatch.setitem(parsers_module.PARSER_REGISTRY, ext, FakeImageParser)
    for ext in (".md", ".txt"):
        monkeypatch.setitem(parsers_module.PARSER_REGISTRY, ext, FakeTextParser)
    return {
        "PDFParser": FakePDFParser,
        "ImageParser": FakeImageParser,
        "TextParser": FakeTextParser,
    }


@pytest.fixture
def selected_strategy(monkeypatch: pytest.MonkeyPatch) -> type[_MarkerStrategy]:
    class SelectedStrategy(_MarkerStrategy):
        instances: ClassVar[list[SelectedStrategy]] = []

        def __init__(self) -> None:
            super().__init__("selected")
            SelectedStrategy.instances.append(self)

    monkeypatch.setitem(INGESTION_REGISTRY, "selected", SelectedStrategy)
    return SelectedStrategy


# ── Descriptor contract ─────────────────────────────────────────


def test_descriptor_is_bounded_and_excludes_the_full_body() -> None:
    body = ("public line " * 200) + "TOP-SECRET-END"
    descriptor = DocumentDescriptor.from_bytes(
        file_bytes=b"x" * 4096,
        filename="/private/tmp/../Report\x00 Name.pdf",
        text=body,
    )

    assert descriptor.filename == "Report Name.pdf"
    assert descriptor.extension == ".pdf"
    assert descriptor.mime_type == "application/pdf"
    assert descriptor.size_bytes == 4096
    assert len(descriptor.text_snapshot) == MAX_SNAPSHOT_CHARS
    assert "TOP-SECRET-END" not in descriptor.text_snapshot
    assert "TOP-SECRET-END" not in repr(descriptor.to_payload())
    assert set(descriptor.to_payload()) == {"filename", "extension", "mime_type", "size_bytes"}


def test_descriptor_has_no_snapshot_before_extraction() -> None:
    descriptor = DocumentDescriptor.from_bytes(file_bytes=b"a,b\n1,2\n", filename="table.csv")
    assert descriptor.text_snapshot == ""
    assert descriptor.mime_type == "text/csv"
    assert descriptor.extension == ".csv"


def test_large_text_snapshot_scans_only_a_bounded_prefix() -> None:
    class RecordingText(str):
        slices: ClassVar[list[slice]] = []

        def __getitem__(self, key):
            if isinstance(key, slice):
                RecordingText.slices.append(key)
            return str.__getitem__(self, key)

    body = RecordingText("word " * 250_000)
    descriptor = DocumentDescriptor.from_bytes(file_bytes=b"x", filename="large.txt", text=body)

    assert len(descriptor.text_snapshot) == MAX_SNAPSHOT_CHARS
    assert len(RecordingText.slices) == 1
    assert RecordingText.slices[0].stop == MAX_SNAPSHOT_CHARS * 4


# ── Parser selection ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("filename", "body", "parser_name", "expected"),
    [
        ("notes.md", b"hello markdown", "FakeTextParser", "hello markdown"),
        ("table.csv", b"a,b\n1,2\n", "CSVParser", "a | b\n1 | 2"),
        ("report.pdf", b"%PDF-1.4 fake", "FakePDFParser", "pdf text"),
        ("scan.png", b"\x89PNG fake", "FakeImageParser", "image text"),
    ],
)
async def test_registered_parser_is_selected_before_parsing(
    parser_fakes: dict[str, Any],
    filename: str,
    body: bytes,
    parser_name: str,
    expected: str,
) -> None:
    provider = _ChoiceProvider(parser_name)

    text = await extract_text(
        file_bytes=body,
        filename=filename,
        ingestion_binding=_binding(provider),
    )

    assert text == expected
    assert provider.calls == 1
    assert provider.questions[0].name == "parser"
    assert parser_name in provider.questions[0].options
    assert set(provider.questions[0].options) == set(list_parser_names())
    assert provider.states[0].text == ""
    assert provider.states[0].payload["document"]["filename"] == filename


async def test_accepted_parser_emits_one_structured_decision_line(
    parser_fakes: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    provider = _ChoiceProvider("FakePDFParser")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(provider),
        )

    lines = _decision_lines(caplog)
    assert len(lines) == 1
    for fragment in (
        "decision point=ingestion",
        "path=decision",
        "provider=recorder",
        "outcome=accepted",
        "fallback_reason=null",
    ):
        assert fragment in lines[0]


async def test_unknown_parser_falls_back_to_the_extension_default(
    parser_fakes: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    provider = _ChoiceProvider("TelepathyParser")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(provider),
        )

    assert text == "pdf text"
    assert "fallback_reason=unknown_parser" in _decision_lines(caplog)[0]


async def test_parser_allowlist_narrows_candidates_and_rejects_an_out_of_allowlist_pick(
    parser_fakes: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("DOCXParser")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(provider, options={"parsers": ["FakePDFParser", "CSVParser"]}),
        )

    assert text == "pdf text"
    assert provider.questions[0].options == ("CSVParser", "FakePDFParser")
    assert "fallback_reason=out_of_allowlist" in _decision_lines(caplog)[0]


async def test_unknown_parser_option_falls_back_without_a_provider_call(
    parser_fakes: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("FakePDFParser")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(provider, options={"parsers": ["FakePDFParser", "NopeParser"]}),
        )

    assert text == "pdf text"
    assert provider.calls == 0
    assert "fallback_reason=invalid_options" in _decision_lines(caplog)[0]
    assert "latency_ms=null" in _decision_lines(caplog)[0]


async def test_single_entry_parser_allowlist_selects_without_a_provider_call(
    parser_fakes: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("FakePDFParser")
    descriptor = DocumentDescriptor.from_bytes(file_bytes=b"%PDF-1.4 fake", filename="report.pdf")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selection = await select_document_parser(
            descriptor,
            _binding(provider, options={"parsers": ["FakePDFParser"]}),
        )

    assert selection.accepted
    assert selection.value == "FakePDFParser"
    assert selection.source == "configured"
    assert provider.calls == 0
    assert _decision_lines(caplog) == []


async def test_single_entry_parser_allowlist_is_used_by_extract_text(
    parser_fakes: dict[str, Any],
) -> None:
    provider = _ChoiceProvider("FakePDFParser")

    text = await extract_text(
        file_bytes=b"%PDF-1.4 fake",
        filename="report.pdf",
        ingestion_binding=_binding(provider, options={"parsers": ["FakePDFParser"]}),
    )

    assert text == "pdf text"
    assert provider.calls == 0


async def test_malformed_parser_provider_result_falls_back(
    parser_fakes: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _MalformedProvider:
        async def decide(self, state, questions, *, timeout=None):
            return object()

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(_MalformedProvider()),
        )

    assert text == "pdf text"
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_too_many_parser_candidates_falls_back_without_a_call(
    parser_fakes: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("CSVParser")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(provider, max_candidates=1),
        )

    assert text == "pdf text"
    assert provider.calls == 0
    assert "fallback_reason=too_many_candidates" in _decision_lines(caplog)[0]
    assert "latency_ms=null" in _decision_lines(caplog)[0]


async def test_too_few_parser_candidates_falls_back_without_a_call(
    parser_fakes: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("orchid_ai.documents.decision.list_parser_names", lambda: ("OnlyParser",))
    provider = _ChoiceProvider("OnlyParser")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(provider),
        )

    assert text == "pdf text"
    assert provider.calls == 0
    assert "fallback_reason=no_candidates" in _decision_lines(caplog)[0]
    assert "latency_ms=null" in _decision_lines(caplog)[0]


async def test_non_mapping_answers_fall_back(
    parser_fakes: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _BadResult:
        provider = "bad"

        def __init__(self) -> None:
            self.answers: list[Any] = []

    class _BadProvider:
        async def decide(self, state, questions, *, timeout=None):
            return _BadResult()

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(_BadProvider()),
        )

    assert text == "pdf text"
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_non_choice_answer_falls_back(
    parser_fakes: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _BooleanAnswerProvider:
        async def decide(self, state, questions, *, timeout=None):
            question = questions[0]
            return OrchidDecisionResult(
                answers={
                    question.name: OrchidDecisionAnswer(
                        name=question.name,
                        kind="boolean",
                        value=True,
                        confidence=0.95,
                    )
                },
                provider="bad-kind",
                model="bad-kind",
            )

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(_BooleanAnswerProvider()),
        )

    assert text == "pdf text"
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_non_list_parser_allowlist_falls_back_without_a_call(
    parser_fakes: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("FakePDFParser")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(provider, options={"parsers": "FakePDFParser"}),
        )

    assert text == "pdf text"
    assert provider.calls == 0
    assert "fallback_reason=invalid_options" in _decision_lines(caplog)[0]
    assert "latency_ms=null" in _decision_lines(caplog)[0]


async def test_low_confidence_parser_falls_back(parser_fakes: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("CSVParser", confidence=0.2)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(provider, threshold=0.8),
        )

    assert text == "pdf text"
    assert "fallback_reason=low_confidence" in _decision_lines(caplog)[0]


async def test_refused_parser_falls_back(parser_fakes: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(_RefusalProvider()),
        )

    assert text == "pdf text"
    assert "fallback_reason=refusal" in _decision_lines(caplog)[0]


async def test_timed_out_parser_falls_back(parser_fakes: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(_TimeoutProvider()),
        )

    assert text == "pdf text"
    assert "fallback_reason=timeout" in _decision_lines(caplog)[0]


async def test_failed_parser_provider_falls_back(
    parser_fakes: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(_ErrorProvider("provider_unavailable")),
        )

    assert text == "pdf text"
    assert "fallback_reason=provider_unavailable" in _decision_lines(caplog)[0]


async def test_unexpected_parser_provider_error_falls_back(
    parser_fakes: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(_RaisingProvider()),
        )

    assert text == "pdf text"
    assert "fallback_reason=provider_error" in _decision_lines(caplog)[0]


async def test_missing_parser_answer_falls_back(parser_fakes: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=_binding(_EmptyProvider()),
        )

    assert text == "pdf text"
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_disabled_parser_point_keeps_the_legacy_path_without_a_decision_line(
    parser_fakes: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        text = await extract_text(
            file_bytes=b"%PDF-1.4 fake",
            filename="report.pdf",
            ingestion_binding=None,
        )

    assert text == "pdf text"
    assert _decision_lines(caplog) == []


# ── Strategy selection ──────────────────────────────────────────


async def test_accepted_strategy_is_used_after_extraction(
    selected_strategy: type[_MarkerStrategy],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("selected")
    writer = _FakeWriter()

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        count = await ingest_document(
            file_bytes=b"ignored",
            filename="notes.md",
            scope=_scope(),
            writer=writer,
            pre_extracted_text="alpha beta gamma",
            ingestion_binding=_binding(provider),
        )

    assert count == 1
    assert writer.batches[0][0][0].page_content == "alpha beta gamma"
    assert writer.batches[0][0][0].metadata["marker"] == "selected"
    assert provider.questions[0].name == "strategy"
    assert provider.states[0].text == "alpha beta gamma"
    assert len(_decision_lines(caplog)) == 1


async def test_unknown_strategy_falls_back_to_the_configured_strategy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("telepathy")
    writer = _FakeWriter()
    configured = _MarkerStrategy("configured")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        count = await ingest_document(
            file_bytes=b"ignored",
            filename="notes.md",
            scope=_scope(),
            writer=writer,
            ingestion=configured,
            pre_extracted_text="alpha beta",
            ingestion_binding=_binding(provider),
        )

    assert count == 1
    assert writer.batches[0][0][0].metadata["marker"] == "configured"
    assert "fallback_reason=unknown_strategy" in _decision_lines(caplog)[0]
    assert provider.calls == 1


async def test_selected_strategy_reuses_the_configured_instance_for_its_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecursiveLike(_MarkerStrategy):
        def __init__(self, marker: str = "selected") -> None:
            super().__init__(marker)

    monkeypatch.setitem(INGESTION_REGISTRY, "recursive", RecursiveLike)
    provider = _ChoiceProvider("recursive")
    writer = _FakeWriter()
    configured = RecursiveLike("configured-knobs")

    count = await ingest_document(
        file_bytes=b"ignored",
        filename="notes.md",
        scope=_scope(),
        writer=writer,
        ingestion=configured,
        pre_extracted_text="alpha beta",
        ingestion_binding=_binding(provider),
    )

    assert count == 1
    assert configured.calls
    assert writer.batches[0][0][0].metadata["marker"] == "configured-knobs"


async def test_strategy_allowlist_unknown_name_falls_back_without_a_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("recursive")
    writer = _FakeWriter()
    configured = _MarkerStrategy("configured")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        count = await ingest_document(
            file_bytes=b"ignored",
            filename="notes.md",
            scope=_scope(),
            writer=writer,
            ingestion=configured,
            pre_extracted_text="alpha",
            ingestion_binding=_binding(provider, options={"strategies": ["recursive", "telepathy"]}),
        )

    assert count == 1
    assert writer.batches[0][0][0].metadata["marker"] == "configured"
    assert provider.calls == 0
    assert "fallback_reason=invalid_options" in _decision_lines(caplog)[0]
    assert "latency_ms=null" in _decision_lines(caplog)[0]


async def test_single_entry_strategy_allowlist_selects_without_a_provider_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("semantic")
    descriptor = DocumentDescriptor.from_bytes(file_bytes=b"x", filename="notes.md", text="alpha beta")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selection = await select_ingestion_strategy(
            descriptor,
            _binding(provider, options={"strategies": ["recursive"]}),
        )

    assert selection.accepted
    assert selection.value == "recursive"
    assert selection.source == "configured"
    assert provider.calls == 0
    assert _decision_lines(caplog) == []


async def test_single_entry_strategy_allowlist_is_used_by_ingest_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConfiguredRecursive(_MarkerStrategy):
        def __init__(self) -> None:
            super().__init__("single-entry-recursive")

    monkeypatch.setitem(INGESTION_REGISTRY, "recursive", ConfiguredRecursive)
    provider = _ChoiceProvider("semantic")
    writer = _FakeWriter()

    count = await ingest_document(
        file_bytes=b"ignored",
        filename="notes.md",
        scope=_scope(),
        writer=writer,
        pre_extracted_text="alpha",
        ingestion_binding=_binding(provider, options={"strategies": ["recursive"]}),
    )

    assert count == 1
    assert writer.batches[0][0][0].metadata["marker"] == "single-entry-recursive"
    assert provider.calls == 0


async def test_strategy_out_of_allowlist_falls_back_to_the_configured_strategy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("semantic")
    writer = _FakeWriter()
    configured = _MarkerStrategy("configured")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await ingest_document(
            file_bytes=b"ignored",
            filename="notes.md",
            scope=_scope(),
            writer=writer,
            ingestion=configured,
            pre_extracted_text="alpha",
            ingestion_binding=_binding(provider, options={"strategies": ["recursive", "headered"]}),
        )

    assert writer.batches[0][0][0].metadata["marker"] == "configured"
    assert "fallback_reason=out_of_allowlist" in _decision_lines(caplog)[0]


async def test_low_confidence_strategy_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("recursive", confidence=0.3)
    writer = _FakeWriter()
    configured = _MarkerStrategy("configured")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await ingest_document(
            file_bytes=b"ignored",
            filename="notes.md",
            scope=_scope(),
            writer=writer,
            ingestion=configured,
            pre_extracted_text="alpha",
            ingestion_binding=_binding(provider, threshold=0.8),
        )

    assert writer.batches[0][0][0].metadata["marker"] == "configured"
    assert "fallback_reason=low_confidence" in _decision_lines(caplog)[0]


async def test_refused_strategy_falls_back() -> None:
    writer = _FakeWriter()
    configured = _MarkerStrategy("configured")

    count = await ingest_document(
        file_bytes=b"ignored",
        filename="notes.md",
        scope=_scope(),
        writer=writer,
        ingestion=configured,
        pre_extracted_text="alpha",
        ingestion_binding=_binding(_RefusalProvider()),
    )

    assert count == 1
    assert writer.batches[0][0][0].metadata["marker"] == "configured"


async def test_failed_strategy_provider_falls_back() -> None:
    writer = _FakeWriter()
    configured = _MarkerStrategy("configured")

    count = await ingest_document(
        file_bytes=b"ignored",
        filename="notes.md",
        scope=_scope(),
        writer=writer,
        ingestion=configured,
        pre_extracted_text="alpha",
        ingestion_binding=_binding(StubDecisionModel(mode="fail")),
    )

    assert count == 1
    assert writer.batches[0][0][0].metadata["marker"] == "configured"


async def test_disabled_strategy_point_keeps_the_configured_strategy_without_a_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer = _FakeWriter()
    configured = _MarkerStrategy("configured")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        count = await ingest_document(
            file_bytes=b"ignored",
            filename="notes.md",
            scope=_scope(),
            writer=writer,
            ingestion=configured,
            pre_extracted_text="alpha",
            ingestion_binding=None,
        )

    assert count == 1
    assert writer.batches[0][0][0].metadata["marker"] == "configured"
    assert _decision_lines(caplog) == []


# ── Parse-once and payload bounds ───────────────────────────────


async def test_parse_once_reuses_the_extracted_text_for_the_strategy_gate(
    parser_fakes: dict[str, Any],
    selected_strategy: type[_MarkerStrategy],
) -> None:
    provider = _QuestionRoutingProvider(parser="FakeTextParser", strategy="selected")
    binding = _binding(provider)
    body = b"one two three"

    text = await extract_text(file_bytes=body, filename="notes.md", ingestion_binding=binding)
    writer = _FakeWriter()
    count = await ingest_document(
        file_bytes=body,
        filename="notes.md",
        scope=_scope(),
        writer=writer,
        pre_extracted_text=text,
        ingestion_binding=binding,
    )

    assert count == 1
    assert parser_fakes["TextParser"].calls == [(body, "notes.md")]
    assert selected_strategy.instances[0].calls[0]["text"] == "one two three"
    assert provider.questions == ["parser", "strategy"]


async def test_full_document_body_never_reaches_the_decision_state(caplog: pytest.LogCaptureFixture) -> None:
    body = ("paragraph text " * 400) + "VISIBLE-ONLY-IN-FULL-BODY"
    provider = _ChoiceProvider("recursive")
    writer = _FakeWriter()

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await ingest_document(
            file_bytes=body.encode(),
            filename="large.md",
            scope=_scope(),
            writer=writer,
            pre_extracted_text=body,
            ingestion_binding=_binding(provider),
        )

    state = provider.states[0]
    assert len(state.text) == MAX_SNAPSHOT_CHARS
    assert "VISIBLE-ONLY-IN-FULL-BODY" not in state.text
    assert "VISIBLE-ONLY-IN-FULL-BODY" not in repr(state.payload)
    assert "VISIBLE-ONLY-IN-FULL-BODY" not in caplog.text


# ── Graph wiring ────────────────────────────────────────────────


def _agent_entry(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "prompt": "you are a test agent",
        "llm": {"model": "test/model"},
        "rag": {"namespace": "kb"},
    }


def _graph_config(*, agent_level: bool = False) -> OrchidAgentsConfig:
    agent = _agent_entry("test agent")
    data: dict[str, Any] = {
        "version": "1",
        "agents": {"a": agent},
        "decision_models": {"stub": {"protocol": "stub"}},
    }
    if agent_level:
        agent["decisions"] = {"points": {"ingestion": {"model": "stub"}}}
    else:
        data["decisions"] = {"points": {"ingestion": {"model": "stub"}}}
    return OrchidAgentsConfig.model_validate(data)


def _graph_runtime() -> OrchidRuntime:
    chat = MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    return OrchidRuntime(
        default_model="test/model",
        chat_model=chat,
        decision_models={"stub": StubDecisionModel(answers={"strategy": {"choice": "recursive", "confidence": 0.95}})},
    )


def test_graph_resolves_the_agent_scoped_ingestion_binding() -> None:
    config = _graph_config(agent_level=True)
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    binding = agents["a"]._ingestion_binding
    assert binding is not None
    assert binding.point == "ingestion"
    assert binding.provider == "stub"


def test_graph_resolves_the_global_ingestion_binding() -> None:
    config = _graph_config(agent_level=False)
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    binding = agents["a"]._ingestion_binding
    assert binding is not None
    assert binding.point == "ingestion"
    assert binding.provider == "stub"
    assert not hasattr(agents["a"]._rag_pipeline, "_ingestion_binding")


def test_graph_without_an_ingestion_binding_keeps_the_legacy_path() -> None:
    config = OrchidAgentsConfig.model_validate({"version": "1", "agents": {"a": _agent_entry("test agent")}})
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    assert agents["a"]._ingestion_binding is None
    assert not hasattr(agents["a"]._rag_pipeline, "_ingestion_binding")


def test_missing_runtime_provider_yields_no_ingestion_binding(caplog: pytest.LogCaptureFixture) -> None:
    config = _graph_config()
    runtime = OrchidRuntime(default_model="test/model")

    with caplog.at_level(logging.WARNING, logger="orchid_ai.graph.decision_bindings"):
        binding = resolve_decision_binding(config, runtime, "ingestion", agent_name="a")

    assert binding is None
    assert "missing from the runtime" in caplog.text


async def test_agent_public_ingest_document_uses_the_binding_end_to_end(
    parser_fakes: dict[str, Any],
    selected_strategy: type[_MarkerStrategy],
) -> None:
    provider = _QuestionRoutingProvider(parser="FakeTextParser", strategy="selected")
    chat = MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    runtime = OrchidRuntime(
        default_model="test/model",
        chat_model=chat,
        decision_models={"stub": provider},
    )
    agents: dict[str, Any] = {}
    build_graph(config=_graph_config(), runtime=runtime, agents_out=agents)
    agent = agents["a"]
    body = b"a b c"

    text = await agent.extract_text(file_bytes=body, filename="notes.txt")
    writer = _FakeWriter()
    count = await agent.ingest_document(
        file_bytes=body,
        filename="notes.txt",
        scope=_scope(),
        writer=writer,
        pre_extracted_text=text,
    )

    assert count == 1
    assert writer.batches[0][1] == "uploads"
    assert writer.batches[0][0][0].metadata["marker"] == "selected"
    assert parser_fakes["TextParser"].calls == [(body, "notes.txt")]
    assert selected_strategy.instances[0].calls[0]["text"] == "a b c"
    assert provider.questions == ["parser", "strategy"]


async def test_agent_ingest_document_uses_the_configured_rag_ingestion() -> None:
    agent = _agent_entry("test agent")
    agent["rag"] = {
        "namespace": "kb",
        "ingestion": {"chunk_size": 400, "chunk_overlap": 0},
    }
    config = OrchidAgentsConfig.model_validate({"version": "1", "agents": {"a": agent}})
    agents: dict[str, Any] = {}
    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    text = "word " * 300
    writer = _FakeWriter()
    count = await agents["a"].ingest_document(
        file_bytes=text.encode(),
        filename="notes.txt",
        scope=_scope(),
        writer=writer,
        pre_extracted_text=text,
    )

    documents = writer.batches[0][0]
    assert count == len(documents) > 1
    assert max(len(document.page_content) for document in documents) <= 400


def test_child_agents_receive_their_own_ingestion_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    import orchid_ai.graph.graph as graph_module

    child = _agent_entry("child")
    child["decisions"] = {"points": {"ingestion": {"model": "stub"}}}
    parent = _agent_entry("parent")
    parent["children"] = {"kid": child, "sibling": _agent_entry("sibling")}
    config = OrchidAgentsConfig.model_validate(
        {
            "version": "1",
            "agents": {"parent": parent},
            "decision_models": {"stub": {"protocol": "stub"}},
        }
    )
    created: dict[str, Any] = {}
    real_instantiate = graph_module._instantiate_agent

    def _capture(name, agent_config, *args, **kwargs):
        agent = real_instantiate(name, agent_config, *args, **kwargs)
        created[name] = agent
        return agent

    monkeypatch.setattr(graph_module, "_instantiate_agent", _capture)
    build_graph(config=config, runtime=_graph_runtime())

    assert created["kid"]._ingestion_binding is not None
    assert created["sibling"]._ingestion_binding is None
    assert not hasattr(created["kid"]._rag_pipeline, "_ingestion_binding")


# ── Low-level helpers ───────────────────────────────────────────


def test_parser_name_lookup_only_accepts_registered_names() -> None:
    assert "PDFParser" in list_parser_names()
    assert isinstance(get_parser_by_name("PDFParser"), DocumentParser)
    with pytest.raises(ValueError, match="Unknown parser"):
        get_parser_by_name("TelepathyParser")


async def test_gate_returns_a_disabled_selection_without_a_binding() -> None:
    descriptor = DocumentDescriptor.from_bytes(file_bytes=b"x", filename="a.txt")
    assert (await select_document_parser(descriptor, None)).source == "disabled"
    assert (await select_ingestion_strategy(descriptor, None)).source == "disabled"


def test_rag_pipeline_does_not_own_ingestion_state() -> None:
    pipeline = RagPipeline(reader=MagicMock())
    assert not hasattr(pipeline, "_ingestion_binding")
    with pytest.raises(TypeError):
        RagPipeline(reader=MagicMock(), ingestion_binding=_binding(_ChoiceProvider("recursive")))


def test_pipeline_extract_text_signature_remains_backward_compatible() -> None:
    assert "ingestion_binding" in pipeline_module.extract_text.__kwdefaults__
