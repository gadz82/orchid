"""Smoke tests — every shipped example YAML loads through the new schema.

The redesign drops ``retriever_type`` and ``reformulate_queries``;
example YAMLs were migrated in the same change.  These tests pin the
post-migration shape so future drift surfaces immediately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchid_ai.config.loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


# Each entry: (yaml path, dict of env vars the loader needs interpolated).
# The loader rejects unresolved ``${VAR}`` placeholders, so the test sets
# placeholder values for examples that declare external services.
_EXAMPLES: list[tuple[Path, dict[str, str]]] = [
    (REPO_ROOT / "examples" / "basketball" / "agents.yaml", {}),
    (REPO_ROOT / "examples" / "restaurant" / "config" / "agents.yaml", {}),
    (REPO_ROOT / "examples" / "helpdesk" / "config" / "agents.yaml", {}),
    (REPO_ROOT / "examples" / "travel-agency" / "config" / "agents.yaml", {}),
    (REPO_ROOT / "examples" / "prompt-customization" / "agents.yaml", {}),
    (REPO_ROOT / "examples" / "custom-storage" / "agents.yaml", {}),
    (REPO_ROOT / "examples" / "rag-strategies" / "agents.yaml", {}),
    (REPO_ROOT / "examples" / "wiki" / "agents.yaml", {}),
    (REPO_ROOT / "examples" / "graph_kb" / "agents.yaml", {}),
    (
        REPO_ROOT / "examples" / "tool-strategies" / "agents.yaml",
        {"KB_MCP_URL": "http://localhost:9001/mcp"},
    ),
    (
        REPO_ROOT / "examples" / "mcp-auth" / "agents.yaml",
        {
            "LOCAL_MCP_URL": "http://localhost:8081/mcp",
            "INTERNAL_MCP_URL": "http://localhost:8082/mcp",
            "CRM_MCP_URL": "http://localhost:8083/mcp",
        },
    ),
]


@pytest.mark.parametrize("path,env", _EXAMPLES, ids=lambda v: str(v) if isinstance(v, Path) else "env")
def test_example_yaml_loads(path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Every example YAML resolves through ``load_config`` without error."""
    if not path.exists():
        pytest.skip(f"Example file not present: {path}")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    config = load_config(str(path))
    assert config.agents, f"{path} resolved with zero agents"


def test_restaurant_uses_multi_query_strategy() -> None:
    path = REPO_ROOT / "examples" / "restaurant" / "config" / "agents.yaml"
    if not path.exists():
        pytest.skip(f"Example file not present: {path}")
    config = load_config(str(path))
    menu = config.agents["menu"]
    assert menu.rag.retrieval.strategy == "multi_query"
    assert menu.rag.retrieval.query_transformers == ["reformulate"]


def test_helpdesk_uses_reformulate_transformer() -> None:
    path = REPO_ROOT / "examples" / "helpdesk" / "config" / "agents.yaml"
    if not path.exists():
        pytest.skip(f"Example file not present: {path}")
    config = load_config(str(path))
    support = config.agents["support"]
    assert support.rag.retrieval.strategy == "simple"
    assert support.rag.retrieval.query_transformers == ["reformulate"]


def test_travel_agency_uses_multi_query_strategy() -> None:
    path = REPO_ROOT / "examples" / "travel-agency" / "config" / "agents.yaml"
    if not path.exists():
        pytest.skip(f"Example file not present: {path}")
    config = load_config(str(path))
    itinerary = config.agents["itinerary"]
    assert itinerary.rag.retrieval.strategy == "multi_query"
    assert itinerary.rag.retrieval.query_transformers == ["reformulate"]


def test_custom_storage_example_loads() -> None:
    """The custom-storage example wires a single agent and no RAG."""
    path = REPO_ROOT / "examples" / "custom-storage" / "agents.yaml"
    if not path.exists():
        pytest.skip(f"Example file not present: {path}")
    config = load_config(str(path))
    assert "echo" in config.agents
    assert config.agents["echo"].rag.enabled is False


def test_tool_strategies_example_threads_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each tool-strategies agent picks a different per-server strategy."""
    path = REPO_ROOT / "examples" / "tool-strategies" / "agents.yaml"
    if not path.exists():
        pytest.skip(f"Example file not present: {path}")
    monkeypatch.setenv("KB_MCP_URL", "http://localhost:9001/mcp")
    config = load_config(str(path))

    expected = {
        "fanout_lookup": "all",
        "pipeline_lookup": "sequential",
        "smart_lookup": "llm_decides",
        "cascade_lookup": "priority",
    }
    for agent_name, expected_strategy in expected.items():
        agent = config.agents[agent_name]
        servers = agent.mcp_servers
        assert servers, f"{agent_name} expected an MCP server"
        assert servers[0].tool_call_strategy == expected_strategy

    # parallel_searcher exercises the orthogonal parallel_tools flag.
    assert config.agents["parallel_searcher"].parallel_tools is True


def test_rag_strategies_example_threads_strategies() -> None:
    """Each rag-strategies agent picks a different retrieval strategy."""
    path = REPO_ROOT / "examples" / "rag-strategies" / "agents.yaml"
    if not path.exists():
        pytest.skip(f"Example file not present: {path}")
    config = load_config(str(path))

    expected = {
        "simple_searcher": "simple",
        "multi_query_searcher": "multi_query",
        "hyde_searcher": "hyde",
        "recency_searcher": "recency_simple",
    }
    for agent_name, expected_strategy in expected.items():
        assert config.agents[agent_name].rag.retrieval.strategy == expected_strategy


def test_wiki_example_threads_per_tool_override() -> None:
    """Stage 7 — wiki example resolves per-tool RAG via ``effective_rag``."""
    path = REPO_ROOT / "examples" / "wiki" / "agents.yaml"
    if not path.exists():
        pytest.skip(f"Example file not present: {path}")
    config = load_config(str(path))

    docs = config.agents["docs"]
    assert docs.rag.ingestion.strategy == "headered"
    assert docs.rag.retrieval.strategy == "hybrid"

    # Per-tool override (ADR-024) flips namespace + ingestion + retrieval.
    eff = docs.effective_rag("lookup_glossary")
    assert eff.namespace == "glossary_cache"
    assert eff.ingestion.strategy == "semantic"
    assert eff.retrieval.strategy == "simple"


def test_graph_kb_example_threads_graph_rag() -> None:
    """Stage 5 — graph_kb example uses ``graph_rag`` retrieval."""
    path = REPO_ROOT / "examples" / "graph_kb" / "agents.yaml"
    if not path.exists():
        pytest.skip(f"Example file not present: {path}")
    config = load_config(str(path))

    org_chart = config.agents["org_chart"]
    assert org_chart.rag.retrieval.strategy == "graph_rag"
    assert org_chart.rag.retrieval.graph.enabled is True
    assert org_chart.rag.retrieval.graph.max_hops == 2
    assert "reports_to" in org_chart.rag.retrieval.graph.relation_filter


def test_prompt_customization_example_threads_overrides() -> None:
    """The prompt-customization example must surface every override site."""
    path = REPO_ROOT / "examples" / "prompt-customization" / "agents.yaml"
    if not path.exists():
        pytest.skip(f"Example file not present: {path}")
    config = load_config(str(path))

    advisor = config.agents["legal_advisor"]
    sections = advisor.prompt_sections
    assert "MEMOIRE FROM PRIOR TURNS" in sections.prior_results_header
    assert "REFERENCE EXHIBITS" in sections.resources_header
    assert "SOURCE CITATIONS" in sections.rag_header
    assert sections.prior_results_max_chars == 8000
    assert sections.resource_max_chars == 4000
    # Summarise overrides applied.
    assert "REMINDER" in sections.summarise_history_reminder
    assert "COUNSEL'S PRIOR FINDINGS" in sections.summarise_prior_results_header
    assert "Authoritative excerpts" in sections.summarise_rag_section_header
    assert "{query}" in sections.summarise_user_template
    assert "{rag_section}" in sections.summarise_user_template
    assert "{mcp_data}" in sections.summarise_user_template
    assert sections.summarise_prior_results_max_chars == 6000

    prompts = advisor.rag.retrieval.transformer_prompts
    # Per-agent overrides applied.
    assert prompts.multi_query is not None
    assert prompts.hyde.single is not None
    assert prompts.hyde.multi is not None
    assert prompts.decompose is not None
    # Inherited from defaults.rag.retrieval.transformer_prompts.
    assert prompts.reformulate is not None
    assert "STANDALONE legal" in prompts.reformulate

    # Mini-agent template threaded through.
    assert advisor.mini_agent.system_prompt_template is not None
    assert "{parent_prompt}" in advisor.mini_agent.system_prompt_template
    assert "{instruction}" in advisor.mini_agent.system_prompt_template
    assert "{tool_list}" in advisor.mini_agent.system_prompt_template
