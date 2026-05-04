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
