from __future__ import annotations

from orchid_ai.llm import get_embedding_kwargs, get_llm_kwargs

# ── get_llm_kwargs ──────────────────────────────────────────


def test_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gkey")
    result = get_llm_kwargs("gemini/gemini-pro")
    assert result == {"api_key": "gkey"}


def test_groq_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "grkey")
    result = get_llm_kwargs("groq/llama3")
    assert result == {"api_key": "grkey"}


def test_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
    result = get_llm_kwargs("anthropic/claude-3")
    assert result == {"api_key": "akey"}


def test_claude_prefix_uses_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "akey2")
    result = get_llm_kwargs("claude-3-opus")
    assert result == {"api_key": "akey2"}


def test_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "okey")
    result = get_llm_kwargs("openai/gpt-4")
    assert result == {"api_key": "okey"}


def test_ollama_api_base(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434")
    result = get_llm_kwargs("ollama/llama3")
    assert result == {"api_base": "http://localhost:11434"}


def test_unknown_prefix_returns_empty():
    result = get_llm_kwargs("some_unknown/model")
    assert result == {}


def test_missing_env_var_returns_empty(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = get_llm_kwargs("gemini/gemini-pro")
    assert result == {}


# ── get_embedding_kwargs ────────────────────────────────────


def test_embedding_delegates_to_llm_kwargs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "okey")
    result = get_embedding_kwargs("openai/text-embedding-3-small")
    assert result == {"api_key": "okey"}


def test_embedding_bare_model_gets_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "okey")
    # "text-embedding-3-small" — no "/", model[:4]="text" has no "-"
    result = get_embedding_kwargs("text-embedding-3-small")
    assert result == {"api_key": "okey"}


def test_embedding_bare_model_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = get_embedding_kwargs("text-embedding-3-small")
    assert result == {}


def test_embedding_bare_model_with_dash_in_first_four_skips_openai(monkeypatch):
    # model[:4] = "gpt-" which contains "-", so bare-model logic is skipped
    monkeypatch.setenv("OPENAI_API_KEY", "okey")
    result = get_embedding_kwargs("gpt-4")
    # No "/" so it won't match any prefix either — empty dict
    assert result == {}
