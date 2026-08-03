"""Loader: registers external-agent CLI tools into TOOL_REGISTRY from config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..config.tool_registry import TOOL_REGISTRY
from ..tools.cli_runner import CLIRunner
from ..tools.external_cli import ExternalAgentCLITool
from ..tools.normalizer import LLMNormalizer, PassthroughNormalizer, PromptNormalizer

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from ..config.schema_external_agent import OrchidExternalAgentConfig

logger = logging.getLogger(__name__)


def load_external_agents_from_config(
    external_agents: dict[str, OrchidExternalAgentConfig],
    *,
    chat_model: BaseChatModel | None = None,
    runner: CLIRunner | None = None,
) -> None:
    for name, cfg in external_agents.items():
        normalizer = _build_normalizer(name, cfg, chat_model)
        tool = ExternalAgentCLITool(
            name=name,
            description=cfg.description or name,
            command=cfg.command,
            args=cfg.args,
            cwd=cfg.cwd or None,
            timeout=cfg.timeout,
            env=cfg.env or None,
            stdin_mode=cfg.stdin_mode,
            normalizer=normalizer,
            runner=runner,
        )
        tool.requires_approval = cfg.requires_approval
        tool.parallel_safe = cfg.parallel_safe
        tool.inject_to_rag = cfg.inject_to_rag
        tool.rag_ttl = cfg.rag_ttl
        if name in TOOL_REGISTRY.get_all():
            TOOL_REGISTRY.unregister(name)
        TOOL_REGISTRY.register(tool)
        logger.info("[Graph] registered external-agent tool '%s'", name)


def _build_normalizer(
    name: str,
    cfg: OrchidExternalAgentConfig,
    chat_model: BaseChatModel | None,
) -> PromptNormalizer:
    if cfg.normalizer == "llm":
        if chat_model is not None:
            return LLMNormalizer(chat_model, instruction=cfg.normalizer_instruction)
        logger.warning(
            "[ExternalAgents] '%s' requested llm normalizer but no chat_model available; falling back to passthrough",
            name,
        )
    return PassthroughNormalizer()
