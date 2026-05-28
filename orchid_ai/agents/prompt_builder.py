"""SystemPromptBuilder — builds the agentic-loop system prompt.

M2 refactoring: extracted from ``GenericAgent._build_agentic_system_prompt``
(833 LOC → ~60 lines).  Owns one concern — assembling the 6 prompt
sections from config + MCP metadata + RAG context.
"""

from __future__ import annotations

import json
from typing import Any

from ..config.schema import OrchidAgentPromptConfig


class SystemPromptBuilder:
    """Builds a rich system prompt for the agentic loop.

    Section templates and truncation knobs are sourced from
    ``OrchidAgentPromptConfig``.  Omitting the block from YAML uses the
    built-in defaults defined on that class.
    """

    def __init__(self, prompt_sections: OrchidAgentPromptConfig) -> None:
        self._sections = prompt_sections

    def build(
        self,
        base_prompt: str,
        *,
        caps: Any,  # MCPCapabilities
        rag_data: list[dict[str, Any]],
        state: dict[str, Any] | None,
        agent_name: str,
        rag_max_context_chars: int = 3000,
    ) -> str:
        """Assemble the full system prompt.

        Parameters
        ----------
        base_prompt : str
            The agent's base prompt from YAML.
        caps : MCPCapabilities
            Discovered MCP capabilities (tools, prompts, resources).
        rag_data : list[dict]
            Retrieved RAG context.
        state : dict | None
            Current graph state (for prior tool context).
        agent_name : str
            Agent name (used for mcp_context key).
        rag_max_context_chars : int
            Max chars for the RAG section.
        """
        parts = [base_prompt]

        # Prior tool results from previous turns
        prior_ctx = (state.get("mcp_context") or {}).get(agent_name) if state else None
        if prior_ctx:
            parts.append(self._sections.prior_results_header)
            parts.append(json.dumps(prior_ctx, indent=2, default=str)[: self._sections.prior_results_max_chars])

        # Rendered MCP prompts (zero-arg prompts evaluated at discovery time)
        if caps.rendered_prompts:
            for prompt in caps.rendered_prompts:
                parts.append(
                    self._sections.mcp_prompt_template.format(
                        name=prompt["name"],
                        text=prompt["text"],
                    )
                )

        # Prompts that require arguments — listed so the LLM knows they exist
        if caps.skipped_prompts:
            for sp in caps.skipped_prompts:
                parts.append(
                    self._sections.skipped_prompt_template.format(
                        name=sp["name"],
                        description=sp["description"],
                        required_args=", ".join(sp["required_args"]),
                    )
                )

        # MCP resource contents
        if caps.resource_contents:
            parts.append(self._sections.resources_header)
            for name, content in caps.resource_contents.items():
                parts.append(
                    self._sections.resource_template.format(
                        name=name,
                        content=content[: self._sections.resource_max_chars],
                    )
                )

        # RAG context
        if rag_data:
            parts.append(self._sections.rag_header)
            parts.append(json.dumps(rag_data, indent=2, default=str)[:rag_max_context_chars])

        return "\n".join(parts)
