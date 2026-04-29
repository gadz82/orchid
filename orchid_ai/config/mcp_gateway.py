"""
MCP gateway exposure configuration.

Describes how an integrator wants Orchid's MCP-facing surface
(e.g. the ``orchid-mcp`` gateway) to present itself to MCP clients:

* **Tool overrides** — replace the default ``title`` / ``description``
  of tools the gateway registers, so the host LLM sees integrator-
  provided context (e.g. "Ask the Acme Knowledge Base" instead of the
  framework default "Ask Orchid").

* **MCP Prompts** — pre-canned prompt templates (MCP spec: ``prompts/list``
  / ``prompts/get``) that the host LLM can discover and expand via the
  gateway.  Templates use simple ``{{arg_name}}`` substitution; the
  gateway renders them client-side.

This module contains **only** Pydantic data models.  The framework
does not know which tools the gateway actually exposes (keeping the
library platform-agnostic), so any string is accepted as a key in
``tools`` — validation of tool-name spelling is a gateway concern.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator


_PROMPT_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")


class OrchidMCPGatewayToolOverride(BaseModel):
    """Optional override for a tool's human-facing metadata.

    Either field can be ``None`` (meaning "leave the gateway's default
    in place") or a non-empty string.  Empty strings are allowed but
    should be treated as "no override" by the consumer so integrators
    can blank a single field back to default via YAML without deleting
    the whole entry.
    """

    title: str | None = None
    description: str | None = None


class OrchidMCPGatewayPromptArgument(BaseModel):
    """An argument accepted by an MCP prompt.

    Mirrors the MCP spec ``PromptArgument`` shape.
    """

    name: str
    description: str | None = None
    required: bool = False

    @model_validator(mode="after")
    def _validate_name(self) -> OrchidMCPGatewayPromptArgument:
        if not _PROMPT_NAME_RE.match(self.name):
            raise ValueError(
                f"Prompt argument name '{self.name}' must match /^[a-zA-Z_][a-zA-Z0-9_-]*$/",
            )
        return self


class OrchidMCPGatewayPrompt(BaseModel):
    """An MCP prompt template exposed via the gateway.

    ``template`` uses simple ``{{arg_name}}`` substitution — no loops,
    no conditionals, no expression language.  Declared arguments are
    **not** cross-validated against the template at load time; a
    template referencing an undeclared argument renders the placeholder
    literally, which makes the failure observable to the LLM caller.
    """

    name: str = Field(description="MCP prompt handle — returned by prompts/list")
    title: str | None = None
    description: str | None = None
    arguments: list[OrchidMCPGatewayPromptArgument] = Field(default_factory=list)
    template: str = Field(description="Prompt body with {{arg_name}} placeholders")

    @model_validator(mode="after")
    def _validate(self) -> OrchidMCPGatewayPrompt:
        if not _PROMPT_NAME_RE.match(self.name):
            raise ValueError(
                f"Prompt name '{self.name}' must match /^[a-zA-Z_][a-zA-Z0-9_-]*$/",
            )
        seen: set[str] = set()
        for arg in self.arguments:
            if arg.name in seen:
                raise ValueError(
                    f"Prompt '{self.name}' declares duplicate argument '{arg.name}'",
                )
            seen.add(arg.name)
        return self


class OrchidMCPGatewayConfig(BaseModel):
    """Top-level MCP gateway exposure configuration.

    ``tools`` is keyed by the canonical tool name the gateway exposes
    (e.g. ``"orchid_ask"``).  Tool names are not validated against any
    known registry — the framework does not know which tools a
    particular gateway exposes.

    ``prompts`` is a list of :class:`OrchidMCPGatewayPrompt`.  Duplicate
    ``name`` values within a single list are a configuration error.

    Example YAML::

        mcp_gateway:
          tools:
            orchid_ask:
              title: "Ask the Acme Knowledge Base"
              description: "Route a question to the Acme support agents."
            orchid_upload_file:
              description: "Index a file into the chat's RAG scope for analysis."
          prompts:
            - name: compliance_report
              title: "Compliance report"
              description: "Generate a compliance-completion report for a department."
              arguments:
                - name: department
                  description: "Department to filter by"
                  required: true
              template: |
                Using the knowledge-base agent, produce a compliance-completion
                report for the {{department}} department.

    Programmatic construction is equivalent — pass nested dicts or
    model instances and Pydantic coerces::

        cfg = OrchidMCPGatewayConfig(
            tools={"orchid_ask": {"title": "Ask the Acme Knowledge Base"}},
            prompts=[
                OrchidMCPGatewayPrompt(
                    name="hello",
                    description="...",
                    template="Say hello to {{who}}.",
                    arguments=[{"name": "who", "required": True}],
                ),
            ],
        )
    """

    tools: dict[str, OrchidMCPGatewayToolOverride] = Field(default_factory=dict)
    prompts: list[OrchidMCPGatewayPrompt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_prompt_names_unique(self) -> OrchidMCPGatewayConfig:
        seen: set[str] = set()
        for prompt in self.prompts:
            if prompt.name in seen:
                raise ValueError(
                    f"Duplicate prompt name '{prompt.name}' in mcp_gateway.prompts",
                )
            seen.add(prompt.name)
        return self
