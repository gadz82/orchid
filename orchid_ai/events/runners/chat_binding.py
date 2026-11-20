"""Shared Bloom chat-binding deliverer for agent and pipeline job runners.

The agent (:class:`GraphJobRunner`) and pipeline (:class:`PipelineJobRunner`)
runners resolve the same §25 binding and post the same success/failure
messages. This module owns that policy exactly once, so the two paths cannot
drift; agent metadata and behaviour are unchanged.
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections.abc import Mapping
from typing import Any

from ...core.events.errors import (
    ChatBindingForbiddenError,
    ChatBindingTargetNotFoundError,
)

_logger = logging.getLogger(__name__)


def _now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


# ORCHID-MEM: 01a1f342-646f-7973-a04e-c5a4ea8bbfb4 — Shared chat-binding deliverer; agent and pipeline Bloom runs resolve and post through one policy.
class ChatBindingDeliverer:
    """Resolve one run's §25 binding and post success/failure messages."""

    def __init__(
        self,
        *,
        chat_storage: Any | None,
        clock: Any | None = None,
    ) -> None:
        self._chat_storage = chat_storage
        self._clock = clock or _now

    async def resolve_binding(self, run: Any, auth: Any) -> dict[str, Any] | None:
        """Return the binding iff it exists, resolves and is writable."""
        binding = run.spec.chat_binding
        if binding is None:
            if run.spec.proactive_chat:
                return await self._create_proactive_chat(run, auth)
            return None
        if self._chat_storage is None:
            _logger.warning(
                "trigger %s opted into chat binding but no chat_storage is wired into the runner — dropping binding",
                run.spec.trigger_id,
            )
            return None

        chat_id = binding.get("chat_id")
        if not isinstance(chat_id, str) or not chat_id:
            raise ChatBindingTargetNotFoundError(str(chat_id))

        chat = await self._chat_storage.get_chat_metadata(chat_id)
        if chat is None:
            raise ChatBindingTargetNotFoundError(chat_id)
        allowed = await self._chat_storage.can_write(chat, auth)
        if not allowed:
            raise ChatBindingForbiddenError(chat_id, getattr(auth, "user_id", ""))
        return binding

    async def _create_proactive_chat(self, run: Any, auth: Any) -> dict[str, Any] | None:
        if self._chat_storage is None:
            _logger.warning(
                "trigger %s has proactive_chat=true but no chat_storage is wired into the runner — skipping",
                run.spec.trigger_id,
            )
            return None
        tenant_id = getattr(auth, "tenant_key", "default")
        user_id = getattr(auth, "user_id", "")
        fallback = run.spec.trigger_id
        if run.spec.target == "pipeline" and run.spec.pipeline_name:
            fallback = f"pipeline:{run.spec.pipeline_name}"
        source = run.spec.prompt or fallback
        title = next((line.strip() for line in source.splitlines() if line.strip()), fallback)[:120]
        session = await self._chat_storage.create_chat(
            tenant_id=tenant_id,
            user_id=user_id,
            title=title,
        )
        _logger.info(
            "[ChatBinding] proactive_chat: created chat %s for user %s (trigger=%s)",
            session.id,
            user_id,
            run.spec.trigger_id,
        )
        return {"chat_id": session.id}

    async def deliver_success(
        self,
        binding: dict[str, Any] | None,
        run: Any,
        result: Mapping[str, Any],
        *,
        agents_used: list[str] | None = None,
    ) -> None:
        if binding is None or self._chat_storage is None:
            return
        metadata = self._message_metadata(run, status="succeeded")
        if binding.get("mode") == "append_with_metadata":
            metadata["bloom_metadata"] = self._bloom_metadata(run)
        try:
            await self._chat_storage.add_message(
                chat_id=binding["chat_id"],
                role="assistant",
                content=self._extract_final_content(result),
                agents_used=self._agents_for(run, agents_used),
                metadata=metadata,
            )
        except Exception:
            _logger.exception(
                "chat-binding append failed for run %s — result is still available via /runs/%s",
                run.run_id,
                run.run_id,
            )

    async def deliver_failure(
        self,
        binding: dict[str, Any] | None,
        run: Any,
        exc: BaseException,
        *,
        agents_used: list[str] | None = None,
    ) -> None:
        if binding is None or self._chat_storage is None:
            return
        if binding.get("on_failure", "post_error") != "post_error":
            return
        metadata = self._message_metadata(run, status="failed")
        try:
            await self._chat_storage.add_message(
                chat_id=binding["chat_id"],
                role="assistant",
                content=self._render_failure_message(run, exc),
                agents_used=self._agents_for(run, agents_used),
                metadata=metadata,
            )
        except Exception:
            _logger.exception("failure-path chat-binding append failed for run %s", run.run_id)

    # ── Helpers ─────────────────────────────────────────────

    def _message_metadata(self, run: Any, *, status: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "origin": "bloom",
            "bloom_run_id": str(run.run_id),
            "trigger_id": run.spec.trigger_id,
            "signal_id": str(run.spec.signal_id),
            "delivered_at": self._clock().isoformat(),
        }
        if run.spec.target == "pipeline":
            metadata["pipeline"] = run.spec.pipeline_name
            metadata["pipeline_run_id"] = (run.result or {}).get("pipeline_run_id")
        if status == "failed":
            metadata["status"] = "failed"
        return metadata

    def _bloom_metadata(self, run: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "trigger_id": run.spec.trigger_id,
            "signal_id": str(run.spec.signal_id),
        }
        if run.spec.target == "pipeline":
            metadata["pipeline"] = run.spec.pipeline_name
            metadata["pipeline_run_id"] = (run.result or {}).get("pipeline_run_id")
        else:
            metadata["agent"] = run.spec.agent_name
        return metadata

    @staticmethod
    def _agents_for(run: Any, override: list[str] | None) -> list[str]:
        if override is not None:
            return list(override)
        if run.spec.target == "pipeline":
            return [f"pipeline:{run.spec.pipeline_name}"]
        return [run.spec.agent_name] if run.spec.agent_name else []

    @staticmethod
    def _extract_final_content(result: Mapping[str, Any]) -> str:
        if not result:
            return ""
        for key in ("final_response", "content", "message"):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
        return str(result)

    @staticmethod
    def _render_failure_message(run: Any, exc: BaseException) -> str:
        return (
            "I wasn't able to finish that background task. "
            f"You can inspect the run for details: run id "
            f"{run.run_id} (trigger {run.spec.trigger_id})."
        )
