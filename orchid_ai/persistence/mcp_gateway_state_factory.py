"""Factory for :class:`OrchidMCPGatewayClientStore` /
:class:`OrchidMCPGatewayAuthCodeStore` /
:class:`OrchidMCPGatewayTokenStore` backends.

A single class implements all three ABCs (see
:class:`OrchidSQLiteMCPGatewayStateStore`), so this factory returns
the concrete instance typed as the union — downstream code can cast
to whichever interface it needs.  Mirrors
:func:`build_mcp_client_registration_store` / :func:`build_mcp_token_store`.
"""

from __future__ import annotations

import logging

from ..core.mcp_gateway_state import (
    OrchidMCPGatewayAuthCodeStore,
    OrchidMCPGatewayClientStore,
    OrchidMCPGatewayTokenStore,
)
from ..utils import import_class

logger = logging.getLogger(__name__)


def build_mcp_gateway_state_store(
    class_path: str,
    dsn: str,
    *,
    extra_migrations_package: str | None = None,
) -> OrchidMCPGatewayClientStore | OrchidMCPGatewayAuthCodeStore | OrchidMCPGatewayTokenStore:
    """Dynamically import and instantiate an MCP-gateway-state backend.

    The returned instance must subclass all three ABCs — the type
    union is declared only to express the caller's freedom to pick
    any of the three interface views.  Runtime error if the resolved
    class doesn't cover all three.
    """
    try:
        cls = import_class(class_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot resolve MCP-gateway-state store class '{class_path}'.  "
            f"Ensure it is a valid dotted import path to a class that "
            f"subclasses all three of OrchidMCPGatewayClientStore, "
            f"OrchidMCPGatewayAuthCodeStore, OrchidMCPGatewayTokenStore.  "
            f"Error: {exc}"
        ) from exc

    required = (
        OrchidMCPGatewayClientStore,
        OrchidMCPGatewayAuthCodeStore,
        OrchidMCPGatewayTokenStore,
    )
    if not (isinstance(cls, type) and all(issubclass(cls, base) for base in required)):
        raise TypeError(
            f"'{class_path}' resolves to {cls!r}, which must implement all three "
            f"MCP-gateway-state ABCs "
            f"(OrchidMCPGatewayClientStore, OrchidMCPGatewayAuthCodeStore, "
            f"OrchidMCPGatewayTokenStore).",
        )

    logger.info("[OrchidMCPGatewayStateStore] Using %s", class_path)
    return cls(dsn=dsn, extra_migrations_package=extra_migrations_package)
