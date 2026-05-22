from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OrchidContentItem:
    path: str
    name: str
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content: str | None = None


class OrchidContentSource(ABC):
    @abstractmethod
    async def list(self, path: str = "", recursive: bool = False, limit: int = 100) -> list[OrchidContentItem]: ...

    @abstractmethod
    async def get(self, path: str) -> OrchidContentItem: ...

    @abstractmethod
    async def search(self, query: str, recursive: bool = True, limit: int = 10) -> list[OrchidContentItem]: ...
