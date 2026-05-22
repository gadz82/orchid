from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OrchidContentSourceConfig(BaseModel):
    path: str
    source: str = "local"
    file_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".txt", ".md", ".docx", ".xlsx", ".csv"])
    metadata: dict[str, str] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")
