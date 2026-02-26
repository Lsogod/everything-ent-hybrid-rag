from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None
    user_id: str | None = None
    debug: bool = True
    route: Literal["auto", "chat", "rag"] = "auto"
    mode: Literal["classic", "agentic"] = "classic"
    agent_max_iterations: int = Field(default=2, ge=1, le=5)
    agent_max_sub_queries: int = Field(default=3, ge=1, le=8)
    file_paths: list[str] | None = Field(
        default=None,
        description="Optional retrieval scope. If omitted, search all indexed files.",
    )
