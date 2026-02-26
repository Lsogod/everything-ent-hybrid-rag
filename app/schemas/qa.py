from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None
    user_id: str | None = None
    debug: bool = True
    file_paths: list[str] | None = Field(
        default=None,
        description="Optional retrieval scope. If omitted, search all indexed files.",
    )
