from pydantic import BaseModel, Field


class IndexFileRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path for the source file")
    delete_source: bool = Field(
        default=False,
        description="Whether to remove the source file from knowledge_root/uploads when deleting index",
    )


class TaskResponse(BaseModel):
    task_id: str
    status: str = "queued"
    message: str


class UploadIndexResponse(BaseModel):
    task_id: str
    status: str = "queued"
    message: str
    file_name: str
    file_path: str
    size_bytes: int


class UploadFileItem(BaseModel):
    file_name: str
    file_path: str
    size_bytes: int
    modified_at: str
    chunk_count: int
    indexed: bool


class UploadListResponse(BaseModel):
    items: list[UploadFileItem]
