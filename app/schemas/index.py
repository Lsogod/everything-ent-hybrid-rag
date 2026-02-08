from pydantic import BaseModel, Field


class IndexFileRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path for the source file")


class TaskResponse(BaseModel):
    task_id: str
    status: str = "queued"
    message: str
