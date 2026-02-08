from pydantic import BaseModel, Field


class RoleCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)


class PermissionCreateRequest(BaseModel):
    code: str = Field(..., min_length=2, max_length=64)
    description: str = Field(default="", max_length=255)
