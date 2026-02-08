from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from app.core.config import get_settings
from app.services.rbac import RBACService


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def get_request_user_id(x_user_id: str | None = Header(default=None)) -> str:
    settings = get_settings()
    if not settings.acl_enabled:
        return x_user_id or "anonymous"
    if not x_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-User-Id is required")
    return x_user_id


def require_permission(permission_code: str):
    async def checker(user_id: str = Depends(get_request_user_id)) -> str:
        settings = get_settings()
        if not settings.acl_enabled:
            return user_id

        rbac = RBACService()
        if rbac.has_permission(user_id, permission_code) or rbac.has_permission(user_id, "rbac:manage"):
            return user_id

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {permission_code}",
        )

    return checker
