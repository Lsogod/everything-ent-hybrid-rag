from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.core.logging import logger
from app.infra.db import SessionLocal
from app.models.entities import Permission, Role, RolePermission, User, UserRole


@dataclass(frozen=True)
class RBACBootstrap:
    permissions: tuple[str, ...] = (
        "kb:index",
        "kb:delete",
        "qa:ask",
        "chat:read",
        "rbac:manage",
    )


class RBACService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def acl_enabled(self) -> bool:
        return self.settings.acl_enabled

    def bootstrap_defaults(self) -> None:
        if not self.settings.acl_enabled or not self.settings.acl_bootstrap:
            return

        spec = RBACBootstrap()
        with SessionLocal() as db:
            try:
                admin_role = self._get_or_create_role(db, "admin")
                user_role = self._get_or_create_role(db, "user")

                for code in spec.permissions:
                    perm = self._get_or_create_permission(db, code)
                    self._grant_role_permission(db, admin_role.id, perm.id)

                for code in ("qa:ask", "chat:read"):
                    perm = db.scalar(select(Permission).where(Permission.code == code))
                    if perm:
                        self._grant_role_permission(db, user_role.id, perm.id)

                default_user = self._get_or_create_user(db, self.settings.acl_default_admin_user)
                self._grant_user_role(db, default_user.id, admin_role.id)
                db.commit()
            except SQLAlchemyError as exc:
                db.rollback()
                logger.exception("RBAC bootstrap failed: %s", exc)

    def create_role(self, role_name: str) -> dict:
        with SessionLocal() as db:
            role = self._get_or_create_role(db, role_name)
            db.commit()
            return {"id": role.id, "name": role.name}

    def create_permission(self, code: str, description: str = "") -> dict:
        with SessionLocal() as db:
            permission = db.scalar(select(Permission).where(Permission.code == code))
            if not permission:
                permission = Permission(code=code, description=description)
                db.add(permission)
                db.flush()
            else:
                permission.description = description or permission.description
            db.commit()
            return {"id": permission.id, "code": permission.code, "description": permission.description}

    def assign_role_to_user(self, user_id: str, role_name: str) -> dict:
        with SessionLocal() as db:
            user = self._get_or_create_user(db, user_id)
            role = self._get_or_create_role(db, role_name)
            self._grant_user_role(db, user.id, role.id)
            db.commit()
            return {"user_id": user.id, "role": role.name}

    def assign_permission_to_role(self, role_name: str, permission_code: str) -> dict:
        with SessionLocal() as db:
            role = self._get_or_create_role(db, role_name)
            permission = self._get_or_create_permission(db, permission_code)
            self._grant_role_permission(db, role.id, permission.id)
            db.commit()
            return {"role": role.name, "permission": permission.code}

    def list_user_permissions(self, user_id: str) -> list[str]:
        if not self.settings.acl_enabled:
            return ["*"]

        with SessionLocal() as db:
            stmt = (
                select(Permission.code)
                .join(RolePermission, RolePermission.permission_id == Permission.id)
                .join(Role, Role.id == RolePermission.role_id)
                .join(UserRole, UserRole.role_id == Role.id)
                .where(UserRole.user_id == user_id)
            )
            rows = db.scalars(stmt).all()
            return sorted(set(rows))

    def has_permission(self, user_id: str, permission_code: str) -> bool:
        if not self.settings.acl_enabled:
            return True
        permissions = self.list_user_permissions(user_id)
        return permission_code in permissions or "*" in permissions

    @staticmethod
    def _get_or_create_user(db, user_id: str) -> User:
        user = db.scalar(select(User).where(User.id == user_id))
        if user:
            return user
        user = User(id=user_id, username=user_id, password_hash="disabled", is_active=True)
        db.add(user)
        db.flush()
        return user

    @staticmethod
    def _get_or_create_role(db, role_name: str) -> Role:
        role = db.scalar(select(Role).where(Role.name == role_name))
        if role:
            return role
        role = Role(name=role_name)
        db.add(role)
        db.flush()
        return role

    @staticmethod
    def _get_or_create_permission(db, code: str, description: str = "") -> Permission:
        permission = db.scalar(select(Permission).where(Permission.code == code))
        if permission:
            return permission
        permission = Permission(code=code, description=description)
        db.add(permission)
        db.flush()
        return permission

    @staticmethod
    def _grant_user_role(db, user_id: str, role_id: int) -> None:
        existing = db.scalar(
            select(UserRole).where(UserRole.user_id == user_id, UserRole.role_id == role_id)
        )
        if existing:
            return
        db.add(UserRole(user_id=user_id, role_id=role_id))

    @staticmethod
    def _grant_role_permission(db, role_id: int, permission_id: int) -> None:
        existing = db.scalar(
            select(RolePermission).where(
                RolePermission.role_id == role_id,
                RolePermission.permission_id == permission_id,
            )
        )
        if existing:
            return
        db.add(RolePermission(role_id=role_id, permission_id=permission_id))
