from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.logging import logger
from app.infra.db import SessionLocal
from app.models.entities import ChatMessage, ChatSession, User


class ChatHistoryService:
    def save_exchange(
        self,
        user_id: str | None,
        conversation_id: str,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> None:
        uid = user_id or "anonymous"
        with SessionLocal() as db:
            try:
                self._ensure_user(db, uid)
                self._ensure_session(db, conversation_id, uid)

                db.add(
                    ChatMessage(
                        session_id=conversation_id,
                        role="user",
                        content=question,
                        citations=None,
                        created_at=datetime.utcnow(),
                    )
                )
                db.add(
                    ChatMessage(
                        session_id=conversation_id,
                        role="assistant",
                        content=answer,
                        citations={"items": citations},
                        created_at=datetime.utcnow(),
                    )
                )
                db.commit()
            except SQLAlchemyError as exc:
                db.rollback()
                logger.exception("chat history persist failed: %s", exc)

    def list_sessions(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            stmt = (
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.created_at.desc())
                .limit(limit)
            )
            sessions = db.scalars(stmt).all()
            return [
                {
                    "id": item.id,
                    "user_id": item.user_id,
                    "title": item.title,
                    "created_at": item.created_at.isoformat(),
                }
                for item in sessions
            ]

    def list_messages(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
            )
            messages = db.scalars(stmt).all()
            return [
                {
                    "id": item.id,
                    "session_id": item.session_id,
                    "role": item.role,
                    "content": item.content,
                    "citations": item.citations,
                    "created_at": item.created_at.isoformat(),
                }
                for item in messages
            ]

    def get_session_owner(self, session_id: str) -> str | None:
        with SessionLocal() as db:
            session = db.scalar(select(ChatSession).where(ChatSession.id == session_id))
            return session.user_id if session else None

    @staticmethod
    def _ensure_user(db, user_id: str) -> None:
        exists = db.scalar(select(User.id).where(User.id == user_id))
        if exists:
            return
        db.add(
            User(
                id=user_id,
                username=user_id,
                password_hash="disabled",
                is_active=True,
            )
        )

    @staticmethod
    def _ensure_session(db, conversation_id: str, user_id: str) -> None:
        exists = db.scalar(select(ChatSession.id).where(ChatSession.id == conversation_id))
        if exists:
            return
        db.add(
            ChatSession(
                id=conversation_id,
                user_id=user_id,
                title=f"Session-{uuid.uuid4().hex[:8]}",
            )
        )
