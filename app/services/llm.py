from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from openai import OpenAI

from app.core.config import get_settings


class LLMClient:
    SYSTEM_PROMPT = "You are an enterprise RAG assistant. Answer only from provided context and cite with [n]."

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = (
            OpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url.rstrip("/"),
                timeout=self.settings.request_timeout_seconds,
                max_retries=2,
            )
            if self.settings.llm_api_key
            else None
        )

    def generate_answer(self, query: str, contexts: list[dict]) -> str:
        if not self.client:
            return self._fallback_answer(query, contexts)

        messages = self._build_messages(query, contexts)
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=0.1,
        )
        if not response.choices:
            return "I cannot produce an answer from current context."

        content = response.choices[0].message.content
        if isinstance(content, str) and content.strip():
            return content
        return "I cannot produce an answer from current context."

    async def stream_answer(self, query: str, contexts: list[dict]) -> AsyncGenerator[str, None]:
        answer = await asyncio.to_thread(self.generate_answer, query, contexts)
        # Stream in short chunks for SSE.
        chunk_size = 30
        for i in range(0, len(answer), chunk_size):
            yield answer[i : i + chunk_size]

    def build_debug_payload(self, query: str, contexts: list[dict]) -> dict:
        messages = self._build_messages(query, contexts)
        return {
            "provider": self.settings.llm_provider,
            "model": self.settings.llm_model,
            "base_url": self.settings.llm_base_url,
            "temperature": 0.1,
            "has_api_key": bool(self.settings.llm_api_key),
            "context_count": len(contexts),
            "messages": messages,
            "contexts": [
                {
                    "index": idx,
                    "file_name": item.get("file_name"),
                    "file_path": item.get("file_path"),
                    "chunk_id": item.get("chunk_id"),
                    "content": item.get("content"),
                }
                for idx, item in enumerate(contexts, start=1)
            ],
        }

    def _build_messages(self, query: str, contexts: list[dict]) -> list[dict[str, str]]:
        prompt = self._build_prompt(query, contexts)
        return [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def _build_prompt(self, query: str, contexts: list[dict]) -> str:
        blocks = []
        for idx, item in enumerate(contexts, start=1):
            blocks.append(f"[{idx}] file={item.get('file_name')} path={item.get('file_path')}\n{item.get('content')}")
        context_text = "\n\n".join(blocks)
        return (
            "Use ONLY the context below. If context is insufficient, say you do not know.\n"
            "Return concise answer and include citation marks like [1].\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context_text}"
        )

    @staticmethod
    def _fallback_answer(query: str, contexts: list[dict]) -> str:
        if not contexts:
            return "No relevant enterprise document was found for this question."

        lines = [f"Question: {query}", "Answer (context-only):"]
        for idx, item in enumerate(contexts[:3], start=1):
            content = (item.get("content") or "").strip().replace("\n", " ")
            preview = content[:180]
            lines.append(f"- [{idx}] {preview}")
        lines.append("Please configure LLM_API_KEY to enable generative summarization.")
        return "\n".join(lines)


def sse_pack(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
