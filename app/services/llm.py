from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import AsyncGenerator

from openai import OpenAI

from app.core.config import get_settings


class LLMClient:
    SYSTEM_PROMPT = "You are an enterprise RAG assistant. Answer only from provided context and cite with [n]."
    CHAT_SYSTEM_PROMPT = (
        "You are a concise enterprise assistant for casual conversation and product guidance. "
        "Do not fabricate private data. Keep answers short and practical."
    )

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
        try:
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
        except Exception:
            return self._fallback_answer(query, contexts)

    async def stream_answer(self, query: str, contexts: list[dict]) -> AsyncGenerator[str, None]:
        if not self.client:
            answer = self._fallback_answer(query, contexts)
            async for token in self.stream_text(answer):
                yield token
            return

        messages = self._build_messages(query, contexts)
        try:
            async for token in self._stream_openai_chat(messages=messages, temperature=0.1):
                yield token
            return
        except Exception:
            answer = await asyncio.to_thread(self.generate_answer, query, contexts)
            async for token in self.stream_text(answer):
                yield token

    async def stream_chat_answer(self, query: str) -> AsyncGenerator[str, None]:
        if not self.client:
            answer = self._fallback_chat_answer(query)
            async for token in self.stream_text(answer):
                yield token
            return

        messages = [
            {"role": "system", "content": self.CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            async for token in self._stream_openai_chat(messages=messages, temperature=0.2):
                yield token
            return
        except Exception:
            answer = await asyncio.to_thread(self.generate_chat_answer, query)
            async for token in self.stream_text(answer):
                yield token

    async def stream_text(self, text: str) -> AsyncGenerator[str, None]:
        # Stream in short chunks for SSE.
        chunk_size = 30
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]

    def generate_chat_answer(self, query: str) -> str:
        if not self.client:
            return self._fallback_chat_answer(query)

        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": self.CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
            )
            if not response.choices:
                return self._fallback_chat_answer(query)
            content = response.choices[0].message.content
            if isinstance(content, str) and content.strip():
                return content
            return self._fallback_chat_answer(query)
        except Exception:
            return self._fallback_chat_answer(query)

    def classify_route(self, query: str) -> dict:
        # Fast rule-based shortcut first to avoid unnecessary LLM calls.
        rule_hit = self._rule_based_route(query)
        if rule_hit:
            return rule_hit

        if not self.client:
            return {
                "route": "rag",
                "confidence": 0.55,
                "source": "fallback",
                "reason": "no llm client for classifier",
            }

        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You classify user intent for a knowledge assistant.\n"
                            "Return strict JSON: {\"route\":\"chat|rag\",\"confidence\":0-1,\"reason\":\"...\"}.\n"
                            "chat = greetings, identity, chit-chat, non-document small talk.\n"
                            "rag = questions that should be answered from enterprise documents."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}",
                    },
                ],
                temperature=0.0,
            )
            if not response.choices:
                raise ValueError("empty classifier response")
            raw = response.choices[0].message.content or ""
            parsed = self._parse_classifier_json(raw)
            if parsed:
                parsed["source"] = "llm"
                return parsed
            raise ValueError("invalid classifier payload")
        except Exception:
            return {
                "route": "rag",
                "confidence": 0.55,
                "source": "fallback",
                "reason": "classifier failed",
            }

    def build_debug_payload(self, query: str, contexts: list[dict], mode: str = "classic") -> dict:
        if mode == "chat":
            messages = [
                {"role": "system", "content": self.CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
        else:
            messages = self._build_messages(query, contexts)
        return {
            "provider": self.settings.llm_provider,
            "model": self.settings.llm_model,
            "mode": mode,
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

    async def _stream_openai_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        if not self.client:
            return

        queue: asyncio.Queue[object] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sentinel = object()

        def producer() -> None:
            try:
                stream = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    if not delta:
                        continue
                    text = getattr(delta, "content", None)
                    if isinstance(text, str) and text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str) and item:
                yield item

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

    @staticmethod
    def _fallback_chat_answer(query: str) -> str:
        return (
            "我是企业知识助手。闲聊问题我可以直接回答；"
            "涉及业务资料的问题我会切换到文档检索再回答。"
        )

    @staticmethod
    def _rule_based_route(query: str) -> dict | None:
        text = re.sub(r"\s+", " ", str(query or "").strip().lower())
        if not text:
            return {"route": "chat", "confidence": 0.95, "source": "rule", "reason": "empty query"}
        chat_patterns = [
            r"^你好[啊呀吗]?$",
            r"^hi$",
            r"^hello$",
            r"^你是谁[？?]?$",
            r"^介绍一下你自己",
            r"^你能做什么",
            r"^在吗$",
            r"^早上好$",
            r"^晚上好$",
        ]
        for pattern in chat_patterns:
            if re.search(pattern, text):
                return {
                    "route": "chat",
                    "confidence": 0.97,
                    "source": "rule",
                    "reason": "matched chit-chat pattern",
                }
        # If query clearly asks to summarize/explain a document, use RAG.
        rag_keywords = ["文档", "根据资料", "根据文件", "readme", "手册", "配置", "数据库", "索引", "chunk"]
        if any(key in text for key in rag_keywords):
            return {
                "route": "rag",
                "confidence": 0.9,
                "source": "rule",
                "reason": "matched document-centric keywords",
            }
        return None

    @staticmethod
    def _parse_classifier_json(raw: str) -> dict | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except Exception:
                return None
        if not isinstance(data, dict):
            return None
        route = str(data.get("route") or "").strip().lower()
        if route not in {"chat", "rag"}:
            return None
        confidence_raw = data.get("confidence")
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.75
        confidence = max(0.0, min(1.0, confidence))
        return {
            "route": route,
            "confidence": round(confidence, 3),
            "reason": str(data.get("reason") or "").strip() or "classifier decision",
        }


def sse_pack(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
