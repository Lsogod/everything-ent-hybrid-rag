from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from app.core.config import get_settings
from app.services.llm import LLMClient
from app.services.retrieval import HybridRetriever


class AgenticRAGService:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient) -> None:
        self.settings = get_settings()
        self.retriever = retriever
        self.llm = llm

    async def run(
        self,
        query: str,
        file_paths: list[str] | None = None,
        max_iterations: int = 2,
        max_sub_queries: int = 3,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        normalized_scope = self._normalize_file_paths(file_paths)
        max_iterations = max(1, min(int(max_iterations), 5))
        max_sub_queries = max(1, min(int(max_sub_queries), 8))

        plan = await self._plan(query, max_sub_queries)
        queue = plan.get("sub_queries") or [query]
        events: list[dict[str, Any]] = [{"type": "agent_plan", "agent_plan": plan}]

        seen_queries: set[str] = set()
        evidence: dict[str, dict[str, Any]] = {}
        steps: list[dict[str, Any]] = []
        refined_count = 0
        stop_reason = "queue_exhausted"

        while queue and len(steps) < max_iterations:
            current_query = queue.pop(0).strip()
            if not current_query or current_query in seen_queries:
                continue
            seen_queries.add(current_query)

            docs, trace = await self.retriever.retrieve_with_trace(current_query, normalized_scope)
            self._merge_evidence(evidence, docs)
            step_no = len(steps) + 1

            step = {
                "iteration": step_no,
                "query": current_query,
                "scope": trace.get("scope", {"mode": "all", "file_count": 0, "file_paths": []}),
                "context_count": len(docs),
                "fusion_count": int(trace.get("fusion", {}).get("count", 0)),
                "timing_ms": trace.get("timing_ms", {}),
                "flow": trace.get("flow", {}),
                "retrieval_mode": trace.get("strategy", {}).get("mode", "classic"),
                "candidate_preview": [self._preview_hit(item) for item in docs[:3]],
            }
            steps.append({"query": current_query, "docs": docs, "trace": trace, "step": step})
            events.append({"type": "agent_step", "agent_step": step})

            sufficient = self._is_sufficient(docs, trace)
            if sufficient:
                stop_reason = "sufficient_evidence"
                events.append(
                    {
                        "type": "agent_judge",
                        "agent_judge": {
                            "iteration": step_no,
                            "action": "stop",
                            "reason": "evidence is sufficient for answer generation",
                            "next_query": None,
                        },
                    }
                )
                break

            if len(steps) >= max_iterations:
                stop_reason = "max_iterations"
                events.append(
                    {
                        "type": "agent_judge",
                        "agent_judge": {
                            "iteration": step_no,
                            "action": "stop",
                            "reason": "max iterations reached",
                            "next_query": None,
                        },
                    }
                )
                break

            next_query = None
            if queue:
                next_query = queue[0]
                action = "continue"
                reason = "continue with planned sub-query"
            else:
                next_query = await self._refine_query(query, docs)
                if next_query and next_query not in seen_queries:
                    queue.append(next_query)
                    refined_count += 1
                    action = "refine"
                    reason = "insufficient evidence; generated follow-up query"
                else:
                    action = "stop"
                    reason = "insufficient evidence and no valid follow-up query"
                    stop_reason = "insufficient_no_refine"

            events.append(
                {
                    "type": "agent_judge",
                    "agent_judge": {
                        "iteration": step_no,
                        "action": action,
                        "reason": reason,
                        "next_query": next_query,
                    },
                }
            )
            if action == "stop":
                break

        final_docs = self._select_final_docs(evidence)
        if not final_docs and steps:
            final_docs = list(steps[-1]["docs"])[: self.settings.retrieval_top_k_final]

        final_trace = self._build_agent_trace(
            query=query,
            scope_paths=normalized_scope,
            plan=plan,
            steps=steps,
            final_docs=final_docs,
            stop_reason=stop_reason,
            refined_count=refined_count,
        )
        events.append(
            {
                "type": "agent_done",
                "agent_done": {
                    "iterations": len(steps),
                    "stop_reason": stop_reason,
                    "evidence_count": len(evidence),
                    "selected_contexts": len(final_docs),
                },
            }
        )
        return final_docs, final_trace, events

    async def _plan(self, query: str, max_sub_queries: int) -> dict[str, Any]:
        llm_plan = await self._plan_with_llm(query, max_sub_queries)
        if llm_plan:
            return llm_plan

        sub_queries = self._fallback_sub_queries(query, max_sub_queries)
        return {
            "goal": query,
            "sub_queries": sub_queries,
            "reasoning": "fallback planner split by punctuation and conjunctions",
            "planner": "rule_based",
        }

    async def _plan_with_llm(self, query: str, max_sub_queries: int) -> dict[str, Any] | None:
        if not self.llm.client:
            return None

        def _call() -> str:
            response = self.llm.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a retrieval planner. Return strict JSON only with keys: "
                            "goal, sub_queries, reasoning."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {query}\n"
                            f"Max sub queries: {max_sub_queries}\n"
                            "Generate focused search sub-queries for enterprise RAG retrieval. "
                            "Do not include markdown."
                        ),
                    },
                ],
                temperature=0.0,
            )
            if not response.choices:
                return ""
            return response.choices[0].message.content or ""

        try:
            raw = await asyncio.to_thread(_call)
            parsed = self._parse_json_object(raw)
            if not parsed:
                return None
            sub_queries = self._normalize_sub_queries(parsed.get("sub_queries"), max_sub_queries, query)
            if not sub_queries:
                return None
            return {
                "goal": str(parsed.get("goal") or query).strip() or query,
                "sub_queries": sub_queries,
                "reasoning": str(parsed.get("reasoning") or "").strip() or "generated by llm planner",
                "planner": "llm",
            }
        except Exception:
            return None

    async def _refine_query(self, query: str, docs: list[dict[str, Any]]) -> str | None:
        if self.llm.client:
            refined = await self._refine_query_with_llm(query, docs)
            if refined:
                return refined

        if not docs:
            return f"{query} 关键步骤"
        key = str(docs[0].get("file_name") or docs[0].get("file_path") or "").strip()
        if not key:
            return f"{query} 详细说明"
        return f"{query} {key} 详细说明"

    async def _refine_query_with_llm(self, query: str, docs: list[dict[str, Any]]) -> str | None:
        if not self.llm.client:
            return None

        preview = "\n".join(
            [
                f"- {item.get('file_name') or item.get('file_path')}: {self._truncate(str(item.get('content') or ''), 140)}"
                for item in docs[:3]
            ]
        )

        def _call() -> str:
            response = self.llm.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You refine enterprise search queries. Return one short query text only.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Original question: {query}\n"
                            f"Current evidence preview:\n{preview}\n"
                            "Give one better retrieval query to improve recall."
                        ),
                    },
                ],
                temperature=0.0,
            )
            if not response.choices:
                return ""
            return response.choices[0].message.content or ""

        try:
            raw = await asyncio.to_thread(_call)
            refined = re.sub(r"\s+", " ", str(raw or "").strip())
            return refined[:200] if refined else None
        except Exception:
            return None

    def _build_agent_trace(
        self,
        query: str,
        scope_paths: list[str],
        plan: dict[str, Any],
        steps: list[dict[str, Any]],
        final_docs: list[dict[str, Any]],
        stop_reason: str,
        refined_count: int,
    ) -> dict[str, Any]:
        if steps:
            base_trace = dict(steps[-1]["trace"])
        else:
            base_trace = {
                "query": query,
                "scope": {
                    "mode": "filtered" if scope_paths else "all",
                    "file_count": len(scope_paths),
                    "file_paths": scope_paths,
                },
                "timing_ms": {},
                "keyword": {"count": 0, "hits": []},
                "vector": {"count": 0, "hits": []},
                "fusion": {"count": 0, "hits": [], "detailed_hits": [], "dropped_candidates": []},
                "metrics": {
                    "keyword_unique": 0,
                    "vector_unique": 0,
                    "union_unique": 0,
                    "overlap_unique": 0,
                    "overlap_rate": 0.0,
                    "fusion_gain": 0,
                },
                "flow": {
                    "keyword_candidates": 0,
                    "vector_candidates": 0,
                    "union_candidates": 0,
                    "overlap_candidates": 0,
                    "final_selected": 0,
                    "dropped_candidates": 0,
                },
            }

        base_trace["query"] = query
        base_trace["strategy"] = {
            "mode": "agentic",
            "max_iterations": len(steps),
            "refined_count": refined_count,
        }
        base_trace["agent"] = {
            "plan": plan,
            "steps": [item["step"] for item in steps],
            "iterations": len(steps),
            "stop_reason": stop_reason,
            "evidence_count": len({item.get("_id") for item in final_docs if item.get("_id")}),
            "selected_contexts": len(final_docs),
        }

        timing_ms = dict(base_trace.get("timing_ms") or {})
        timing_ms["agent_total"] = round(
            sum(float(item["step"].get("timing_ms", {}).get("total", 0.0)) for item in steps), 2
        )
        base_trace["timing_ms"] = timing_ms
        return base_trace

    def _select_final_docs(self, evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        scored = list(evidence.values())
        scored.sort(
            key=lambda item: (
                float(item.get("rrf_score") or 0.0),
                float(item.get("_score") or 0.0),
            ),
            reverse=True,
        )
        final = scored[: self.settings.retrieval_top_k_final]
        for idx, item in enumerate(final, start=1):
            item["_rank"] = idx
        return final

    def _merge_evidence(self, evidence: dict[str, dict[str, Any]], docs: list[dict[str, Any]]) -> None:
        for item in docs:
            doc_id = str(item.get("_id") or "").strip()
            if not doc_id:
                continue
            existing = evidence.get(doc_id)
            if existing is None:
                evidence[doc_id] = dict(item)
                continue
            candidate_score = (float(item.get("rrf_score") or 0.0), float(item.get("_score") or 0.0))
            current_score = (float(existing.get("rrf_score") or 0.0), float(existing.get("_score") or 0.0))
            if candidate_score > current_score:
                evidence[doc_id] = dict(item)

    def _is_sufficient(self, docs: list[dict[str, Any]], trace: dict[str, Any]) -> bool:
        if len(docs) >= min(3, self.settings.retrieval_top_k_final):
            return True
        fusion_count = int(trace.get("fusion", {}).get("count", 0))
        if fusion_count >= min(3, self.settings.retrieval_top_k_final):
            return True
        overlap_rate = float(trace.get("metrics", {}).get("overlap_rate", 0.0) or 0.0)
        return len(docs) >= 2 and overlap_rate >= 0.2

    @staticmethod
    def _normalize_file_paths(file_paths: list[str] | None) -> list[str]:
        if not file_paths:
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in file_paths:
            value = str(raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    @staticmethod
    def _fallback_sub_queries(query: str, max_sub_queries: int) -> list[str]:
        parts = re.split(r"[，,。；;？?\n]+", query)
        tokens = []
        for part in parts:
            piece = re.sub(r"\s+", " ", part.strip())
            if len(piece) < 2:
                continue
            tokens.append(piece)
        if not tokens:
            return [query]
        deduped: list[str] = []
        seen: set[str] = set()
        for item in tokens:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if len(deduped) >= max_sub_queries:
                break
        return deduped or [query]

    @staticmethod
    def _normalize_sub_queries(raw: Any, max_sub_queries: int, query: str) -> list[str]:
        if not isinstance(raw, list):
            return [query]
        deduped: list[str] = []
        seen: set[str] = set()
        for item in raw:
            value = re.sub(r"\s+", " ", str(item or "").strip())
            if len(value) < 2 or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
            if len(deduped) >= max_sub_queries:
                break
        return deduped or [query]

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _preview_hit(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "file_name": item.get("file_name"),
            "file_path": item.get("file_path"),
            "chunk_id": item.get("chunk_id"),
            "rrf_score": item.get("rrf_score"),
            "score": item.get("_score"),
        }

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip())
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 3] + "..."
