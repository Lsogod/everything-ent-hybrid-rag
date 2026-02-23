from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.security import get_request_user_id, require_permission, verify_api_key
from app.infra.es import ElasticsearchStore
from app.infra.queue import celery_app
from app.schemas.admin import PermissionCreateRequest, RoleCreateRequest
from app.schemas.chat import MessagesResponse, SessionsResponse
from app.schemas.index import IndexFileRequest, TaskResponse
from app.schemas.qa import AskRequest
from app.services.chat_history import ChatHistoryService
from app.services.llm import LLMClient, sse_pack
from app.services.rbac import RBACService
from app.services.retrieval import HybridRetriever

router = APIRouter(dependencies=[Depends(verify_api_key)])
settings = get_settings()
SAMPLE_DOC_URL = "https://raw.githubusercontent.com/fastapi/fastapi/master/README.md"
SAMPLE_DOC_NAME = "fastapi_official_readme.md"
DATASET_MANIFEST = "datasets/duretrieval_manifest.json"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "reports"


@router.post("/index/file", response_model=TaskResponse)
async def index_file(
    payload: IndexFileRequest,
    _: str = Depends(require_permission("kb:index")),
) -> TaskResponse:
    file_path = str(Path(payload.file_path).resolve())
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="file not found")
    if not settings.allow_external_paths and not Path(file_path).is_relative_to(
        Path(settings.knowledge_root).resolve()
    ):
        raise HTTPException(status_code=403, detail="file path is outside knowledge_root")

    task = celery_app.send_task("worker.tasks.index_file", args=[file_path])
    return TaskResponse(task_id=task.id, message=f"index task queued: {file_path}")


@router.delete("/index/file", response_model=TaskResponse)
async def delete_file(
    payload: IndexFileRequest,
    _: str = Depends(require_permission("kb:delete")),
) -> TaskResponse:
    file_path = str(Path(payload.file_path).resolve())
    if not settings.allow_external_paths and not Path(file_path).is_relative_to(
        Path(settings.knowledge_root).resolve()
    ):
        raise HTTPException(status_code=403, detail="file path is outside knowledge_root")
    task = celery_app.send_task("worker.tasks.delete_file", args=[file_path])
    return TaskResponse(task_id=task.id, message=f"delete task queued: {file_path}")


@router.post("/qa/ask")
async def ask(
    payload: AskRequest,
    request_user_id: str = Depends(require_permission("qa:ask")),
) -> StreamingResponse:
    retriever = HybridRetriever()
    llm = LLMClient()
    chat_history = ChatHistoryService()
    rbac = RBACService()

    if payload.user_id and payload.user_id != request_user_id and not rbac.has_permission(
        request_user_id, "rbac:manage"
    ):
        raise HTTPException(status_code=403, detail="cannot impersonate another user")

    effective_user_id = payload.user_id or request_user_id

    async def event_stream():
        conversation_id = payload.conversation_id or uuid4().hex
        docs, trace = await retriever.retrieve_with_trace(payload.query)
        citations = [
            {
                "index": idx,
                "file_path": item.get("file_path"),
                "file_name": item.get("file_name"),
                "chunk_id": item.get("chunk_id"),
            }
            for idx, item in enumerate(docs, start=1)
        ]

        yield sse_pack(
            {
                "type": "context",
                "conversation_id": conversation_id,
                "count": len(docs),
                "citations": citations,
            }
        )
        if payload.debug:
            yield sse_pack({"type": "trace", "trace": trace})
            yield sse_pack({"type": "llm_input", "llm_input": llm.build_debug_payload(payload.query, docs)})
        answer_parts: list[str] = []
        async for token in llm.stream_answer(payload.query, docs):
            answer_parts.append(token)
            yield sse_pack({"type": "token", "text": token})
        await asyncio.to_thread(
            chat_history.save_exchange,
            effective_user_id,
            conversation_id,
            payload.query,
            "".join(answer_parts),
            citations,
        )
        yield sse_pack({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/chat/sessions", response_model=SessionsResponse)
async def list_sessions(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    request_user_id: str = Depends(require_permission("chat:read")),
) -> SessionsResponse:
    rbac = RBACService()
    chat_history = ChatHistoryService()

    target_user_id = request_user_id
    if user_id and user_id != request_user_id:
        if not rbac.has_permission(request_user_id, "rbac:manage"):
            raise HTTPException(status_code=403, detail="cannot read other user's sessions")
        target_user_id = user_id
    elif user_id:
        target_user_id = user_id

    return SessionsResponse(items=chat_history.list_sessions(target_user_id, limit=limit))


@router.get("/chat/sessions/{session_id}/messages", response_model=MessagesResponse)
async def list_session_messages(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    request_user_id: str = Depends(require_permission("chat:read")),
) -> MessagesResponse:
    rbac = RBACService()
    chat_history = ChatHistoryService()

    owner_id = chat_history.get_session_owner(session_id)
    if not owner_id:
        raise HTTPException(status_code=404, detail="session not found")

    if owner_id != request_user_id and not rbac.has_permission(request_user_id, "rbac:manage"):
        raise HTTPException(status_code=403, detail="cannot read other user's session messages")

    return MessagesResponse(items=chat_history.list_messages(session_id, limit=limit))


@router.post("/admin/bootstrap")
async def bootstrap_rbac() -> dict:
    rbac = RBACService()
    rbac.bootstrap_defaults()
    return {
        "status": "ok",
        "acl_enabled": settings.acl_enabled,
        "default_admin": settings.acl_default_admin_user,
    }


@router.post("/admin/roles")
async def create_role(
    payload: RoleCreateRequest,
    _: str = Depends(require_permission("rbac:manage")),
) -> dict:
    return RBACService().create_role(payload.name)


@router.post("/admin/permissions")
async def create_permission(
    payload: PermissionCreateRequest,
    _: str = Depends(require_permission("rbac:manage")),
) -> dict:
    return RBACService().create_permission(payload.code, payload.description)


@router.post("/admin/roles/{role_name}/permissions/{permission_code}")
async def bind_permission_to_role(
    role_name: str,
    permission_code: str,
    _: str = Depends(require_permission("rbac:manage")),
) -> dict:
    return RBACService().assign_permission_to_role(role_name, permission_code)


@router.post("/admin/users/{user_id}/roles/{role_name}")
async def assign_role_to_user(
    user_id: str,
    role_name: str,
    _: str = Depends(require_permission("rbac:manage")),
) -> dict:
    return RBACService().assign_role_to_user(user_id, role_name)


@router.get("/admin/users/{user_id}/permissions")
async def get_user_permissions(
    user_id: str,
    _: str = Depends(require_permission("rbac:manage")),
) -> dict:
    return {"user_id": user_id, "permissions": RBACService().list_user_permissions(user_id)}


@router.get("/me/permissions")
async def get_my_permissions(
    request_user_id: str = Depends(get_request_user_id),
) -> dict:
    return {"user_id": request_user_id, "permissions": RBACService().list_user_permissions(request_user_id)}


@router.post("/debug/sample-doc")
async def download_sample_doc(
    _: str = Depends(require_permission("kb:index")),
) -> dict:
    target = Path(settings.knowledge_root).resolve() / SAMPLE_DOC_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    source_candidates = [
        SAMPLE_DOC_URL,
        "https://cdn.jsdelivr.net/gh/fastapi/fastapi@master/README.md",
    ]
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for source_url in source_candidates:
            try:
                response = await client.get(source_url)
                response.raise_for_status()
                target.write_bytes(response.content)
                return {
                    "status": "ok",
                    "file_path": str(target),
                    "size_bytes": target.stat().st_size,
                    "source_url": source_url,
                }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source_url}: {exc}")

    if target.exists() and target.stat().st_size > 0:
        return {
            "status": "using_cached",
            "file_path": str(target),
            "size_bytes": target.stat().st_size,
            "source_url": None,
            "warning": "download failed, fallback to cached file",
            "errors": errors,
        }

    raise HTTPException(status_code=502, detail={"message": "failed to download sample doc", "errors": errors})


@router.get("/debug/file-stats")
async def file_stats(
    file_path: str = Query(...),
    _: str = Depends(require_permission("chat:read")),
) -> dict:
    resolved = str(Path(file_path).resolve())
    if not settings.allow_external_paths and not Path(resolved).is_relative_to(Path(settings.knowledge_root).resolve()):
        raise HTTPException(status_code=403, detail="file path is outside knowledge_root")
    count = ElasticsearchStore().count_by_file(resolved)
    return {
        "file_path": resolved,
        "chunk_count": count,
        "indexed": count > 0,
    }


@router.get("/eval/datasets/manifest")
async def eval_dataset_manifest(
    _: str = Depends(require_permission("chat:read")),
) -> dict:
    manifest_path = Path(settings.knowledge_root).resolve() / DATASET_MANIFEST
    if not manifest_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "message": "dataset manifest not found",
                "hint": "run: python scripts/materialize_duretrieval_knowledge.py",
                "path": str(manifest_path),
            },
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"invalid dataset manifest: {exc}") from exc

    store = ElasticsearchStore()
    items: list[dict] = []
    for item in raw.get("items", []):
        runtime_path = str(item.get("corpus_file_path_runtime", "")).strip()
        if runtime_path:
            resolved_runtime_path = str(Path(runtime_path).resolve())
        else:
            resolved_runtime_path = ""

        try:
            chunk_count = store.count_by_file(resolved_runtime_path) if resolved_runtime_path else 0
        except Exception:  # noqa: BLE001
            chunk_count = 0
        items.append(
            {
                **item,
                "corpus_file_path_runtime": runtime_path,
                "resolved_file_path": resolved_runtime_path,
                "chunk_count": chunk_count,
                "indexed": chunk_count > 0,
                "exists": bool(resolved_runtime_path and Path(resolved_runtime_path).exists()),
            }
        )

    return {
        "generated_at": raw.get("generated_at"),
        "manifest_path": str(manifest_path),
        "items": items,
    }


def _read_report(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _report_summary(path: Path, payload: dict) -> dict:
    results = payload.get("results") or {}
    method_metrics = {
        method: (details.get("metrics") or {})
        for method, details in results.items()
        if isinstance(details, dict)
    }
    method_cost = {
        method: details.get("cost_seconds")
        for method, details in results.items()
        if isinstance(details, dict)
    }
    stat = path.stat()
    return {
        "file_name": path.name,
        "file_path": str(path),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "created_at": payload.get("created_at"),
        "dataset": payload.get("dataset"),
        "method": payload.get("method"),
        "ks": payload.get("ks") or [],
        "top_k": payload.get("top_k"),
        "counts": payload.get("counts") or {},
        "cost_seconds_total": payload.get("cost_seconds_total"),
        "metrics": method_metrics,
        "method_cost_seconds": method_cost,
    }


@router.get("/eval/reports")
async def eval_reports(
    limit: int = Query(default=20, ge=1, le=200),
    _: str = Depends(require_permission("chat:read")),
) -> dict:
    report_dir = REPORTS_DIR
    if not report_dir.exists():
        return {
            "report_dir": str(report_dir),
            "count": 0,
            "latest": None,
            "items": [],
        }

    report_files = sorted(
        report_dir.glob("duretrieval_baseline_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    summaries: list[dict] = []
    for path in report_files[:limit]:
        payload = _read_report(path)
        if not payload:
            continue
        summaries.append(_report_summary(path, payload))

    return {
        "report_dir": str(report_dir),
        "count": len(summaries),
        "latest": summaries[0] if summaries else None,
        "items": summaries,
    }


@router.get("/eval/reports/latest")
async def eval_latest_report(
    _: str = Depends(require_permission("chat:read")),
) -> dict:
    report_dir = REPORTS_DIR
    if not report_dir.exists():
        return {"report_dir": str(report_dir), "latest": None}

    report_files = sorted(
        report_dir.glob("duretrieval_baseline_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not report_files:
        return {"report_dir": str(report_dir), "latest": None}

    path = report_files[0]
    payload = _read_report(path)
    if not payload:
        return {"report_dir": str(report_dir), "latest": None}

    return {
        "report_dir": str(report_dir),
        "latest": {
            "summary": _report_summary(path, payload),
            "report": payload,
        },
    }
