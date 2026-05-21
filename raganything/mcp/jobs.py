"""raganything.mcp.jobs — 背景 job 管理

讓長時間 tool（add_document）能「提交即回 job_id」，背景在 MCP server 的
event loop 上跑完，避開 Claude Code 對單次 MCP tool 呼叫的 ~60s 硬 timeout（R1）。

job 狀態存在 server 進程記憶體（session 級別，server 重啟即清空）。
背景 task 用一個 module-level set 持有 reference，避免被 GC 中途回收。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

_JOB_STATES = ("pending", "processing", "completed", "failed")


@dataclass
class Job:
    """單一背景工作的狀態。"""

    job_id: str
    kind: str
    status: str = "pending"  # pending / processing / completed / failed
    file_name: str = ""
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def elapsed_seconds(self) -> float:
        end = self.completed_at or time.time()
        return round(end - self.created_at, 2)


# 模組級狀態
_jobs: dict[str, Job] = {}
_tasks: set[asyncio.Task] = set()


def create_job(kind: str, file_name: str = "") -> Job:
    """建立一個 pending job 並登記。"""
    job = Job(job_id=str(uuid.uuid4()), kind=kind, file_name=file_name)
    _jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    """取得 job（不存在回 None）。"""
    return _jobs.get(job_id)


def list_jobs() -> list[Job]:
    """回傳所有 job（新到舊）。"""
    return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def launch(
    job: Job,
    coro_factory: Callable[[], Awaitable[dict[str, Any]]],
) -> None:
    """在背景跑 coro_factory()，把結果/錯誤寫回 job。

    coro_factory 是「呼叫後回傳 awaitable」的工廠（延後建立 coroutine，
    確保在背景 task 內才真正開始執行）。
    """

    async def _runner() -> None:
        job.status = "processing"
        job.started_at = time.time()
        try:
            result = await coro_factory()
            job.result = result
            job.status = "completed"
        except Exception as exc:  # noqa: BLE001 — 任何失敗都記錄到 job，不該炸掉 loop
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "failed"
        finally:
            job.completed_at = time.time()

    task = asyncio.create_task(_runner())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
