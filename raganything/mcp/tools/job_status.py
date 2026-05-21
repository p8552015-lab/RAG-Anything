"""rag_get_job_status — 查背景 job（add_document 等）的進度與結果

搭配非同步化的 rag_add_document 使用：add 提交後回 job_id，這裡輪詢直到
status 變 completed（result 內含 doc_id/chunk_count 等）或 failed（error 有訊息）。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from raganything.mcp import jobs
from raganything.mcp.schemas import JobStatusOutput


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def rag_get_job_status(job_id: str) -> JobStatusOutput:
        """查背景工作的狀態與結果。

        Args:
            job_id: rag_add_document 回傳的 job_id。

        Returns:
            JobStatusOutput: status 為 pending/processing/completed/failed/not_found。
            completed 時 result 含 AddDocumentOutput（doc_id、chunk_count、elapsed 等）；
            failed 時 error 有錯誤訊息。
        """
        job = jobs.get_job(job_id)
        if job is None:
            return JobStatusOutput(job_id=job_id, status="not_found")
        return JobStatusOutput(
            job_id=job.job_id,
            status=job.status,
            file_name=job.file_name,
            elapsed_seconds=job.elapsed_seconds(),
            result=job.result,
            error=job.error,
        )
