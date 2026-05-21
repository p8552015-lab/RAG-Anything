"""rag_query — 對知識庫做純文字查詢

mode 對應 LightRAG 的 QueryParam.mode：

| mode    | 行為                                              |
|---------|---------------------------------------------------|
| local   | 走局部圖檢索（圍繞具體實體）                      |
| global  | 走全局圖檢索（主題級概念）                        |
| hybrid  | local + global 合併（預設）                       |
| mix     | hybrid + 純向量檢索（最完整但最慢）               |
| naive   | 純向量檢索，無圖譜                                |
| bypass  | 略過 LLM，直接回原文 chunks（適合取原文）         |

注意 RAG-Anything 原 ``aquery`` 預設 mode 是 ``"mix"`` 而非 README 範例的
``"hybrid"``；本 tool 預設 ``"hybrid"`` 為了在「品質 vs 速度」之間取折衷。
"""

from __future__ import annotations

import time
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP

from raganything.mcp import rag_instance
from raganything.mcp.schemas import QueryOutput


QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def rag_query(
        query: str,
        mode: QueryMode = "hybrid",
        top_k: Optional[int] = None,
        vlm_enhanced: Optional[bool] = None,
    ) -> QueryOutput:
        """對已索引的文件做純文字查詢。

        Args:
            query: 查詢文字。
            mode: 查詢模式 (local/global/hybrid/mix/naive/bypass)。
            top_k: 檢索 top_k 數量；未指定走 LightRAG 預設 (通常 40)。
            vlm_enhanced: 是否啟用 VLM 把 context 中的圖片轉 base64 給 vision_model
                          回答。``None`` 表示「有 VISION_MODEL 就啟用」(預設)；
                          ``False`` 強制純文字；``True`` 沒 VISION_MODEL 會走純文字 fallback。

        Returns:
            QueryOutput: answer (含 LightRAG 的 citation tag)、實際 mode、耗時。
        """
        start = time.time()
        rag = await rag_instance.get_rag()

        kwargs: dict = {}
        if top_k is not None:
            kwargs["top_k"] = top_k
        if vlm_enhanced is not None:
            kwargs["vlm_enhanced"] = vlm_enhanced

        answer = await rag.aquery(query, mode=mode, **kwargs)

        return QueryOutput(
            answer=answer or "",
            mode=mode,
            elapsed_seconds=round(time.time() - start, 2),
        )
