"""raganything.mcp.mineru_remote — 把 mineru 解析 outsource 到遠端 mineru-api

設計重點（DRY）：
* 繼承 ``MineruParser``，只覆寫 ``_run_mineru_command`` 與 ``check_installation``
* 其餘 ``parse_pdf`` / ``parse_image`` / ``parse_document`` / ``_read_output_files``
  全部復用 —— 因為遠端 mineru-api 回傳的 ``content_list`` 與本地 mineru CLI
  輸出的 ``*_content_list.json`` 格式完全相同
* 把遠端回傳的 content_list / md / images 寫成本地 mineru 輸出目錄結構，
  讓既有的 ``_read_output_files`` 照常讀取

設定（從環境變數讀，缺值 fail-fast，遵守「禁止降級」原則）：
* ``MINERU_REMOTE_URL``       必填，例如 http://your-dgx-host:8889
* ``MINERU_REMOTE_BACKEND``   選填，預設 pipeline（DGX Spark GPU 上最快且穩）
* ``MINERU_REMOTE_TIMEOUT``   選填，單次解析逾時秒數，預設 600
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional, Union

import httpx

from raganything.parser import MineruParser


def _remote_base_url() -> str:
    url = os.environ.get("MINERU_REMOTE_URL")
    if not url:
        raise RuntimeError(
            "MINERU_REMOTE_URL 未設定。PARSER=mineru-remote 需要遠端 mineru-api 位址，"
            "例如 http://your-dgx-host:8889"
        )
    return url.rstrip("/")


class MineruRemoteParser(MineruParser):
    """走遠端 mineru-api 的 MinerU parser。"""

    __slots__ = ()
    logger = logging.getLogger(__name__)

    @classmethod
    def _run_mineru_command(
        cls,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        method: str = "auto",
        lang: Optional[str] = None,
        backend: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        formula: bool = True,
        table: bool = True,
        device: Optional[str] = None,
        source: Optional[str] = None,
        vlm_url: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> None:
        """POST 檔案到遠端 mineru-api，把結果寫成本地 mineru 輸出結構。

        寫出檔案：
        * ``{output_dir}/{stem}_content_list.json``
        * ``{output_dir}/{stem}.md``
        * ``{output_dir}/images/{name}``（若有圖）

        後續 ``MineruParser._read_output_files(output_dir, stem)`` 會讀這些檔案。
        """
        base_url = _remote_base_url()
        remote_backend = backend or os.environ.get("MINERU_REMOTE_BACKEND") or "pipeline"
        req_timeout = timeout or int(os.environ.get("MINERU_REMOTE_TIMEOUT") or "600")

        in_path = Path(input_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = in_path.stem

        if not in_path.exists():
            raise FileNotFoundError(f"檔案不存在: {in_path}")

        file_bytes = in_path.read_bytes()
        mime = mimetypes.guess_type(str(in_path))[0] or "application/octet-stream"

        data = {
            "backend": remote_backend,
            "parse_method": method or "auto",
            "lang_list": lang or "ch",
            "formula_enable": "true" if formula else "false",
            "table_enable": "true" if table else "false",
            "return_content_list": "true",
            "return_images": "true",
            "return_md": "true",
            "return_middle_json": "false",
            "return_model_output": "false",
            "response_format_zip": "false",
        }
        if start_page is not None:
            data["start_page_id"] = str(start_page)
        if end_page is not None:
            data["end_page_id"] = str(end_page)

        cls.logger.info(
            "Remote mineru-api parse: %s -> %s (backend=%s)",
            in_path.name,
            base_url,
            remote_backend,
        )

        files = {"files": (in_path.name, file_bytes, mime)}
        try:
            resp = httpx.post(
                f"{base_url}/file_parse",
                files=files,
                data=data,
                timeout=req_timeout,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"遠端 mineru-api 連線失敗 {base_url}/file_parse: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"遠端 mineru-api 回傳 HTTP {resp.status_code}: {resp.text[:500]}"
            )

        body = resp.json()
        results = body.get("results") or {}
        if not results:
            raise RuntimeError(
                f"遠端 mineru-api 沒有回傳 results: {json.dumps(body)[:500]}"
            )

        # 取第一筆（單檔上傳）
        entry = next(iter(results.values()))

        content_list_raw = entry.get("content_list")
        if content_list_raw is None:
            raise RuntimeError(
                "遠端 mineru-api 回傳缺少 content_list（請確認 return_content_list）"
            )
        # content_list 是 JSON 字串，直接寫檔讓 _read_output_files 讀
        if isinstance(content_list_raw, str):
            content_list_text = content_list_raw
        else:
            content_list_text = json.dumps(content_list_raw, ensure_ascii=False)

        (out_dir / f"{stem}_content_list.json").write_text(
            content_list_text, encoding="utf-8"
        )

        md_content = entry.get("md_content") or ""
        (out_dir / f"{stem}.md").write_text(md_content, encoding="utf-8")

        images = entry.get("images") or {}
        if images:
            images_dir = out_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            for name, data_uri in images.items():
                b64 = data_uri.split(",", 1)[-1] if "," in data_uri else data_uri
                try:
                    (images_dir / name).write_bytes(base64.b64decode(b64))
                except Exception as exc:  # 單張圖失敗不該整份失敗
                    cls.logger.warning("寫入圖片 %s 失敗: %s", name, exc)

        cls.logger.info(
            "Remote mineru-api done: %s (images=%d)", in_path.name, len(images)
        )

    def check_installation(self) -> bool:
        """以遠端 /health 取代本地 ``mineru --version`` 檢查。"""
        try:
            base_url = _remote_base_url()
        except RuntimeError as exc:
            self.logger.error("%s", exc)
            return False
        try:
            resp = httpx.get(f"{base_url}/health", timeout=10)
            if resp.status_code == 200:
                return True
            self.logger.error(
                "遠端 mineru-api /health 回 HTTP %s", resp.status_code
            )
            return False
        except httpx.HTTPError as exc:
            self.logger.error("遠端 mineru-api 連線失敗: %s", exc)
            return False
