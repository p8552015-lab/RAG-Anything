"""baseline_test.py — 驗證 raganything + 遠端 Ollama 端到端 (Phase 0)

跟 examples/ollama_integration_example.py 不同之處：
1. 全部走遠端 Ollama，模型名稱與 host 從 .env 讀
2. embedding 回 numpy.ndarray，繞過 LightRAG query path 對 .size 屬性的依賴
3. 用 numpy 是 Phase 1 providers.py 的固化版前置驗證

成功標準：兩個 query 都能拿到非空答案，沒有 traceback。
"""

import asyncio
import os
import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# 讓子進程能在 PATH 找到 mineru / soffice (Phase 1 server.py 會做同樣的事)
ROOT = Path(__file__).resolve().parent.parent.parent
VENV_BIN = ROOT / ".venv" / "bin"
LIBREOFFICE_BIN = Path("/Applications/LibreOffice.app/Contents/MacOS")
os.environ["PATH"] = f"{VENV_BIN}:{LIBREOFFICE_BIN}:{os.environ.get('PATH', '')}"

load_dotenv(dotenv_path=ROOT / ".env", override=False)

from raganything import RAGAnything, RAGAnythingConfig  # noqa: E402
from lightrag.llm.openai import openai_complete_if_cache  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

LLM_HOST = os.environ["LLM_BINDING_HOST"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_API_KEY = os.environ.get("LLM_BINDING_API_KEY", "ollama")

EMBED_HOST = os.environ["EMBEDDING_BINDING_HOST"]
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]
EMBED_DIM = int(os.environ["EMBEDDING_DIM"])


async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=LLM_HOST,
        api_key=LLM_API_KEY,
        **kwargs,
    )


async def embed_func(texts):
    import ollama

    client = ollama.AsyncClient(host=EMBED_HOST)
    response = await client.embed(model=EMBED_MODEL, input=texts)
    return np.array(response.embeddings, dtype=np.float32)


async def main() -> int:
    working = ROOT / "rag_storage_baseline" / str(uuid.uuid4())
    print(f"[baseline] working_dir={working}")

    cfg = RAGAnythingConfig(
        working_dir=str(working),
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=cfg,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=embed_func,
        ),
    )

    sample_text = (
        "RAG-Anything 是一個多模態 RAG 框架，支援 PDF、Office、圖片等格式。\n"
        "底層使用 LightRAG 建構知識圖譜，並由 MinerU 負責 PDF 解析。\n"
        "本機可用 Ollama 跑 LLM 與 Embedding，達成完全本機部署。\n"
        "Bge-m3 是一個多語言 embedding 模型，向量維度為 1024。"
    )

    # 已知 issue: raganything.insert_content_list 在呼叫 lightrag.ainsert 之前
    # 會先 _upsert_doc_status，導致 LightRAG 1.4.16 的 filter_keys 把該 doc_id
    # 排除，entity extraction 不會跑。baseline 直接打 lightrag 確認鏈路通。
    await rag._ensure_lightrag_initialized()
    print("[baseline] direct lightrag.ainsert ...")
    await rag.lightrag.ainsert(
        input=sample_text,
        ids=f"baseline-{uuid.uuid4()}",
        file_paths="baseline_demo.txt",
    )
    print("[baseline] insert OK")

    questions = [
        "RAG-Anything 底層用什麼建構知識圖譜？",
        "bge-m3 的 embedding 維度是多少？",
    ]
    failures = 0
    for q in questions:
        print(f"\n[baseline] query: {q}")
        ans = await rag.aquery(q, mode="hybrid")
        if not ans:
            print("  ✗ EMPTY ANSWER")
            failures += 1
            continue
        print(f"  answer: {ans[:300]}")

    await rag.finalize_storages()
    return failures


if __name__ == "__main__":
    rc = asyncio.run(main())
    if rc == 0:
        print("\n[baseline] ALL PASS — Phase 0 完成")
    else:
        print(f"\n[baseline] {rc} 個查詢失敗")
    raise SystemExit(rc)
