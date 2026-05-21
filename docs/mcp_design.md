# RAG-Anything MCP Server 設計文件

> 版本：草案 v1（2026-05-20）
> 對應 fork：`p8552015-lab/RAG-Anything`
> 目標環境：macOS（darwin 24.3.0）+ Python 3.10+ + Ollama / LM Studio 全本機路線

本文件是「把 RAG-Anything 包成 MCP Server」這個改造任務的單一參考來源。所有實作細節在動工前都應該以這份文件為準；任何偏離都要回頭更新本文件。

---

## 1. 目標與非目標

### 目標
- 在 fork 內新增 `raganything/mcp/` 子模組，提供一個可被 Claude Desktop / Cursor / Claude Code 直接 spawn 的 stdio MCP server。
- 對外提供 7 個 MCP tool，覆蓋「加入文件 / 刪除文件 / 查詢 / 多模態查詢 / 列出文件 / 抽取結構化內容 / 取得狀態」七大行為。
- 全部走本機 Ollama / LM Studio，免任何雲端 API key 即可啟動。
- 不修改 upstream 的核心 Python 檔案（`raganything/raganything.py`、`query.py`、`processor.py` 等），確保未來同步 HKUDS 主線時不衝突。

### 非目標
- 不提供 HTTP / SSE / WebSocket transport（僅 stdio；如未來要 web 化再分開做一份）。
- 不做多租戶（同一個 working_dir 同一時間只服務一個 MCP client）。
- 不做 RAG-Anything 既有功能的擴充（例如新增 parser、新增 modal processor 不在此範圍）。
- 不做 UI / Web Console。
- 不打包成 Docker（後續再評估）。

---

## 2. 整體架構

```
┌────────────────────────────────┐
│ MCP Client                     │
│ (Claude Desktop / Cursor / CC) │
└──────────────┬─────────────────┘
               │ JSON-RPC over stdio
               ▼
┌────────────────────────────────────────┐
│ raganything-mcp (console_script)       │
│ ├─ server.py    FastMCP entry          │
│ ├─ rag_instance lazy async singleton   │
│ ├─ schemas.py   Pydantic 入參/出參     │
│ └─ tools/*.py   7 個 tool 實作         │
└──────────────┬─────────────────────────┘
               │ 直接 import & 呼叫 (DRY)
               ▼
┌────────────────────────────────────────┐
│ RAGAnything (既有套件，不動)           │
│ ├─ ProcessorMixin.process_document_*   │
│ ├─ QueryMixin.aquery / multimodal      │
│ ├─ BatchMixin                          │
│ └─ parser / modalprocessors            │
└──────────────┬─────────────────────────┘
               ▼
┌────────────────────────────────────────┐
│ LightRAG + storage (file-based)        │
│ working_dir/                           │
│   ├─ kv_store_*.json                   │
│   ├─ vdb_*.json                        │
│   ├─ graph_chunk_entity_relation.*     │
│   └─ doc_status.json                   │
└────────────────────────────────────────┘

外部依賴：
  - Ollama  (LLM + Embedding，本機 :11434)
  - LM Studio (可選，Vision 模型走這條)
  - mineru[core] (PDF 解析，首次自動下載模型)
  - LibreOffice (Office 檔解析)
```

---

## 3. 模組與檔案配置

新增檔案（全部位於 fork 內）：

```
RAG-Anything/
├─ raganything/
│  └─ mcp/                          ← 全新子模組
│     ├─ __init__.py                 export server.main 給 entry point 用
│     ├─ server.py                   FastMCP 註冊 + 啟動 main()
│     ├─ rag_instance.py             RAGAnything 單例 + lazy async init
│     ├─ schemas.py                  全部 tool 的 Pydantic 模型
│     ├─ providers.py                Ollama/LM Studio 的 llm/vision/embedding 工廠
│     └─ tools/
│        ├─ __init__.py              註冊函式 register_all(mcp)
│        ├─ add_document.py
│        ├─ delete_document.py
│        ├─ query.py                 含 rag_query + rag_query_multimodal
│        ├─ list_documents.py
│        ├─ extract_content.py
│        └─ status.py
├─ examples/mcp/
│  ├─ claude_desktop_config.json     參考設定
│  ├─ cursor_config.json             參考設定
│  └─ test_with_mcp_inspector.md     本地測試指引
└─ pyproject.toml                    新增 [project.optional-dependencies].mcp
                                     新增 [project.scripts] raganything-mcp
```

修改的既有檔案僅限：
- `pyproject.toml`（加 optional deps 與 entry point）
- 不動 `raganything/*.py`、不動 `requirements.txt`、不動 `setup.py`、不動 `README.md`（最後 Phase 4 才碰）

---

## 4. 環境變數規範

全部從 env 讀；**任一必要變數缺失 → server 啟動時 fail-fast，禁止用任何 placeholder / 預設值替代**。沿用 RAG-Anything 既有的 `env.example` 命名，避免雙重定義。

### 必要
| 變數 | 範例值 | 說明 |
|---|---|---|
| `WORKING_DIR` | `/path/to/RAG_MCP/rag_storage` | LightRAG storage 路徑 |
| `LLM_BINDING` | `ollama` | 固定 `ollama` 或 `lmstudio` 或 `openai` |
| `LLM_BINDING_HOST` | `http://localhost:11434/v1` | OpenAI 相容 endpoint |
| `LLM_MODEL` | `qwen2.5:7b` | Chat 模型 tag |
| `EMBEDDING_BINDING` | `ollama` | 固定 `ollama` |
| `EMBEDDING_MODEL` | `bge-m3:latest` | Embedding 模型 tag |
| `EMBEDDING_DIM` | `1024` | bge-m3=1024、nomic-embed-text=768 |
| `EMBEDDING_BINDING_HOST` | `http://localhost:11434` | Ollama 原生 host（**不要加 /v1**） |

### 可選
| 變數 | 預設 | 說明 |
|---|---|---|
| `VISION_MODEL` | 未設 | 設了才啟用 VLM enhanced query；建議 `llama3.2-vision:11b` 或 `qwen2-vl:7b` |
| `VISION_BINDING_HOST` | `http://localhost:11434/v1` | Vision endpoint，可指向 LM Studio |
| `PARSER` | `mineru` | `mineru` / `docling` / `paddleocr` |
| `PARSE_METHOD` | `auto` | `auto` / `ocr` / `txt` |
| `ENABLE_IMAGE_PROCESSING` | `true` | 沒設 VISION_MODEL 時建議 `false` 加速 |
| `MAX_ASYNC` | `4` | LLM 並發 |
| `TIMEOUT` | `240` | LLM 單次呼叫秒數 |
| `LOG_LEVEL` | `INFO` | MCP server 自己的 log |

### 嚴禁出現
- 任何 hard-coded model name 寫在 `.py` 裡（連預設值都不行，所有預設都從 `lightrag.utils.get_env_value` 取）
- 任何 fallback API key（如 `"ollama"` 字串），雖然 Ollama 不檢查，但程式碼裡看到固定字串會引導使用者誤以為可省略
- 任何「Embedding 失敗就用 hash」「LLM 失敗就回 'OK'」這類降級

---

## 5. MCP Tools 規格

所有 tool 命名前綴 `rag_`，避免與其他 MCP server 衝突。入參與出參都用 Pydantic v2 模型在 `schemas.py` 統一定義。

### 5.1 `rag_add_document`

| 屬性 | 值 |
|---|---|
| 底層 | `RAGAnything.process_document_complete` (processor.py:1654) |
| 同步/非同步 | tool 內部 await，外部看是同步阻塞 |
| 預期耗時 | 文字 PDF: 30s–2min；含圖 PDF: 2–10min；Office: 5–30s |

**入參**
```python
class AddDocumentInput(BaseModel):
    file_path: str = Field(..., description="絕對路徑或相對於 WORKING_DIR")
    parse_method: Literal["auto", "ocr", "txt"] | None = None
    doc_id: str | None = Field(None, description="未指定則用內容 hash 自動產生")
    split_by_character: str | None = None
    split_by_character_only: bool = False
```

**出參**
```python
class AddDocumentOutput(BaseModel):
    doc_id: str
    file_name: str
    chunk_count: int
    entity_count: int
    relation_count: int
    elapsed_seconds: float
    parser_used: str
```

**錯誤行為**
- 檔案不存在 → `FileNotFoundError`（讓 MCP 自然回給 client）
- 副檔名不支援 → `ValueError`，訊息列出 `supported_file_extensions`
- mineru 模型未下載 → 不攔截，由 mineru 自己抛；server 在 log 顯示提示

### 5.2 `rag_delete_document`

| 屬性 | 值 |
|---|---|
| 底層 | `lightrag.adelete_by_doc_id(doc_id)` |

```python
class DeleteDocumentInput(BaseModel):
    doc_id: str

class DeleteDocumentOutput(BaseModel):
    doc_id: str
    deleted: bool
    removed_chunks: int
```

### 5.3 `rag_query`

| 屬性 | 值 |
|---|---|
| 底層 | `QueryMixin.aquery` (query.py:102) |

```python
class QueryInput(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    top_k: int | None = None
    vlm_enhanced: bool | None = None  # None = 依 vision_model_func 是否存在自動判斷
    system_prompt: str | None = None

class QueryOutput(BaseModel):
    answer: str
    mode: str
    elapsed_seconds: float
```

備註：`mode` 預設沿用 RAG-Anything 的 `"mix"`，與 README 示意的 `"hybrid"` 不同；以原始碼為準。

### 5.4 `rag_query_multimodal`

| 屬性 | 值 |
|---|---|
| 底層 | `QueryMixin.aquery_with_multimodal` (query.py:195) |

```python
class MultimodalItem(BaseModel):
    type: Literal["image", "table", "equation"]
    img_path: str | None = None
    table_data: str | None = None
    table_caption: str | None = None
    latex: str | None = None
    equation_caption: str | None = None

class QueryMultimodalInput(BaseModel):
    query: str
    multimodal_content: list[MultimodalItem]
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    system_prompt: str | None = None
```

出參同 `QueryOutput`。如果 `multimodal_content` 為空，底層會 fallback 到 `aquery`，這個行為直接透傳，不額外攔截。

### 5.5 `rag_list_documents`

| 屬性 | 值 |
|---|---|
| 底層 | `lightrag.doc_status.get_docs_by_status(...)` 直讀 |

```python
class ListDocumentsInput(BaseModel):
    status_filter: Literal["processed", "processing", "failed", "all"] = "all"
    limit: int = 100
    offset: int = 0

class DocumentEntry(BaseModel):
    doc_id: str
    file_name: str
    status: str
    created_at: str
    updated_at: str
    chunk_count: int | None = None

class ListDocumentsOutput(BaseModel):
    total: int
    documents: list[DocumentEntry]
```

### 5.6 `rag_extract_content`

| 屬性 | 值 |
|---|---|
| 底層 | `RAGAnything.parse_document` 或 `doc_parser.parse_*`，**不**呼叫 `process_document_complete`（不入庫） |
| 用途 | 純抽取 PDF/Office 內的表格、圖片、公式給 Claude 直接看，不污染知識庫 |

```python
class ExtractContentInput(BaseModel):
    file_path: str
    types: list[Literal["text", "table", "image", "equation"]] = ["table", "image", "equation"]
    parse_method: Literal["auto", "ocr", "txt"] | None = None

class ExtractedItem(BaseModel):
    type: str
    page_idx: int | None = None
    content: dict  # 結構依 type 而定，遵循 RAG-Anything 既有 content_list 格式

class ExtractContentOutput(BaseModel):
    file_path: str
    parser_used: str
    items: list[ExtractedItem]
    counts: dict[str, int]  # {"text": 12, "table": 3, "image": 5, "equation": 2}
```

### 5.7 `rag_get_status`

| 屬性 | 值 |
|---|---|
| 底層 | 自己讀 config + LightRAG storage 統計 |

```python
class StatusOutput(BaseModel):
    working_dir: str
    parser: str
    llm_model: str
    embedding_model: str
    embedding_dim: int
    vision_model: str | None
    document_count: int
    entity_count: int
    relation_count: int
    initialized: bool
    raganything_version: str
```

---

## 6. 全本機 Ollama 路線配置

### 必裝模型清單
```bash
# Chat（擇一）
ollama pull qwen2.5:7b           # 8GB RAM 可跑，中文友善（推薦）
ollama pull llama3.1:8b          # 通用
ollama pull mistral:7b           # 較輕

# Embedding
ollama pull bge-m3:latest        # 1024-dim，多語言（推薦中文文件）
# 或
ollama pull nomic-embed-text     # 768-dim，純英文較快

# Vision（可選；想啟用圖片解析才裝）
ollama pull llama3.2-vision:11b  # 8GB+ VRAM
# 或走 LM Studio 載 qwen2-vl
```

### `providers.py` 必須做的事

1. **LLM**：用 `lightrag.llm.openai.openai_complete_if_cache` 透過 Ollama 的 OpenAI 相容 `/v1` endpoint
2. **Embedding**：用 `ollama` Python client 的 `AsyncClient.embed` 走原生 `/api/embed`（**不**走 `/v1/embeddings`，部分模型在該 endpoint 會失敗）。**必須 `return np.array(response.embeddings, dtype=np.float32)`，不可 return list**（見 R9）
3. **Vision**：
   - 若 `VISION_MODEL` 未設 → `vision_model_func=None`，VLM enhanced query 自動停用
   - 若設了 → 也用 OpenAI 相容介面包成 `vision_model_func(prompt, image_data=..., messages=...)`，簽章對齊 `examples/raganything_example.py:135-187`

### 啟動前自檢（fail-fast）
`rag_instance.ensure_initialized()` 第一次被呼叫時要做：
1. `ollama.AsyncClient(host=...).list()` 確認 Ollama 連得上
2. 對 `LLM_MODEL`、`EMBEDDING_MODEL`、（若有）`VISION_MODEL` 做 prefix match，缺哪個就 raise，訊息包含 `ollama pull <model>` 指令
3. 跑一次 1-token embedding，驗證實際維度 == `EMBEDDING_DIM`，不符即 raise（這個誤配常常導致 vector store 寫入後查不到，要在第一時間擋掉）

---

## 7. 啟動 / 關閉生命週期

### 啟動
```
Claude Desktop spawn raganything-mcp
       │
       ▼
1. server.py main()
       │ load_dotenv(.env, override=False)
       │ 從 env 讀 LOG_LEVEL / WORKING_DIR
       ▼
2. FastMCP("rag-anything") 建立實例
       │ 註冊所有 tool（透過 tools/__init__.py:register_all(mcp)）
       ▼
3. mcp.run(transport="stdio")
       │ 此時尚未建構 RAGAnything（lazy）
       ▼
4. Client 呼叫第一個 tool → rag_instance.ensure_initialized()
       │ providers.build_llm_func / build_embedding_func / build_vision_func
       │ 自檢 Ollama + 模型 + embedding dim
       │ 建構 RAGAnything(config, llm_model_func, vision_model_func, embedding_func)
       │ 第一次 process_document_complete 才觸發 LightRAG storage init
       ▼
5. 後續 tool 呼叫直接重用同一個 RAGAnything 實例
```

### 關閉
- 註冊 `signal.SIGTERM` / `SIGINT` handler，呼叫 `asyncio.run(rag.finalize_storages())`
- **不**依賴 `atexit`（RAGAnything `close()` 有 atexit 但在 stdio MCP 場景下事件迴圈狀態詭異，已知會 silently swallow）

---

## 8. 錯誤處理與「禁止降級」原則

依照 `~/.claude/CLAUDE.md` 規範，本專案嚴格遵守：

| 違規模式 | 為什麼禁止 | 正確做法 |
|---|---|---|
| Embedding 失敗 → 用 hash 假裝 | 之後查詢全錯，但系統不會自己發現 | raise，讓 add_document 整個失敗 |
| LLM 超時 → 回 `"處理中..."` | Claude 不知道沒答到 | raise `TimeoutError`，client 看到真實錯誤 |
| Vision model 缺 → 用 LLM 看 `[image]` placeholder | 多模態功能名存實亡 | `vlm_enhanced=False`，並在 status tool 明確 false |
| 任何 tool 內部 `try/except: pass` | 隱藏問題 | 只在「真的可以忽略」處攔，且 log warning |

所有抛出的例外都讓 MCP 框架自然轉成 `isError=true` 的 tool response，client 端會收到完整 traceback。

---

## 9. 已知風險與緩解

| # | 風險 | 緩解 |
|---|---|---|
| R1 | mineru 首次跑下載 ~3GB 模型，超過 Claude Desktop 預設 60s tool timeout | 提供 `python -m raganything.mcp.warmup` CLI，請使用者在安裝後手動跑一次；README 標註 |
| R2 | Ollama bge-m3 embedding 慢（CPU 約 0.5s/段），大 PDF 入庫要 10+ 分鐘 | 接受現狀；建議使用者 add_document 後切走做別的事，回來查 status |
| R3 | LightRAG file-based storage 無 lock，多進程 MCP 並發會寫壞 | 在 README 明確標示「同一 working_dir 同時只能一個 MCP server」 |
| R4 | 中文 PDF + mineru `auto` 偶爾誤判語言 | 暴露 `parse_method=ocr` 給使用者強制 OCR |
| R5 | Ollama vision 模型品質遠不如 GPT-4o | 不在程式碼裡掩蓋；status tool 回報 vision_model 名稱，使用者自己判斷 |
| R6 | RAGAnything `__post_init__` 會 `atexit.register(self.close)`，stdio MCP 結束時可能 double-finalize | 顯式呼叫 `finalize_storages` 並讓 `close` 內部容錯（既有實作已 try/except） |
| R7 | embedding_dim 設錯（最常見：bge-m3 用 768） | 自檢階段抓出來；錯了就 raise |
| **R8** | **RAGAnything 1.3.0 提早 upsert doc_status 與 LightRAG 1.4.16 的 `filter_keys` 衝突** —— [processor.py:2144](../raganything/processor.py) 與 [processor.py:1711](../raganything/processor.py) 在呼叫 `lightrag.ainsert` 前先 `_upsert_doc_status(status=HANDLING)`，導致 LightRAG `filter_keys` 把該 doc_id 視為已存在 → entity extraction 不跑 → query 全部 `[no-context]` | **Phase 1 必須抉擇**：（A）在 fork 內把那兩處 upsert 移到 `insert_text_content` 之後，保留 RAGAnything 完整 multimodal flow；（B）MCP wrapper `rag_add_document` 改走 `parse_document` + 直接 `lightrag.ainsert`，跳過 `process_document_complete` 整段。Phase 0 baseline 用 (B) 驗證可行（見 `examples/mcp/baseline_test.py`） |
| **R9** | **Ollama embedding 必須回 `np.ndarray`** —— `examples/ollama_integration_example.py` 直接 return list 會在 LightRAG query path 觸發 `'list' object has no attribute 'size'` | Phase 1 `providers.py` 的 `build_embedding_func` 統一 wrap `np.array(..., dtype=np.float32)`，禁止 return list |
| **R10** | **Claude Desktop spawn MCP server 不會繼承 venv shell PATH** —— mineru / soffice CLI 找不到 → parser 自檢失敗 | Phase 1 `server.py` `main()` 第一步把 `<venv>/bin` 與 `/Applications/LibreOffice.app/Contents/MacOS`（或 `LIBREOFFICE_BIN_DIR` env 指定的目錄）prepend 到 `os.environ['PATH']`，這對 `subprocess.run(["mineru", ...])` 才有效 |
| **R11** | **`RAGAnythingConfig` 的 dataclass field default 用 `field(default=get_env_value(...))`**（[config.py:30](../raganything/config.py)），default 在 module load 時 evaluate。Console_script `from raganything.mcp.server import main` 會先觸發 `raganything/__init__.py` import，這比 `server.main()` 內的 `_setup_env()` 早，導致 `.env` 載入後 `RAGAnythingConfig(working_dir=...)` 拿到的 parser/parse_method 仍是 module load 時 freeze 的舊值（通常是 `"mineru"` fallback） | [rag_instance.py](../raganything/mcp/rag_instance.py) `get_rag()` 內顯式從 `os.environ` 讀 `PARSER` / `PARSE_METHOD` / `OUTPUT_DIR` / `ENABLE_*_PROCESSING` 傳給 RAGAnythingConfig 建構子覆寫 default。violates DRY 但這是 upstream import 時序問題的最務實解法，不需 fork raganything |

---

## 10. 測試策略

### Phase 0 baseline
- 跑 `examples/ollama_integration_example.py`，確認 Ollama 配置正確
- 跑 `examples/raganything_example.py path/to/test.pdf`（要先把 LLM 改成 Ollama）

### Phase 2 MCP MVP
1. **單元層**：每個 tool 一個 `tests/mcp/test_<tool>.py`，用 pytest-asyncio + 真實 Ollama（不 mock）
2. **整合層**：用 `mcp dev raganything.mcp.server:main` 跑 MCP Inspector（官方測試工具），手動點 7 個 tool
3. **端到端**：在 Claude Desktop 設定 server，問一份真實中文 PDF（建議用一份政府計畫書）

### Phase 3 進階
- 多模態：餵一個含表格的 PDF（例如預算表），驗證 `rag_extract_content(types=["table"])` 拿到結構化資料
- 刪除回歸：add → delete → list，確認 doc_count 歸零

### 不做
- 不對 LLM/Embedding 做 mock 測試（違反禁止模擬數據原則）
- 不做 load/stress test（單 MCP server 無此需求）

---

## 11. 實作清單（Phase 0~4 對應 Todo）

### Phase 0：環境驗證（預估 30 分鐘）
- [ ] 建立 venv：`python3.10 -m venv .venv && source .venv/bin/activate`
- [ ] 安裝：`pip install -e ".[all]" "mcp[cli]" python-dotenv ollama`
- [ ] 確認 LibreOffice：`which soffice`，無則 `brew install --cask libreoffice`
- [ ] Ollama 安裝與啟動：`ollama serve` + pull 三個模型
- [ ] 從 `env.example` 複製 `.env`，改 LLM_BINDING / EMBEDDING_BINDING 為 ollama
- [ ] 跑 `examples/ollama_integration_example.py`，整個流程 pass
- [ ] 跑 `examples/raganything_example.py` 處理一份 sample PDF（小於 3 頁）

### Phase 1：Server 骨架（預估 1.5 小時，比原估多 30 分鐘因 R8）
- [ ] **R8 抉擇**：決定 `rag_add_document` 走 (A) 修 fork raganything processor.py 把 upsert 延後，或 (B) wrapper 直接呼叫 `parse_document` + `lightrag.ainsert`。Phase 0 baseline 已驗證 (B) 可行
- [ ] `raganything/mcp/__init__.py`：`from .server import main`
- [ ] `raganything/mcp/schemas.py`：列入所有 Pydantic 模型（先寫殼）
- [ ] `raganything/mcp/providers.py`：`build_llm_func` / `build_embedding_func` / `build_vision_func`，**embedding 必須 wrap `np.array(..., dtype=np.float32)`**（R9）
- [ ] `raganything/mcp/rag_instance.py`：`get_rag()` async 函式 + 啟動自檢（Ollama 連線 / 模型存在 / 實測 embedding dim == EMBEDDING_DIM）
- [ ] `raganything/mcp/server.py`：`main()` 第一步補 PATH（R10：venv/bin + LIBREOFFICE_BIN_DIR），建 FastMCP、註冊 `rag_get_status` 一個 tool、`mcp.run("stdio")`
- [ ] `pyproject.toml`：加 `[project.optional-dependencies].mcp = ["mcp[cli]>=1.0", "ollama>=0.6", "python-dotenv", "numpy"]` 與 `[project.scripts] raganything-mcp = "raganything.mcp.server:main"`
- [ ] `pip install -e ".[mcp]"`
- [ ] `mcp dev raganything-mcp` 在 Inspector 看到 `rag_get_status`

### Phase 2：MVP tools（預估 2 小時）
- [ ] `tools/add_document.py`
- [ ] `tools/query.py`（只先做 `rag_query`）
- [ ] `tools/list_documents.py`
- [ ] `tools/__init__.py`：`register_all(mcp)` 把上面三個註冊
- [ ] 用 Inspector 對一份小 PDF 跑 add → list → query

### Phase 3：進階 tools（預估 2 小時）
- [ ] `tools/delete_document.py`
- [ ] `tools/extract_content.py`
- [ ] `tools/query.py` 補 `rag_query_multimodal`
- [ ] `tools/status.py` 補完 `rag_get_status`
- [ ] 跑表格抽取 + 多模態查詢端到端

### Phase 4：對外接入（預估 30 分鐘）
- [ ] `examples/mcp/claude_desktop_config.json`
- [ ] `examples/mcp/cursor_config.json`
- [ ] `examples/mcp/test_with_mcp_inspector.md`
- [ ] 在 Claude Desktop 實接，問一份政府計畫書
- [ ] README 補一個「### MCP Server」短章節，指向 `docs/mcp_design.md`

---

## 附錄 A：Claude Desktop 接入範例

```json
{
  "mcpServers": {
    "rag-anything": {
      "command": "/path/to/RAG_MCP/RAG-Anything/.venv/bin/raganything-mcp",
      "env": {
        "WORKING_DIR": "/path/to/RAG_MCP/rag_storage",
        "LLM_BINDING": "ollama",
        "LLM_BINDING_HOST": "http://localhost:11434/v1",
        "LLM_MODEL": "qwen2.5:7b",
        "EMBEDDING_BINDING": "ollama",
        "EMBEDDING_MODEL": "bge-m3:latest",
        "EMBEDDING_DIM": "1024",
        "EMBEDDING_BINDING_HOST": "http://localhost:11434",
        "VISION_MODEL": "llama3.2-vision:11b",
        "VISION_BINDING_HOST": "http://localhost:11434/v1",
        "PARSER": "mineru",
        "PARSE_METHOD": "auto",
        "MAX_ASYNC": "2",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## 附錄 B：底層 API 對應表

| MCP Tool | 底層方法 | 檔案行號 |
|---|---|---|
| `rag_add_document` | `ProcessorMixin.process_document_complete` | `raganything/processor.py:1654` |
| `rag_delete_document` | `lightrag.adelete_by_doc_id` | (LightRAG) |
| `rag_query` | `QueryMixin.aquery` | `raganything/query.py:102` |
| `rag_query_multimodal` | `QueryMixin.aquery_with_multimodal` | `raganything/query.py:195` |
| `rag_list_documents` | `lightrag.doc_status.get_docs_by_status` | (LightRAG) |
| `rag_extract_content` | `RAGAnything.parse_document` + `doc_parser` | `raganything/processor.py` |
| `rag_get_status` | `self.config` + `lightrag` storage stats | n/a |

---

## 修訂歷史

| 日期 | 版本 | 變更 |
|---|---|---|
| 2026-05-20 | v1 | 初稿 |
| 2026-05-20 | v1.1 | Phase 0 baseline 完成；補 R8（_upsert_doc_status regression）、R9（numpy embed）、R10（PATH 補丁）三項風險；Phase 1 清單加 R8 抉擇與 numpy/PATH 對策 |
| 2026-05-20 | v1.2 | Phase 1 骨架 + rag_get_status 驗證通過；Phase 2/3 共 7 個 tool 程式碼完成（B 方案）；Phase 4 README MCP 章節 + Claude Desktop config 範例完成；剩 Phase 2/3 端到端驗證 |
