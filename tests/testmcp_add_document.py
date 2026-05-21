from pathlib import Path

import pytest

from raganything.mcp.tools.add_document import _perform_add


class FakeDocStatus:
    def __init__(self):
        self.records = {}

    async def get_by_id(self, doc_id):
        return self.records.get(doc_id)


class FakeLightRAG:
    def __init__(self):
        self.doc_status = FakeDocStatus()
        self.multimodal_called = False

    async def ainsert(self, **kwargs):
        doc_id = kwargs["ids"]
        self.doc_status.records[doc_id] = {
            "status": "failed",
            "error_msg": "Ollama embed failed with json: unsupported value: NaN",
            "chunks_count": 2,
            "chunks_list": ["chunk-a", "chunk-b"],
        }


class FakeRAG:
    def __init__(self):
        self.config = type(
            "Config",
            (),
            {
                "parse_method": "auto",
                "parser_output_dir": "/tmp",
                "parser": "mineru-remote",
                "content_format": "content_list",
            },
        )()
        self.lightrag = FakeLightRAG()

    async def parse_document(self, *args, **kwargs):
        return ([{"type": "text", "text": "資安業者 資安經費"}], "doc-test")

    def _get_file_reference(self, path):
        return Path(path).name

    def set_content_source_for_context(self, *args, **kwargs):
        raise AssertionError("no multimodal content should be processed")

    async def _process_multimodal_content(self, *args, **kwargs):
        self.lightrag.multimodal_called = True

    async def _mark_multimodal_processing_complete(self, doc_id):
        raise AssertionError("failed text ingestion should stop before completion")


@pytest.mark.asyncio
async def test_perform_add_fails_fast_when_text_insert_marks_doc_failed(tmp_path):
    fake_pdf = tmp_path / "security.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")
    rag = FakeRAG()

    with pytest.raises(RuntimeError, match="Ollama embed failed"):
        await _perform_add(
            rag=rag,
            path=fake_pdf,
            parse_method=None,
            doc_id=None,
            split_by_character=None,
            split_by_character_only=False,
        )

    assert rag.lightrag.multimodal_called is False
