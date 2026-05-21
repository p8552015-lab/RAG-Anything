import math
from types import SimpleNamespace

import numpy as np
import pytest

from raganything.mcp.providers import (
    _embed_with_ollama_fallback,
    _normalize_embedding_text,
)


class FakeOllamaClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def embed(self, model, input):
        self.calls.append((model, list(input)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(embeddings=response)


def test_normalize_embedding_text_converts_non_finite_numbers_to_text():
    assert _normalize_embedding_text(math.nan) == ""
    assert _normalize_embedding_text(float("inf")) == ""
    assert _normalize_embedding_text({"name": "資安業者"}) == '{"name": "資安業者"}'


@pytest.mark.asyncio
async def test_embed_with_ollama_fallback_splits_batch_after_nan_500():
    client = FakeOllamaClient(
        [
            RuntimeError("Ollama embed failed with json: unsupported value: NaN"),
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ]
    )

    result = await _embed_with_ollama_fallback(
        client=client,
        model="bge-m3:latest",
        texts=["資安業者", "資安經費"],
        expected_dim=2,
    )

    assert result.dtype == np.float32
    assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert client.calls == [
        ("bge-m3:latest", ["資安業者", "資安經費"]),
        ("bge-m3:latest", ["資安業者"]),
        ("bge-m3:latest", ["資安經費"]),
    ]


@pytest.mark.asyncio
async def test_embed_with_ollama_fallback_retries_non_finite_single_response():
    client = FakeOllamaClient(
        [
            [[math.nan, 1.0]],
            [[0.25, 0.75]],
        ]
    )

    result = await _embed_with_ollama_fallback(
        client=client,
        model="bge-m3:latest",
        texts=["bad vector"],
        expected_dim=2,
    )

    assert result.tolist() == [[0.25, 0.75]]
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_embed_with_ollama_fallback_strips_punctuation_for_single_nan_500():
    client = FakeOllamaClient(
        [
            RuntimeError("failed to encode response: json: unsupported value: NaN"),
            RuntimeError("failed to encode response: json: unsupported value: NaN"),
            [[0.5, 0.5]],
        ]
    )

    text = "資通安全通識教育訓練 这是公司内部資訊人力必须获取的，用于提升安全意识的强制性培训。"
    result = await _embed_with_ollama_fallback(
        client=client,
        model="bge-m3:latest",
        texts=[text],
        expected_dim=2,
    )

    assert result.tolist() == [[0.5, 0.5]]
    assert client.calls[-1] == (
        "bge-m3:latest",
        ["資通安全通識教育訓練 这是公司内部資訊人力必须获取的 用于提升安全意识的强制性培训"],
    )


@pytest.mark.asyncio
async def test_embed_with_ollama_fallback_uses_first_token_for_entity_description_nan():
    client = FakeOllamaClient(
        [
            RuntimeError("failed to encode response: json: unsupported value: NaN"),
            RuntimeError("failed to encode response: json: unsupported value: NaN"),
            RuntimeError("failed to encode response: json: unsupported value: NaN"),
            [[0.75, 0.25]],
        ]
    )

    text = "業務持續運作計畫(BCP) 指組織需要建立的計畫，用於維持關鍵業務在危機期間的連續運作。"
    result = await _embed_with_ollama_fallback(
        client=client,
        model="bge-m3:latest",
        texts=[text],
        expected_dim=2,
    )

    assert result.tolist() == [[0.75, 0.25]]
    assert client.calls[-1] == ("bge-m3:latest", ["業務持續運作計畫(BCP)"])
