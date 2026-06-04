import asyncio as asyncio_module
import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelle_video.services import cpa_image as cpa_image_module
from pixelle_video.services import media as media_module
from pixelle_video.services.cpa_image import (
    build_cpa_image_payload,
    choose_cpa_image_size,
    extract_image_b64,
    generate_cpa_image,
)
from pixelle_video.services.media import MediaService


def test_build_cpa_image_payload_uses_images_endpoint_shape():
    payload = build_cpa_image_payload(
        "Hong Kong company registration concept art",
        main_model="gpt-5.4",
        image_model="gpt-image-2",
        size="1024x1536",
    )

    assert payload == {
        "model": "gpt-image-2",
        "prompt": "Hong Kong company registration concept art",
        "size": "1024x1536",
        "response_format": "b64_json",
    }


def test_extract_image_b64_from_images_api_result():
    expected = base64.b64encode(b"fake-png").decode()
    response = {"data": [{"b64_json": expected}]}

    assert extract_image_b64(response) == expected


def test_extract_image_b64_from_image_generation_result():
    expected = base64.b64encode(b"fake-png").decode()
    response = {
        "output": [
            {
                "type": "image_generation_call",
                "result": expected,
            }
        ]
    }

    assert extract_image_b64(response) == expected


def test_extract_image_b64_rejects_missing_result():
    with pytest.raises(RuntimeError, match="did not contain an image result"):
        extract_image_b64({"output": [{"type": "message", "content": []}]})


def test_choose_cpa_image_size_matches_aspect_ratio():
    assert choose_cpa_image_size(1080, 1920) == "1024x1536"
    assert choose_cpa_image_size(1920, 1080) == "1536x1024"
    assert choose_cpa_image_size(1024, 1024) == "1024x1024"
    assert choose_cpa_image_size(None, 1024) == "1024x1024"


def test_media_service_lists_local_cpa_workflow():
    service = MediaService({"comfyui": {"image": {"default_workflow": "cpa/gpt-image-2"}}})

    workflows = service.list_workflows()

    assert any(
        workflow["key"] == "cpa/gpt-image-2"
        and workflow["display_name"] == "gpt-image-2 - Local CPA"
        for workflow in workflows
    )


async def test_media_service_passes_configured_cpa_models(monkeypatch, tmp_path):
    output_path = tmp_path / "image.png"
    captured = {}

    from pixelle_video.config import config_manager

    monkeypatch.setattr(
        config_manager,
        "get_llm_config",
        lambda: {
            "api_key": "local-test-key",
            "base_url": "http://127.0.0.1:8317/v1",
            "model": "gpt-5.4-mini",
        },
    )

    async def fake_generate_cpa_image(**kwargs):
        captured.update(kwargs)
        return str(output_path)

    monkeypatch.setattr(media_module, "generate_cpa_image", fake_generate_cpa_image)

    service = MediaService({"comfyui": {"image": {"default_workflow": "cpa/gpt-image-2"}}})
    result = await service(
        prompt="Hong Kong CPA office illustration",
        workflow="cpa/gpt-image-2",
        media_type="image",
        width=1080,
        height=1920,
        output_path=str(output_path),
    )

    assert result.media_type == "image"
    assert result.url == str(output_path)
    assert captured["api_key"] == "local-test-key"
    assert captured["base_url"] == "http://127.0.0.1:8317/v1"
    assert captured["main_model"] == "gpt-5.4-mini"
    assert captured["image_model"] == "gpt-image-2"
    assert captured["size"] == "1024x1536"
    assert captured["timeout"] == 300.0


async def test_generate_cpa_image_retries_slow_image_route(monkeypatch, tmp_path):
    calls = []
    expected_b64 = base64.b64encode(b"fake-png").decode()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"b64_json": expected_b64}]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, json, headers):
            calls.append({"endpoint": endpoint, "timeout": self.timeout, "payload": json})
            if len(calls) == 1:
                raise cpa_image_module.httpx.ReadTimeout("image route was still rendering")
            return FakeResponse()

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(cpa_image_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(asyncio_module, "sleep", fake_sleep)

    out = tmp_path / "image.png"
    result = await generate_cpa_image(
        prompt="Hong Kong office tower",
        output_path=str(out),
        api_key="local-test-key",
    )

    assert result == str(out.resolve())
    assert out.read_bytes() == b"fake-png"
    assert len(calls) == 2
    assert calls[0]["timeout"] == 300.0
    assert calls[0]["payload"]["model"] == "gpt-image-2"


async def test_generate_cpa_image_retries_transient_server_errors(monkeypatch, tmp_path):
    calls = []
    expected_b64 = base64.b64encode(b"fake-png").decode()

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = "temporary upstream failure"

        def raise_for_status(self):
            if self.status_code >= 400:
                request = cpa_image_module.httpx.Request(
                    "POST",
                    "http://127.0.0.1:8317/v1/images/generations",
                )
                response = cpa_image_module.httpx.Response(
                    self.status_code,
                    text=self.text,
                    request=request,
                )
                raise cpa_image_module.httpx.HTTPStatusError(
                    "server error",
                    request=request,
                    response=response,
                )

        def json(self):
            return {"data": [{"b64_json": expected_b64}]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, json, headers):
            calls.append({"endpoint": endpoint, "timeout": self.timeout, "payload": json})
            return FakeResponse(500 if len(calls) == 1 else 200)

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(cpa_image_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(asyncio_module, "sleep", fake_sleep)

    out = tmp_path / "image.png"
    result = await generate_cpa_image(
        prompt="Hong Kong office tower",
        output_path=str(out),
        api_key="local-test-key",
    )

    assert result == str(out.resolve())
    assert out.read_bytes() == b"fake-png"
    assert len(calls) == 2


async def test_generate_cpa_image_does_not_retry_auth_errors(monkeypatch, tmp_path):
    calls = []

    class FakeResponse:
        status_code = 401
        text = '{"error":"auth_unavailable"}'

        def raise_for_status(self):
            request = cpa_image_module.httpx.Request(
                "POST",
                "http://127.0.0.1:8317/v1/images/generations",
            )
            response = cpa_image_module.httpx.Response(
                self.status_code,
                text=self.text,
                request=request,
            )
            raise cpa_image_module.httpx.HTTPStatusError(
                "auth error",
                request=request,
                response=response,
            )

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, json, headers):
            calls.append({"endpoint": endpoint, "timeout": self.timeout, "payload": json})
            return FakeResponse()

    monkeypatch.setattr(cpa_image_module.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        await generate_cpa_image(
            prompt="Hong Kong office tower",
            output_path=str(tmp_path / "image.png"),
            api_key="local-test-key",
        )

    assert len(calls) == 1
