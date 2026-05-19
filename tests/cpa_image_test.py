import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelle_video.services import media as media_module
from pixelle_video.services.cpa_image import (
    build_cpa_image_payload,
    choose_cpa_image_size,
    extract_image_b64,
)
from pixelle_video.services.media import MediaService


def test_build_cpa_image_payload_uses_image_generation_tool():
    payload = build_cpa_image_payload(
        "Hong Kong company registration concept art",
        main_model="gpt-5.4",
        image_model="gpt-image-2",
        size="1024x1536",
    )

    assert payload["model"] == "gpt-5.4"
    assert payload["input"] == "Hong Kong company registration concept art"
    assert payload["tool_choice"] == {"type": "image_generation"}
    assert payload["tools"] == [
        {
            "type": "image_generation",
            "action": "generate",
            "model": "gpt-image-2",
            "size": "1024x1536",
        }
    ]


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
