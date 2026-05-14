import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preflight import check_config, format_results


def test_check_config_masks_keys_and_warns_for_missing_media_keys(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config = {
        "llm": {
            "api_key": "dummy-test-key",
            "base_url": "http://127.0.0.1:8317/v1",
            "model": "gemini-3-flash",
        },
        "comfyui": {
            "comfyui_url": "http://127.0.0.1:8188",
            "runninghub_api_key": None,
            "comfyui_api_key": None,
            "tts": {
                "inference_mode": "local",
                "local": {"voice": "zh-CN-XiaoxiaoNeural", "speed": 1.1},
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    results = check_config(config, config_path)
    rendered = format_results(results)

    assert "dummy-test-key" not in rendered
    assert "api_key=<set>" in rendered
    assert "model=gemini-3-flash" in rendered
    assert "Edge-TTS config: local mode" in rendered
    assert "RunningHub workflows cannot generate images/videos yet" in rendered
    assert "selfhost workflows need a running ComfyUI" in rendered


def test_check_config_fails_when_llm_required_fields_are_missing(tmp_path: Path):
    config_path = tmp_path / "config.yaml"

    results = check_config({"llm": {}, "comfyui": {}}, config_path)
    rendered = format_results(results)

    assert any(result.name == "llm config" and result.status == "fail" for result in results)
    assert "missing api_key, base_url, or model" in rendered
