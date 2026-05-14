#!/usr/bin/env python3
"""Offline preflight checks for a local Pixelle-Video setup."""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def _has_value(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def is_tcp_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_python_version() -> CheckResult:
    version = sys.version_info
    current = f"{version.major}.{version.minor}.{version.micro}"
    if version >= (3, 11):
        return CheckResult("python", "ok", f"Python {current}")
    return CheckResult("python", "fail", f"Python {current}; requires >= 3.11")


def check_command(name: str) -> CheckResult:
    path = shutil.which(name)
    if path:
        return CheckResult(name, "ok", path)
    return CheckResult(name, "fail", f"{name} not found in PATH")


def check_config(config: dict[str, Any], config_path: Path) -> list[CheckResult]:
    if not config:
        return [
            CheckResult(
                "config.yaml",
                "fail",
                f"{config_path} not found or empty; copy config.example.yaml and fill LLM fields",
            )
        ]

    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    comfyui = config.get("comfyui") if isinstance(config.get("comfyui"), dict) else {}
    tts = comfyui.get("tts") if isinstance(comfyui.get("tts"), dict) else {}
    local_tts = tts.get("local") if isinstance(tts.get("local"), dict) else {}

    results: list[CheckResult] = []
    if all(_has_value(llm.get(key)) for key in ("api_key", "base_url", "model")):
        results.append(
            CheckResult(
                "llm config",
                "ok",
                f"model={llm.get('model')} base_url={llm.get('base_url')} api_key=<set>",
            )
        )
    else:
        results.append(
            CheckResult(
                "llm config",
                "fail",
                "missing api_key, base_url, or model; fill these in System Configuration",
            )
        )

    mode = tts.get("inference_mode", "local")
    voice = local_tts.get("voice", "zh-CN-YunjianNeural")
    if mode == "local":
        results.append(CheckResult("Edge-TTS config", "ok", f"local mode, voice={voice}"))
    else:
        results.append(
            CheckResult(
                "Edge-TTS config",
                "warn",
                f"tts inference_mode={mode}; local Edge-TTS is the no-key option",
            )
        )

    runninghub_key = comfyui.get("runninghub_api_key")
    if _has_value(runninghub_key):
        results.append(CheckResult("RunningHub key", "ok", "configured (<set>)"))
    else:
        results.append(
            CheckResult(
                "RunningHub key",
                "warn",
                "not configured; RunningHub workflows cannot generate images/videos yet",
            )
        )

    comfyui_url = comfyui.get("comfyui_url") or "http://127.0.0.1:8188"
    parsed = urlparse(comfyui_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if is_tcp_listening(host, port):
        results.append(CheckResult("local ComfyUI", "ok", f"{comfyui_url} is listening"))
    else:
        results.append(
            CheckResult(
                "local ComfyUI",
                "warn",
                f"{comfyui_url} is not listening; selfhost workflows need a running ComfyUI",
            )
        )

    return results


def run_preflight(project_root: Path, check_web: bool = True) -> list[CheckResult]:
    config_path = project_root / "config.yaml"
    config = load_config(config_path)

    results = [
        check_python_version(),
        check_command("ffmpeg"),
        check_command("uv"),
        *check_config(config, config_path),
    ]

    if check_web:
        if is_tcp_listening("127.0.0.1", 8501):
            results.append(CheckResult("web ui", "ok", "http://127.0.0.1:8501 is listening"))
        else:
            results.append(
                CheckResult(
                    "web ui",
                    "warn",
                    "not listening on 127.0.0.1:8501; run ./start_web.sh or use screen startup",
                )
            )
    return results


def format_results(results: list[CheckResult]) -> str:
    icons = {"ok": "[OK]", "warn": "[WARN]", "fail": "[FAIL]"}
    lines = ["Pixelle-Video local preflight (offline; no paid API calls)"]
    for result in results:
        lines.append(f"{icons.get(result.status, '[INFO]')} {result.name}: {result.detail}")
    if any(result.status == "warn" for result in results):
        lines.append("")
        lines.append("Warnings mean setup is usable for no-key parts, but some AI media features need keys.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline Pixelle-Video setup checks.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Pixelle-Video project root",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Skip checking whether Streamlit is listening on 127.0.0.1:8501",
    )
    args = parser.parse_args()

    results = run_preflight(args.project_root.resolve(), check_web=not args.no_web)
    print(format_results(results))
    return 1 if any(result.status == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
