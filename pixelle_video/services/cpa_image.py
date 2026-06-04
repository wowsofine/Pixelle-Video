# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Local CPA image generation helper.

Uses the local CLIProxyAPI/CPA OpenAI-compatible images generation endpoint.
This keeps image generation local to the user's proxy
instead of requiring RunningHub or a self-hosted ComfyUI workflow.
"""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from loguru import logger

DEFAULT_BASE_URL = "http://127.0.0.1:8317/v1"
DEFAULT_MAIN_MODEL = "gpt-5.4-mini"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_CHAT_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_SQUARE_SIZE = "1024x1024"
DEFAULT_PORTRAIT_SIZE = "1024x1536"
DEFAULT_LANDSCAPE_SIZE = "1536x1024"
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 2.0


def choose_cpa_image_size(width: Optional[int], height: Optional[int]) -> str:
    """Choose a supported image size closest to the requested aspect ratio."""
    if not width or not height:
        return DEFAULT_SQUARE_SIZE
    if height > width:
        return DEFAULT_PORTRAIT_SIZE
    if width > height:
        return DEFAULT_LANDSCAPE_SIZE
    return DEFAULT_SQUARE_SIZE


def build_cpa_image_payload(
    prompt: str,
    *,
    main_model: str = DEFAULT_MAIN_MODEL,
    image_model: str = DEFAULT_IMAGE_MODEL,
    size: str = DEFAULT_SQUARE_SIZE,
    quality: Optional[str] = None,
    output_format: Optional[str] = None,
) -> dict[str, Any]:
    """Build a minimal local CPA images generation request payload."""
    payload: dict[str, Any] = {
        "model": image_model,
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json",
    }
    if quality:
        payload["quality"] = quality
    if output_format:
        payload["output_format"] = output_format

    return payload


def _strip_data_url_prefix(value: str) -> str:
    """Return raw base64 when a response uses a data URL."""
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def extract_image_b64(response: dict[str, Any]) -> str:
    """Extract a generated image base64 payload from a CPA image result."""
    for item in response.get("data", []):
        if not isinstance(item, dict):
            continue
        for key in ("b64_json", "b64", "image_base64", "result"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return _strip_data_url_prefix(value.strip())

    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue

        if item.get("type") == "image_generation_call":
            for key in ("result", "image_base64", "b64_json", "b64"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return _strip_data_url_prefix(value.strip())

            for part in item.get("content", []) or []:
                if not isinstance(part, dict):
                    continue
                for key in ("result", "image_base64", "b64_json", "b64"):
                    value = part.get(key)
                    if isinstance(value, str) and value.strip():
                        return _strip_data_url_prefix(value.strip())

    raise RuntimeError("CPA image response did not contain an image result.")


def extract_openai_chat_image_b64(response: dict[str, Any]) -> str:
    """Extract an image data URL payload from OpenAI-compatible chat output."""
    for choice in response.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            continue

        for image in message.get("images", []) or []:
            if not isinstance(image, dict):
                continue
            image_url = image.get("image_url") or {}
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = image_url
            if isinstance(url, str) and url.strip():
                return _strip_data_url_prefix(url.strip())

        for part in message.get("content", []) or []:
            if not isinstance(part, dict):
                continue
            image_url = part.get("image_url") or {}
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = image_url
            if isinstance(url, str) and url.strip():
                return _strip_data_url_prefix(url.strip())

    raise RuntimeError("CPA chat image response did not contain an image result.")


async def generate_cpa_image(
    *,
    prompt: str,
    output_path: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    main_model: str = DEFAULT_MAIN_MODEL,
    image_model: str = DEFAULT_IMAGE_MODEL,
    size: str = DEFAULT_SQUARE_SIZE,
    quality: Optional[str] = None,
    output_format: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> str:
    """
    Generate an image through the local CPA endpoint and save it to output_path.

    Returns:
        Absolute local path of the saved image.
    """
    api_key = api_key or os.getenv("CPA_API_KEY")
    if not api_key:
        raise RuntimeError("CPA image generation requires llm.api_key or CPA_API_KEY.")
    if not base_url:
        base_url = DEFAULT_BASE_URL

    payload = build_cpa_image_payload(
        prompt,
        main_model=main_model,
        image_model=image_model,
        size=size,
        quality=quality,
        output_format=output_format,
    )

    endpoint = f"{base_url.rstrip('/')}/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    attempts = max(1, max_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            break
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            if exc.response.status_code >= 500 and attempt < attempts:
                logger.warning(
                    "CPA image generation attempt "
                    f"{attempt}/{attempts} failed: HTTP {exc.response.status_code}: {body}. "
                    "Retrying..."
                )
                await asyncio.sleep(retry_delay)
                continue
            raise RuntimeError(
                f"CPA image generation failed: HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            if attempt >= attempts:
                raise RuntimeError(f"CPA image generation failed: {exc}") from exc
            logger.warning(
                f"CPA image generation attempt {attempt}/{attempts} failed: {exc}. Retrying..."
            )
            await asyncio.sleep(retry_delay)

    image_b64 = extract_image_b64(data)
    image_bytes = base64.b64decode(image_b64)

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)

    logger.info(f"✅ Generated CPA image: {path}")
    return str(path)


async def generate_cpa_image_from_chat(
    *,
    prompt: str,
    output_path: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    image_model: str = DEFAULT_CHAT_IMAGE_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> str:
    """
    Generate an image through an OpenAI-compatible chat endpoint.

    Some CPA routes expose Gemini image models as chat completions that return
    images in message.images rather than /images/generations.
    """
    api_key = api_key or os.getenv("CPA_API_KEY")
    if not api_key:
        raise RuntimeError("CPA chat image generation requires llm.api_key or CPA_API_KEY.")
    if not base_url:
        base_url = DEFAULT_BASE_URL

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": image_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }

    attempts = max(1, max_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            break
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            if exc.response.status_code >= 500 and attempt < attempts:
                logger.warning(
                    "CPA chat image generation attempt "
                    f"{attempt}/{attempts} failed: HTTP {exc.response.status_code}: {body}. "
                    "Retrying..."
                )
                await asyncio.sleep(retry_delay)
                continue
            raise RuntimeError(
                f"CPA chat image generation failed: HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            if attempt >= attempts:
                raise RuntimeError(f"CPA chat image generation failed: {exc}") from exc
            logger.warning(
                f"CPA chat image generation attempt {attempt}/{attempts} failed: {exc}. Retrying..."
            )
            await asyncio.sleep(retry_delay)

    image_b64 = extract_openai_chat_image_b64(data)
    image_bytes = base64.b64decode(image_b64)

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)

    logger.info(f"✅ Generated CPA chat image: {path}")
    return str(path)
