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
Media Generation Service - ComfyUI Workflow-based implementation

Supports both image and video generation workflows.
Automatically detects output type based on ExecuteResult.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

from pixelle_video.models.media import MediaResult
from pixelle_video.services.comfy_base_service import ComfyBaseService
from pixelle_video.services.cpa_image import (
    DEFAULT_BASE_URL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MAIN_MODEL,
    DEFAULT_TIMEOUT,
    choose_cpa_image_size,
    generate_cpa_image,
)
from pixelle_video.utils.os_util import (
    get_output_path,
    get_resource_path,
    list_resource_dirs,
    list_resource_files,
)


class MediaService(ComfyBaseService):
    """
    Media generation service - Workflow-based
    
    Uses ComfyKit to execute image/video generation workflows.
    Supports both image_ and video_ workflow prefixes.
    
    Usage:
        # Use default workflow (workflows/image_flux.json)
        media = await pixelle_video.media(prompt="a cat")
        if media.is_image:
            print(f"Generated image: {media.url}")
        elif media.is_video:
            print(f"Generated video: {media.url} ({media.duration}s)")
        
        # Use specific workflow
        media = await pixelle_video.media(
            prompt="a cat",
            workflow="image_flux.json"
        )
        
        # List available workflows
        workflows = pixelle_video.media.list_workflows()
    """
    
    WORKFLOW_PREFIX = ""  # Will be overridden by _scan_workflows
    DEFAULT_WORKFLOW = None  # No hardcoded default, must be configured
    WORKFLOWS_DIR = "workflows"
    CPA_WORKFLOW_KEY = "cpa/gpt-image-2"
    
    def __init__(self, config: dict, core=None):
        """
        Initialize media service
        
        Args:
            config: Full application config dict
            core: PixelleVideoCore instance (for accessing shared ComfyKit)
        """
        super().__init__(config, service_name="image", core=core)  # Keep "image" for config compatibility
    
    def _scan_workflows(self):
        """
        Scan workflows for both image_ and video_ prefixes
        
        Override parent method to support multiple prefixes
        """
        if self._workflows_cache is not None:
            return self._workflows_cache

        workflows = []
        
        # Get all workflow source directories
        source_dirs = list_resource_dirs("workflows")
        
        if not source_dirs:
            logger.warning("No workflow source directories found")
            return workflows
        
        # Scan each source directory for workflow files
        for source_name in source_dirs:
            # Get all JSON files for this source
            workflow_files = list_resource_files("workflows", source_name)
            
            # Filter to only files matching image_ or video_ prefix
            matching_files = [
                f for f in workflow_files 
                if (f.startswith("image_") or f.startswith("video_")) and f.endswith('.json')
            ]
            
            for filename in matching_files:
                try:
                    # Get actual file path
                    file_path = Path(get_resource_path("workflows", source_name, filename))
                    workflow_info = self._parse_workflow_file(file_path, source_name)
                    workflows.append(workflow_info)
                    logger.debug(f"Found workflow: {workflow_info['key']}")
                except Exception as e:
                    logger.error(f"Failed to parse workflow {source_name}/{filename}: {e}")

        workflows.append({
            "name": "gpt-image-2",
            "display_name": "gpt-image-2 - Local CPA",
            "source": "cpa",
            "path": self.CPA_WORKFLOW_KEY,
            "key": self.CPA_WORKFLOW_KEY,
            "model": DEFAULT_IMAGE_MODEL,
        })
        
        # Sort by key (source/name)
        self._workflows_cache = sorted(workflows, key=lambda w: w["key"])
        return self._workflows_cache

    def _get_cpa_output_path(
        self,
        *,
        task_id: Optional[str],
        index: Optional[int],
        output_path: Optional[str],
        output_format: Optional[str],
    ) -> str:
        """Choose a stable local output path for a CPA-generated image."""
        if output_path:
            return output_path

        if task_id is not None and index is not None:
            from pixelle_video.utils.os_util import get_task_frame_path
            return get_task_frame_path(task_id, int(index) - 1, "image")

        ext = (output_format or "png").lower()
        if ext == "jpeg":
            ext = "jpg"

        import uuid

        return get_output_path("cpa_previews", f"{uuid.uuid4().hex[:12]}.{ext}")

    async def _generate_cpa_image(
        self,
        *,
        prompt: str,
        media_type: str,
        width: Optional[int],
        height: Optional[int],
        params: dict,
    ) -> MediaResult:
        """Generate an image through the local CPA route."""
        if media_type != "image":
            raise ValueError(
                "CPA workflow cpa/gpt-image-2 only supports image generation. "
                "Use an image template, or choose a RunningHub/ComfyUI video workflow for video media."
            )

        from pixelle_video.config import config_manager

        llm_config = config_manager.get_llm_config()
        api_key = params.get("cpa_api_key") or llm_config.get("api_key") or os.getenv("CPA_API_KEY")
        base_url = params.get("cpa_base_url") or llm_config.get("base_url") or os.getenv("CPA_BASE_URL") or DEFAULT_BASE_URL
        main_model = params.get("cpa_main_model") or llm_config.get("model") or DEFAULT_MAIN_MODEL
        image_model = params.get("cpa_image_model") or DEFAULT_IMAGE_MODEL
        output_format = params.get("cpa_output_format")
        size = params.get("cpa_size") or choose_cpa_image_size(width, height)

        output_path = self._get_cpa_output_path(
            task_id=params.get("task_id"),
            index=params.get("index"),
            output_path=params.get("output_path"),
            output_format=output_format,
        )

        logger.info(f"Executing local CPA image workflow: {self.CPA_WORKFLOW_KEY}")
        local_path = await generate_cpa_image(
            prompt=prompt,
            output_path=output_path,
            api_key=api_key,
            base_url=base_url,
            main_model=main_model,
            image_model=image_model,
            size=size,
            quality=params.get("cpa_quality"),
            output_format=output_format,
            timeout=float(params.get("cpa_timeout", DEFAULT_TIMEOUT)),
        )

        return MediaResult(media_type="image", url=local_path)
    
    async def __call__(
        self,
        prompt: str,
        workflow: Optional[str] = None,
        # Media type specification (required for proper handling)
        media_type: str = "image",  # "image" or "video"
        # ComfyUI connection (optional overrides)
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        # Common workflow parameters
        width: Optional[int] = None,
        height: Optional[int] = None,
        duration: Optional[float] = None,  # Video duration in seconds (for video workflows)
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        cfg: Optional[float] = None,
        sampler: Optional[str] = None,
        **params
    ) -> MediaResult:
        """
        Generate media (image or video) using workflow
        
        Media type must be specified explicitly via media_type parameter.
        Returns a MediaResult object containing media type and URL.
        
        Args:
            prompt: Media generation prompt
            workflow: Workflow filename (default: from config or "image_flux.json")
            media_type: Type of media to generate - "image" or "video" (default: "image")
            comfyui_url: ComfyUI URL (optional, overrides config)
            runninghub_api_key: RunningHub API key (optional, overrides config)
            width: Media width
            height: Media height
            duration: Target video duration in seconds (only for video workflows, typically from TTS audio duration)
            negative_prompt: Negative prompt
            steps: Sampling steps
            seed: Random seed
            cfg: CFG scale
            sampler: Sampler name
            **params: Additional workflow parameters
        
        Returns:
            MediaResult object with media_type ("image" or "video") and url
        
        Examples:
            # Simplest: use default workflow (workflows/image_flux.json)
            media = await pixelle_video.media(prompt="a beautiful cat")
            if media.is_image:
                print(f"Image: {media.url}")
            
            # Use specific workflow
            media = await pixelle_video.media(
                prompt="a cat",
                workflow="image_flux.json"
            )
            
            # Video workflow
            media = await pixelle_video.media(
                prompt="a cat running",
                workflow="image_video.json"
            )
            if media.is_video:
                print(f"Video: {media.url}, duration: {media.duration}s")
            
            # With additional parameters
            media = await pixelle_video.media(
                prompt="a cat",
                workflow="image_flux.json",
                width=1024,
                height=1024,
                steps=20,
                seed=42
            )
            
            # With absolute path
            media = await pixelle_video.media(
                prompt="a cat",
                workflow="/path/to/custom.json"
            )
            
            # With custom ComfyUI server
            media = await pixelle_video.media(
                prompt="a cat",
                comfyui_url="http://192.168.1.100:8188"
            )
        """
        # 1. Resolve workflow (returns structured info)
        workflow_info = self._resolve_workflow(workflow=workflow)

        if workflow_info["source"] == "cpa":
            return await self._generate_cpa_image(
                prompt=prompt,
                media_type=media_type,
                width=width,
                height=height,
                params=params,
            )
        
        # 2. Build workflow parameters (ComfyKit config is now managed by core)
        workflow_params = {"prompt": prompt}
        
        # Add optional parameters
        if width is not None:
            workflow_params["width"] = width
        if height is not None:
            workflow_params["height"] = height
        if duration is not None:
            workflow_params["duration"] = duration
            if media_type == "video":
                logger.info(f"📏 Target video duration: {duration:.2f}s (from TTS audio)")
        if negative_prompt is not None:
            workflow_params["negative_prompt"] = negative_prompt
        if steps is not None:
            workflow_params["steps"] = steps
        if seed is not None:
            workflow_params["seed"] = seed
        if cfg is not None:
            workflow_params["cfg"] = cfg
        if sampler is not None:
            workflow_params["sampler"] = sampler
        
        # Add any additional parameters
        workflow_params.update(params)
        
        logger.debug(f"Workflow parameters: {workflow_params}")
        
        # 4. Execute workflow using shared ComfyKit instance from core
        try:
            # Get shared ComfyKit instance (lazy initialization + config hot-reload)
            kit = await self.core._get_or_create_comfykit()
            
            # Determine what to pass to ComfyKit based on source
            if workflow_info["source"] == "runninghub" and "workflow_id" in workflow_info:
                # RunningHub: pass workflow_id (ComfyKit will use runninghub backend)
                workflow_input = workflow_info["workflow_id"]
                logger.info(f"Executing RunningHub workflow: {workflow_input}")
            else:
                # Selfhost: pass file path (ComfyKit will use local ComfyUI)
                workflow_input = workflow_info["path"]
                logger.info(f"Executing selfhost workflow: {workflow_input}")
            
            result = await kit.execute(workflow_input, workflow_params)
            
            # 5. Handle result based on specified media_type
            if result.status != "completed":
                error_msg = result.msg or "Unknown error"
                logger.error(f"Media generation failed: {error_msg}")
                raise Exception(f"Media generation failed: {error_msg}")
            
            # Extract media based on specified type
            if media_type == "video":
                # Video workflow - get video from result
                if not result.videos:
                    logger.error("No video generated (workflow returned no videos)")
                    raise Exception("No video generated")
                
                video_url = result.videos[0]
                logger.info(f"✅ Generated video: {video_url}")
                
                # Try to extract duration from result (if available)
                duration = None
                if hasattr(result, 'duration') and result.duration:
                    duration = result.duration
                
                return MediaResult(
                    media_type="video",
                    url=video_url,
                    duration=duration
                )
            else:  # image
                # Image workflow - get image from result
                if not result.images:
                    logger.error("No image generated (workflow returned no images)")
                    raise Exception("No image generated")
                
                image_url = result.images[0]
                logger.info(f"✅ Generated image: {image_url}")
                
                return MediaResult(
                    media_type="image",
                    url=image_url
                )
        
        except Exception as e:
            logger.error(f"Media generation error: {e}")
            raise
