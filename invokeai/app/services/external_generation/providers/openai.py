from __future__ import annotations

import io

import requests
from PIL import Image
from PIL.Image import Image as PILImageType

from invokeai.app.services.config.config_default import CONFIG_UPDATE_LOCK
from invokeai.app.services.external_generation.errors import (
    ExternalProviderRateLimitError,
    ExternalProviderRequestError,
)
from invokeai.app.services.external_generation.external_generation_base import ExternalProvider
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGeneratedImage,
    ExternalGenerationRequest,
    ExternalGenerationResult,
)
from invokeai.app.services.external_generation.image_utils import decode_image_base64
from invokeai.app.services.external_generation.providers.custom_gemini import generate_custom_gemini_image
from invokeai.app.util.ssrf import UnsafeDownloadURLException, build_guarded_session, validate_download_url


class OpenAIProvider(ExternalProvider):
    provider_id = "openai"

    _DEFAULT_TIMEOUT = 120
    _GPT_IMAGE_TIMEOUT = 300
    _DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024
    # Preserve the existing timeout for older models; new GPT Image aliases can take longer.
    _MODEL_TIMEOUTS: dict[str, int] = {
        "gpt-image-1": 120,
        "gpt-image-1.5": 120,
        "gpt-image-1-mini": 120,
        "gpt-image-2": 300,
    }

    def _configuration(self) -> tuple[str | None, str | None]:
        with CONFIG_UPDATE_LOCK:
            return self._app_config.external_openai_api_key, self._app_config.external_openai_base_url

    def is_configured(self) -> bool:
        return bool(self._configuration()[0])

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        return self._generate(request, *self._configuration())

    def _generate(
        self,
        request: ExternalGenerationRequest,
        api_key: str | None,
        configured_base_url: str | None,
        *,
        use_xai_protocol: bool = False,
    ) -> ExternalGenerationResult:
        if use_xai_protocol and (
            request.mode != "txt2img"
            or request.init_image is not None
            or request.mask_image is not None
            or request.reference_images
        ):
            raise ExternalProviderRequestError("Custom xAI models currently support text-to-image only")
        if not api_key:
            raise ExternalProviderRequestError("OpenAI API key is not configured")

        model_id = request.model.provider_model_id
        is_gpt_image = model_id.startswith("gpt-image-") and not use_xai_protocol
        timeout = self._MODEL_TIMEOUTS.get(model_id, self._GPT_IMAGE_TIMEOUT if is_gpt_image else self._DEFAULT_TIMEOUT)
        size = f"{request.width}x{request.height}"
        base_url = (configured_base_url or "https://api.openai.com").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        headers = {"Authorization": f"Bearer {api_key}"}

        use_edits_endpoint = request.mode != "txt2img" or bool(request.reference_images)

        opts = request.provider_options or {}

        if not use_edits_endpoint:
            payload: dict[str, object] = {
                "model": model_id,
                "prompt": request.prompt,
                "n": request.num_images,
            }
            if use_xai_protocol:
                payload["aspect_ratio"] = _select_xai_aspect_ratio(request.width, request.height)
            else:
                payload["size"] = size
            # GPT Image models use output_format; DALL-E uses response_format
            if is_gpt_image:
                payload["output_format"] = "png"
            else:
                payload["response_format"] = "b64_json"
            if is_gpt_image:
                if opts.get("quality") and opts["quality"] != "auto":
                    payload["quality"] = opts["quality"]
                if opts.get("background") and opts["background"] != "auto":
                    payload["background"] = opts["background"]
            request_options = {"allow_redirects": False} if use_xai_protocol else {}
            response = self._post(
                f"{base_url}/images/generations",
                headers=headers,
                json=payload,
                timeout=timeout,
                **request_options,
            )
        else:
            images: list[PILImageType] = []
            if request.init_image is not None:
                images.append(request.init_image)
            images.extend(reference.image for reference in request.reference_images)
            if not images:
                raise ExternalProviderRequestError(
                    "OpenAI image edits require at least one image (init image or reference image)"
                )

            files: list[tuple[str, tuple[str, io.BytesIO, str]]] = []
            image_field_name = "image" if len(images) == 1 else "image[]"
            for index, image in enumerate(images):
                image_buffer = io.BytesIO()
                image.save(image_buffer, format="PNG")
                image_buffer.seek(0)
                files.append((image_field_name, (f"image_{index}.png", image_buffer, "image/png")))

            if request.mask_image is not None:
                mask_buffer = io.BytesIO()
                request.mask_image.save(mask_buffer, format="PNG")
                mask_buffer.seek(0)
                files.append(("mask", ("mask.png", mask_buffer, "image/png")))

            data: dict[str, object] = {
                "model": model_id,
                "prompt": request.prompt,
                "n": request.num_images,
                "size": size,
            }
            if is_gpt_image:
                data["output_format"] = "png"
            else:
                data["response_format"] = "b64_json"
            if is_gpt_image:
                if opts.get("quality") and opts["quality"] != "auto":
                    data["quality"] = opts["quality"]
                if opts.get("background") and opts["background"] != "auto":
                    data["background"] = opts["background"]
                if opts.get("input_fidelity"):
                    data["input_fidelity"] = opts["input_fidelity"]
            response = self._post(
                f"{base_url}/images/edits",
                headers=headers,
                data=data,
                files=files,
                timeout=timeout,
            )

        if use_xai_protocol and 300 <= response.status_code < 400:
            raise ExternalProviderRequestError("xAI image API redirects are not supported")
        if not response.ok:
            if response.status_code == 429 and not use_xai_protocol:
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                raise ExternalProviderRateLimitError(
                    f"OpenAI rate limit exceeded. {f'Retry after {retry_after:.0f}s.' if retry_after else 'Please try again later.'}",
                    retry_after=retry_after,
                )
            detail = f": {response.text}" if self.provider_id == "openai" else ""
            if self.provider_id == "custom":
                hint = _custom_image_api_error_hint(response)
                if hint:
                    detail = f". {hint}"
            label = "OpenAI" if self.provider_id == "openai" else "Image API"
            raise ExternalProviderRequestError(f"{label} request failed with status {response.status_code}{detail}")

        try:
            response_payload = response.json()
        except ValueError:
            raise ExternalProviderRequestError("Image API response was not valid JSON") from None
        if not isinstance(response_payload, dict):
            raise ExternalProviderRequestError("OpenAI response payload was not a JSON object")
        images: list[ExternalGeneratedImage] = []
        data_items = response_payload.get("data")
        if not isinstance(data_items, list):
            raise ExternalProviderRequestError("OpenAI response payload missing image data")
        for item in data_items:
            if not isinstance(item, dict):
                continue
            encoded = item.get("b64_json")
            if isinstance(encoded, str) and encoded:
                try:
                    image = decode_image_base64(encoded)
                except (ValueError, OSError):
                    raise ExternalProviderRequestError("Image API returned invalid image data") from None
            elif isinstance(item.get("url"), str) and item["url"]:
                image = self._download_image(item["url"])
            else:
                continue
            images.append(ExternalGeneratedImage(image=image, seed=request.seed))

        if not images:
            raise ExternalProviderRequestError("OpenAI response contained no images")

        return ExternalGenerationResult(
            images=images,
            seed_used=request.seed,
            provider_request_id=response.headers.get("x-request-id"),
            provider_metadata={"model": model_id},
        )

    def _post(self, *args: object, **kwargs: object) -> requests.Response:
        try:
            return requests.post(*args, **kwargs)
        except requests.Timeout:
            raise ExternalProviderRequestError(
                "Image API request timed out; the remote endpoint may still be generating. "
                "Check its status before trying again."
            ) from None
        except requests.RequestException:
            raise ExternalProviderRequestError("Image API request failed; check the endpoint and connection") from None

    def _download_image(self, url: str) -> PILImageType:
        max_bytes = self._DOWNLOAD_MAX_BYTES
        try:
            # Generated URLs are untrusted. Do not forward the provider's Authorization header.
            validate_download_url(url)
            with build_guarded_session() as session:
                with session.get(url, timeout=60, stream=True) as response:
                    if not response.ok:
                        raise ExternalProviderRequestError("Failed to download generated image")
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) > max_bytes:
                        raise ExternalProviderRequestError("Generated image exceeds the 32 MiB download limit")
                    buffer = bytearray()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        buffer.extend(chunk)
                        if len(buffer) > max_bytes:
                            raise ExternalProviderRequestError("Generated image exceeds the 32 MiB download limit")
            return Image.open(io.BytesIO(buffer)).convert("RGB")
        except (UnsafeDownloadURLException, requests.RequestException, ValueError, OSError):
            raise ExternalProviderRequestError("Unable to download the generated image safely") from None


class CustomOpenAIProvider(OpenAIProvider):
    provider_id = "custom"

    def _configuration(self) -> tuple[str | None, str | None]:
        with CONFIG_UPDATE_LOCK:
            return self._app_config.external_custom_api_key, self._app_config.external_custom_base_url

    def is_configured(self) -> bool:
        api_key, base_url = self._configuration()
        return bool(api_key and base_url)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        # Read protocol and credentials as one snapshot, then release the lock before any network I/O.
        with CONFIG_UPDATE_LOCK:
            api_key = self._app_config.external_custom_api_key
            base_url = self._app_config.external_custom_base_url
            configured_model = next(
                (
                    model
                    for model in self._app_config.external_custom_models
                    if model.id == request.model.provider_model_id
                ),
                None,
            )
            if configured_model is None:
                raise ExternalProviderRequestError("Custom image model is no longer configured")
            protocol = configured_model.protocol
        if not api_key or not base_url:
            raise ExternalProviderRequestError("Custom image API credentials and base URL are not configured")
        # A shared endpoint accepts any protocol's conventional version suffix.
        base_url = base_url.rstrip("/")
        for version in ("/v1beta", "/v1"):
            if base_url.endswith(version):
                base_url = base_url.removesuffix(version)
                break
        if protocol == "gemini":
            return generate_custom_gemini_image(request, api_key, base_url)
        return self._generate(request, api_key, base_url, use_xai_protocol=protocol == "xai")


def _custom_image_api_error_hint(response: requests.Response) -> str | None:
    fallback = None
    if response.status_code == 502:
        fallback = "The remote image endpoint returned a gateway error. Check the relay and model availability."
    elif response.status_code == 503:
        fallback = "The remote image endpoint is unavailable. Check the relay and model availability."
    try:
        payload = response.json()
    except ValueError:
        return fallback
    if not isinstance(payload, dict):
        return fallback
    error = payload.get("error")
    if not isinstance(error, dict):
        return fallback
    message = error.get("message")
    if not isinstance(message, str):
        return fallback
    # Only translate confirmed, fixed upstream errors; never expose arbitrary provider response text.
    if message == "Images API is not supported for this platform":
        return "The upstream platform does not support the Images API. Check the endpoint and API key group."
    if message == "API key group platform is not gemini":
        return "The API key group uses a different platform. Check the API key group and model protocol."
    if message == "xAI upstream returned status 400":
        return (
            "The relay reports that xAI rejected the request without providing a reason. "
            "Check the prompt and model parameters."
        )
    return fallback


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# xAI SDK ImageAspectRatio values; the API selects pixels from this ratio rather than an OpenAI size string.
_XAI_ASPECT_RATIOS = (
    "1:1",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "2:3",
    "3:2",
    "9:19.5",
    "19.5:9",
    "9:20",
    "20:9",
    "1:2",
    "2:1",
)


def _select_xai_aspect_ratio(width: int, height: int) -> str:
    ratio = width / height
    return min(
        _XAI_ASPECT_RATIOS,
        key=lambda value: abs(float(value.split(":")[0]) / float(value.split(":")[1]) - ratio),
    )
