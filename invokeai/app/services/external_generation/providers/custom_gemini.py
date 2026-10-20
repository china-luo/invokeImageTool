"""Native Gemini text-to-image transport for explicitly configured custom models."""

from urllib.parse import quote

import requests

from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGeneratedImage,
    ExternalGenerationRequest,
    ExternalGenerationResult,
)
from invokeai.app.services.external_generation.image_utils import decode_image_base64

_GEMINI_ASPECT_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")


def generate_custom_gemini_image(
    request: ExternalGenerationRequest, api_key: str, base_url: str
) -> ExternalGenerationResult:
    if (
        request.mode != "txt2img"
        or request.init_image is not None
        or request.mask_image is not None
        or request.reference_images
    ):
        raise ExternalProviderRequestError("Custom Gemini models currently support text-to-image only")
    if request.num_images != 1:
        raise ExternalProviderRequestError("Custom Gemini models support one image request at a time")
    if request.image_size is not None:
        raise ExternalProviderRequestError("Custom Gemini models do not support image size presets")

    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url.removesuffix("/v1") + "/v1beta"
    elif not base_url.endswith("/v1beta"):
        base_url += "/v1beta"
    model_id = request.model.provider_model_id.removeprefix("models/")
    if not model_id or any(ord(character) < 32 or ord(character) == 127 for character in model_id):
        raise ExternalProviderRequestError("Custom Gemini model ID is invalid")
    endpoint = f"{base_url}/models/{quote(model_id, safe='')}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": _select_aspect_ratio(request.width, request.height)},
        },
    }
    try:
        response = requests.post(
            endpoint,
            headers={"x-goog-api-key": api_key},
            json=payload,
            timeout=120,
            # The credential header must never be forwarded to another endpoint via a redirect.
            allow_redirects=False,
        )
    except requests.RequestException:
        raise ExternalProviderRequestError(
            "Gemini image API request failed; check the endpoint and connection"
        ) from None
    if not response.ok:
        raise ExternalProviderRequestError(f"Gemini image API request failed with status {response.status_code}")
    # Requests considers redirects successful, but this transport deliberately follows none.
    if 300 <= response.status_code < 400:
        raise ExternalProviderRequestError("Gemini image API redirects are not supported")
    try:
        data = response.json()
    except ValueError:
        raise ExternalProviderRequestError("Gemini image API response was not valid JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        raise ExternalProviderRequestError("Gemini image API response is missing image candidates")
    images: list[ExternalGeneratedImage] = []
    for candidate in data["candidates"]:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            continue
        for part in content["parts"]:
            if not isinstance(part, dict):
                continue
            inline_data = part.get("inlineData") or part.get("inline_data")
            if not isinstance(inline_data, dict):
                continue
            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type")
            encoded = inline_data.get("data")
            if (
                not isinstance(mime_type, str)
                or not mime_type.startswith("image/")
                or not isinstance(encoded, str)
                or not encoded
            ):
                continue
            try:
                image = decode_image_base64(encoded)
            except (ValueError, OSError):
                raise ExternalProviderRequestError("Gemini image API returned invalid image data") from None
            images.append(ExternalGeneratedImage(image=image, seed=request.seed))
    if not images:
        # Upstream errors, text, finish messages and file URIs may include secrets or private prompts.
        raise ExternalProviderRequestError("Gemini image API response contained no inline images")
    return ExternalGenerationResult(
        images=images, seed_used=request.seed, provider_metadata={"model": request.model.provider_model_id}
    )


def _select_aspect_ratio(width: int, height: int) -> str:
    ratio = width / height
    return min(
        _GEMINI_ASPECT_RATIOS,
        key=lambda value: abs(float(value.split(":")[0]) / float(value.split(":")[1]) - ratio),
    )
