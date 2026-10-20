import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from invokeai.app.services.config.config_default import CONFIG_UPDATE_LOCK, InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_common import ExternalGenerationRequest
from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService
from invokeai.app.services.external_generation.image_utils import encode_image_base64
from invokeai.app.services.external_generation.providers import openai
from invokeai.app.services.external_generation.providers.openai import CustomOpenAIProvider
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities

MODEL_ID = "gemini-3.1-flash-image"


def _request(model_id: str = MODEL_ID) -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=ExternalApiModelConfig(
            name="Gemini",
            provider_id="custom",
            provider_model_id=model_id,
            capabilities=ExternalModelCapabilities(supports_negative_prompt=False),
        ),
        mode="txt2img",
        prompt="A blue square",
        seed=None,
        num_images=1,
        width=1024,
        height=1024,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=[],
        metadata=None,
    )


def _provider(url: str = "https://relay.example", model_id: str = MODEL_ID) -> CustomOpenAIProvider:
    return CustomOpenAIProvider(
        InvokeAIAppConfig(
            external_custom_api_key="fake-gemini-key",
            external_custom_base_url=url,
            external_custom_models=[{"id": model_id, "name": "Gemini", "protocol": "gemini"}],
        ),
        logging.getLogger("test"),
    )


def _response(payload: object, status: int = 200) -> Mock:
    return Mock(ok=status < 400, status_code=status, json=Mock(return_value=payload), text="sensitive upstream body")


def _image_payload(format: str = "PNG") -> dict:
    image = Image.new("RGB", (3, 4), "blue")
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Private upstream text"},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg" if format == "JPEG" else "image/png",
                                "data": encode_image_base64(image, format=format),
                            }
                        },
                    ],
                }
            }
        ]
    }


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("https://relay.example", "https://relay.example/v1beta"),
        ("https://relay.example/v1", "https://relay.example/v1beta"),
        ("https://relay.example/v1beta/", "https://relay.example/v1beta"),
        ("https://relay.example/prefix", "https://relay.example/prefix/v1beta"),
        ("https://relay.example/prefix/v1", "https://relay.example/prefix/v1beta"),
        ("https://relay.example/prefix/v1beta", "https://relay.example/prefix/v1beta"),
    ],
)
@pytest.mark.parametrize("format", ["JPEG", "PNG"])
def test_custom_gemini_native_request_uses_header_and_decodes_image_once(
    base_url: str, expected: str, format: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = Mock(return_value=_response(_image_payload(format)))
    monkeypatch.setattr(requests, "post", post)
    result = _provider(base_url).generate(_request())
    post.assert_called_once_with(
        f"{expected}/models/{MODEL_ID}:generateContent",
        headers={"x-goog-api-key": "fake-gemini-key"},
        json={
            "contents": [{"role": "user", "parts": [{"text": "A blue square"}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": "1:1"}},
        },
        timeout=120,
        allow_redirects=False,
    )
    assert "params" not in post.call_args.kwargs
    assert result.images[0].image.size == (3, 4)
    assert result.images[0].image.mode == "RGB"
    assert result.provider_metadata == {"model": MODEL_ID}


@pytest.mark.parametrize(
    "model_id,encoded",
    [
        ("models/gemini-3.1-flash-image", "gemini-3.1-flash-image"),
        ("alias/../other?key=hidden#part", "alias%2F..%2Fother%3Fkey%3Dhidden%23part"),
        ("%2Felsewhere", "%252Felsewhere"),
    ],
)
def test_gemini_model_id_cannot_add_url_path_query_or_fragment(
    model_id: str, encoded: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = Mock(return_value=_response(_image_payload()))
    monkeypatch.setattr(requests, "post", post)
    _provider(model_id=model_id).generate(_request(model_id))
    assert post.call_args.args == (f"https://relay.example/v1beta/models/{encoded}:generateContent",)


@pytest.mark.parametrize("invalid_id", ["models/", "gemini\nimage", "gemini\x00image", "gemini\x7fimage"])
def test_invalid_gemini_model_id_fails_without_network(invalid_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    request = _request()
    invalid_model = request.model.model_copy(update={"provider_model_id": invalid_id})
    # A stale or manually supplied model record is still checked at the transport boundary.
    provider._app_config.external_custom_models[0].id = invalid_id
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ExternalProviderRequestError, match="model ID is invalid"):
        provider.generate(dataclasses.replace(request, model=invalid_model))
    post.assert_not_called()


@pytest.mark.parametrize("status", [400, 401, 429, 503])
def test_gemini_http_failure_is_sanitized_and_not_retried(
    status: int, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    response = _response({"error": {"message": "fake-gemini-key private request text"}}, status=status)
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    service = ExternalGenerationService({"custom": _provider()}, logging.getLogger("test"))
    with pytest.raises(ExternalProviderRequestError) as error:
        service.generate(_request())
    assert str(error.value) == f"Gemini image API request failed with status {status}"
    assert "fake-gemini-key" not in caplog.text
    assert "private request text" not in caplog.text
    response.json.assert_not_called()
    post.assert_called_once()


def test_gemini_redirect_is_not_followed_or_reported_with_sensitive_location(monkeypatch: pytest.MonkeyPatch) -> None:
    post = Mock(return_value=_response({}, status=307))
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ExternalProviderRequestError, match="redirects are not supported"):
        _provider().generate(_request())
    post.assert_called_once()
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"candidates": None},
        {"candidates": [None, {"content": None}]},
        {"candidates": [{"content": {"parts": [{"text": "fake-gemini-key private text"}]}}]},
        {
            "candidates": [
                {
                    "finishMessage": "fake-gemini-key private status",
                    "content": {"parts": [{"fileData": {"fileUri": "https://private.example?token=fake-gemini-key"}}]},
                }
            ]
        },
        {
            "candidates": [
                {"content": {"parts": [{"inlineData": {"mimeType": "text/plain", "data": "fake-gemini-key"}}]}}
            ]
        },
    ],
)
def test_gemini_missing_images_do_not_echo_upstream_content(
    payload: object, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    post = Mock(return_value=_response(payload))
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ExternalProviderRequestError) as error:
        _provider().generate(_request())
    assert str(error.value) in {
        "Gemini image API response is missing image candidates",
        "Gemini image API response contained no inline images",
    }
    assert "fake-gemini-key" not in caplog.text
    assert "private" not in caplog.text
    post.assert_called_once()


@pytest.mark.parametrize(
    "failure,expected",
    [
        (requests.ConnectionError("private token=fake-gemini-key"), "check the endpoint"),
    ],
)
def test_gemini_connection_error_is_sanitized(
    failure: Exception, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = Mock(side_effect=failure)
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ExternalProviderRequestError, match=expected) as error:
        _provider().generate(_request())
    assert "fake-gemini-key" not in str(error.value)
    post.assert_called_once()


def test_gemini_malformed_json_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response({})
    response.json.side_effect = ValueError("fake-gemini-key private response")
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    with pytest.raises(ExternalProviderRequestError, match="not valid JSON"):
        _provider().generate(_request())


def test_gemini_malformed_inline_image_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "not-image"}}]}}]}
    monkeypatch.setattr(requests, "post", Mock(return_value=_response(payload)))
    with pytest.raises(ExternalProviderRequestError, match="invalid image data"):
        _provider().generate(_request())


@pytest.mark.parametrize("old_protocol,new_protocol", [("openai", "gemini"), ("gemini", "xai"), ("xai", "openai")])
def test_protocol_and_credentials_are_one_publication_snapshot(
    old_protocol: str, new_protocol: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = InvokeAIAppConfig(
        external_custom_api_key="old-openai-key",
        external_custom_base_url="https://old.example/v1",
        external_custom_models=[{"id": MODEL_ID, "name": "Model", "protocol": old_protocol}],
    )
    provider = CustomOpenAIProvider(config, logging.getLogger("test"))
    updates = {
        "external_custom_api_key": "new-gemini-key",
        "external_custom_base_url": "https://new.example/v1beta",
        "external_custom_models": [{"id": MODEL_ID, "name": "Model", "protocol": new_protocol}],
    }
    first_field_published = Event()
    finish_publication = Event()
    read_attempted = Event()
    immediate_acquisitions: list[bool] = []
    original_setattr = InvokeAIAppConfig.__setattr__

    def pause_publication(self: InvokeAIAppConfig, name: str, value: object) -> None:
        original_setattr(self, name, value)
        if self is config and name in updates and not first_field_published.is_set():
            first_field_published.set()
            assert finish_publication.wait(timeout=5)

    class ObservedReadLock:
        def __enter__(self) -> None:
            acquired = CONFIG_UPDATE_LOCK.acquire(blocking=False)
            immediate_acquisitions.append(acquired)
            read_attempted.set()
            if not acquired:
                CONFIG_UPDATE_LOCK.acquire()

        def __exit__(self, *args: object) -> None:
            CONFIG_UPDATE_LOCK.release()

    payload = (
        _image_payload()
        if new_protocol == "gemini"
        else {"data": [{"b64_json": encode_image_base64(Image.new("RGB", (2, 2)))}]}
    )
    post = Mock(return_value=_response(payload))
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(InvokeAIAppConfig, "__setattr__", pause_publication)
    monkeypatch.setattr(openai, "CONFIG_UPDATE_LOCK", ObservedReadLock())
    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(config.update_config, updates)
        try:
            assert first_field_published.wait(timeout=5)
            generation = executor.submit(provider.generate, _request())
            assert read_attempted.wait(timeout=5)
            assert immediate_acquisitions == [False]
            assert not generation.done()
        finally:
            finish_publication.set()
        writer.result(timeout=5)
        assert len(generation.result(timeout=5).images) == 1
    post.assert_called_once()
    if new_protocol == "gemini":
        assert post.call_args.args == (f"https://new.example/v1beta/models/{MODEL_ID}:generateContent",)
        assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "new-gemini-key"}
    else:
        assert post.call_args.args == ("https://new.example/v1/images/generations",)
        assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer new-gemini-key"}
        assert ("aspect_ratio" in post.call_args.kwargs["json"]) == (new_protocol == "xai")


def test_inflight_gemini_request_does_not_block_protocol_or_key_change(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    request_started = Event()
    finish_request = Event()

    def blocked_post(*args: object, **kwargs: object) -> Mock:
        request_started.set()
        assert finish_request.wait(timeout=5)
        return _response(_image_payload())

    monkeypatch.setattr(requests, "post", blocked_post)
    with ThreadPoolExecutor(max_workers=2) as executor:
        generation = executor.submit(provider.generate, _request())
        try:
            assert request_started.wait(timeout=5)
            update = executor.submit(
                provider._app_config.update_config,
                {
                    "external_custom_api_key": "replacement",
                    "external_custom_base_url": "https://new.example",
                    "external_custom_models": [{"id": MODEL_ID, "name": "Model", "protocol": "xai"}],
                },
            )
            update.result(timeout=5)
            assert not generation.done()
        finally:
            finish_request.set()
        assert len(generation.result(timeout=5).images) == 1


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"mode": "img2img"}, "text-to-image only"),
        ({"mode": "inpaint"}, "text-to-image only"),
        ({"init_image": Image.new("RGB", (2, 2))}, "text-to-image only"),
        ({"mask_image": Image.new("L", (2, 2))}, "text-to-image only"),
        ({"reference_images": [object()]}, "text-to-image only"),
        ({"num_images": 2}, "one image request"),
        ({"image_size": "2K"}, "do not support image size presets"),
    ],
)
def test_gemini_unsupported_inputs_are_rejected_before_network(
    changes: dict, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ExternalProviderRequestError, match=expected):
        _provider().generate(dataclasses.replace(_request(), **changes))
    post.assert_not_called()


@pytest.mark.parametrize(
    "width,height,expected", [(1024, 1024, "1:1"), (1600, 900, "16:9"), (900, 1600, "9:16"), (1400, 1000, "4:3")]
)
def test_gemini_dimensions_are_translated_into_a_supported_aspect_ratio(
    width: int, height: int, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = Mock(return_value=_response(_image_payload()))
    monkeypatch.setattr(requests, "post", post)
    _provider().generate(dataclasses.replace(_request(), width=width, height=height))
    post.assert_called_once()
    image_config = post.call_args.kwargs["json"]["generationConfig"]["imageConfig"]
    assert image_config == {"aspectRatio": expected}
