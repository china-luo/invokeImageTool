import dataclasses
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from invokeai.app.services.config.config_default import CONFIG_UPDATE_LOCK, InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalReferenceImage,
)
from invokeai.app.services.external_generation.image_utils import encode_image_base64
from invokeai.app.services.external_generation.providers import openai
from invokeai.app.services.external_generation.providers.openai import CustomOpenAIProvider, OpenAIProvider
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities


def _request(model_id: str = "vendor/my-image") -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=ExternalApiModelConfig(
            name="My model",
            provider_id="custom",
            provider_model_id=model_id,
            capabilities=ExternalModelCapabilities(supports_negative_prompt=False),
        ),
        mode="txt2img",
        prompt="A red cube",
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


def _provider(url: str = "https://images.example/v1", model_id: str = "vendor/my-image") -> CustomOpenAIProvider:
    return CustomOpenAIProvider(
        InvokeAIAppConfig(
            external_custom_api_key="fake-key",
            external_custom_base_url=url,
            external_custom_models=[{"id": model_id, "name": "My model"}],
        ),
        logging.getLogger("test"),
    )


def _response(payload: dict) -> Mock:
    return Mock(ok=True, status_code=200, headers={"x-request-id": "req-test"}, json=Mock(return_value=payload))


@pytest.mark.parametrize("model_id", ["vendor/my-image", "dall-e-2", "dall-e-3", "other-gpt-image-2", "gpt-image"])
@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://images.example", "https://images.example/v1/images/generations"),
        ("https://images.example/v1/", "https://images.example/v1/images/generations"),
        ("https://images.example/proxy/v1", "https://images.example/proxy/v1/images/generations"),
    ],
)
def test_custom_requests_preserve_arbitrary_model_id_and_normalize_api_root(
    model_id: str, base_url: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (16, 16), "red")
    post = Mock(return_value=_response({"data": [{"b64_json": encode_image_base64(image)}]}))
    monkeypatch.setattr(requests, "post", post)
    provider = _provider(base_url, model_id)
    result = provider.generate(_request(model_id))
    assert provider.is_configured()
    assert post.call_args.args == (expected,)
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer fake-key"}
    assert post.call_args.kwargs["json"] == {
        "model": model_id,
        "prompt": "A red cube",
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json",
    }
    assert post.call_args.kwargs["timeout"] == 120
    assert result.images[0].image.getpixel((0, 0)) == (255, 0, 0)
    assert result.provider_request_id == "req-test"


@pytest.mark.parametrize(
    "model_id,timeout",
    [("gpt-image-1", 120), ("gpt-image-1.5", 120), ("gpt-image-1-mini", 120), ("gpt-image-2", 300)],
)
def test_custom_gpt_image_requests_use_gpt_image_format(
    model_id: str, timeout: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = Mock(return_value=_response({"data": [{"b64_json": encode_image_base64(Image.new("RGB", (2, 2)))}]}))
    monkeypatch.setattr(requests, "post", post)
    _provider(model_id=model_id).generate(_request(model_id))
    assert post.call_args.kwargs["json"]["output_format"] == "png"
    assert "response_format" not in post.call_args.kwargs["json"]
    assert post.call_args.kwargs["timeout"] == timeout


@pytest.mark.parametrize("model_id", ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"])
@pytest.mark.parametrize("with_reference", [False, True])
def test_custom_gpt_image_aliases_use_png_options_and_extended_timeout(
    model_id: str, with_reference: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (2, 2), "red")
    post = Mock(return_value=_response({"data": [{"b64_json": encode_image_base64(image)}]}))
    monkeypatch.setattr(requests, "post", post)
    request = dataclasses.replace(
        _request(model_id),
        width=768,
        height=1368,
        reference_images=[ExternalReferenceImage(image=image)] if with_reference else [],
        provider_options={"quality": "high", "background": "opaque", "input_fidelity": "high"},
    )

    result = _provider(model_id=model_id).generate(request)

    post.assert_called_once()
    endpoint = "edits" if with_reference else "generations"
    assert post.call_args.args == (f"https://images.example/v1/images/{endpoint}",)
    expected_payload = {
        "model": model_id,
        "prompt": "A red cube",
        "n": 1,
        "size": "768x1368",
        "output_format": "png",
        "quality": "high",
        "background": "opaque",
    }
    if with_reference:
        expected_payload["input_fidelity"] = "high"
        assert post.call_args.kwargs["data"] == expected_payload
        field, (filename, image_buffer, content_type) = post.call_args.kwargs["files"][0]
        assert (field, filename, content_type) == ("image", "image_0.png", "image/png")
        assert Image.open(image_buffer).getpixel((0, 0)) == (255, 0, 0)
    else:
        assert post.call_args.kwargs["json"] == expected_payload
    assert post.call_args.kwargs["timeout"] == 300
    assert result.images[0].image.getpixel((0, 0)) == (255, 0, 0)


def test_custom_edits_use_same_versioned_root(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (2, 2))
    post = Mock(return_value=_response({"data": [{"b64_json": encode_image_base64(image)}]}))
    monkeypatch.setattr(requests, "post", post)
    _provider().generate(dataclasses.replace(_request(), mode="img2img", init_image=image))
    assert post.call_args.args == ("https://images.example/v1/images/edits",)
    assert post.call_args.kwargs["data"]["model"] == "vendor/my-image"
    assert post.call_args.kwargs["files"][0][0] == "image"


def test_url_response_downloads_with_guard_and_without_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://cdn.example/generated.png"
    post = Mock(return_value=_response({"data": [{"url": url}]}))
    image = Image.new("RGB", (3, 4), "blue")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = Mock(ok=True, headers={}, iter_content=Mock(return_value=[buffer.getvalue()]))
    session = Mock(get=Mock(return_value=nullcontext(response)))
    guard = Mock()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(openai, "validate_download_url", guard)
    monkeypatch.setattr(openai, "build_guarded_session", lambda: nullcontext(session))
    result = _provider().generate(_request())
    guard.assert_called_once_with(url)
    session.get.assert_called_once_with(url, timeout=60, stream=True)
    assert result.images[0].image.size == (3, 4)
    assert result.images[0].image.getpixel((0, 0)) == (0, 0, 255)


def test_unsafe_image_url_is_rejected_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "post", Mock(return_value=_response({"data": [{"url": "http://127.0.0.1/private"}]})))
    session = Mock()
    monkeypatch.setattr(openai, "build_guarded_session", session)
    with pytest.raises(ExternalProviderRequestError, match="safely"):
        _provider().generate(_request())
    session.assert_not_called()


@pytest.mark.parametrize("headers,chunks", [({"Content-Length": "101"}, []), ({}, [b"a" * 60, b"b" * 60])])
def test_download_cap_checks_header_and_stream(
    headers: dict, chunks: list[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        requests, "post", Mock(return_value=_response({"data": [{"url": "https://cdn.example/image"}]}))
    )
    response = Mock(ok=True, headers=headers, iter_content=Mock(return_value=chunks))
    session = Mock(get=Mock(return_value=nullcontext(response)))
    monkeypatch.setattr(openai, "validate_download_url", lambda _: None)
    monkeypatch.setattr(openai, "build_guarded_session", lambda: nullcontext(session))
    monkeypatch.setattr(CustomOpenAIProvider, "_DOWNLOAD_MAX_BYTES", 100)
    with pytest.raises(ExternalProviderRequestError, match="download limit"):
        _provider().generate(_request())


def test_remote_error_does_not_echo_keys_or_provider_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests, "post", Mock(return_value=Mock(ok=False, status_code=401, text="fake-key sensitive upstream body"))
    )
    with pytest.raises(ExternalProviderRequestError) as error:
        _provider().generate(_request())
    assert str(error.value) == "Image API request failed with status 401"


@pytest.mark.parametrize(
    "message,hint",
    [
        (
            "Images API is not supported for this platform",
            "The upstream platform does not support the Images API. Check the endpoint and API key group.",
        ),
        (
            "API key group platform is not gemini",
            "The API key group uses a different platform. Check the API key group and model protocol.",
        ),
        (
            "xAI upstream returned status 400",
            "The relay reports that xAI rejected the request without providing a reason. "
            "Check the prompt and model parameters.",
        ),
    ],
)
@pytest.mark.parametrize(
    "protocol,status", [("openai", 400), ("openai", 502), ("xai", 400), ("xai", 429), ("xai", 503)]
)
def test_known_custom_platform_errors_provide_safe_guidance_without_retry(
    message: str, hint: str, protocol: str, status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService

    provider = _provider()
    provider._app_config.update_config(
        {"external_custom_models": [{"id": "vendor/my-image", "name": "My model", "protocol": protocol}]}
    )
    response = Mock(
        ok=False,
        status_code=status,
        text="fake-key sensitive upstream body",
        json=Mock(return_value={"error": {"message": message, "details": "https://private.example/fake-key"}}),
    )
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    service = ExternalGenerationService({"custom": provider}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderRequestError) as error:
        service.generate(_request())

    assert str(error.value) == f"Image API request failed with status {status}. {hint}"
    post.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "sensitive upstream body",
        {},
        {"error": None},
        {"error": []},
        {"error": "sensitive upstream body"},
        {"error": {}},
        {"error": {"message": None}},
        {"error": {"message": 123}},
        {"error": {"message": ["Images API is not supported for this platform"]}},
        {"error": {"message": "Images API is not supported for this platform fake-key"}},
        {"error": {"message": "API key group platform is not grok"}},
        {"error": {"message": "xAI upstream returned status 400 fake-key"}},
    ],
)
def test_unknown_or_malformed_custom_errors_preserve_sanitized_status(
    payload: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = Mock(
        ok=False,
        status_code=400,
        text="fake-key sensitive upstream body",
        json=Mock(return_value=payload),
    )
    monkeypatch.setattr(requests, "post", Mock(return_value=response))

    with pytest.raises(ExternalProviderRequestError) as error:
        _provider().generate(_request())

    assert str(error.value) == "Image API request failed with status 400"


@pytest.mark.parametrize(
    "status,hint",
    [
        (400, ""),
        (
            502,
            ". The remote image endpoint returned a gateway error. Check the relay and model availability.",
        ),
        (
            503,
            ". The remote image endpoint is unavailable. Check the relay and model availability.",
        ),
    ],
)
def test_invalid_json_custom_error_preserves_safe_guidance(
    status: int, hint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService

    response = Mock(
        ok=False,
        status_code=status,
        text="fake-key sensitive upstream body",
        json=Mock(side_effect=ValueError("fake-key sensitive JSON error")),
    )
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    service = ExternalGenerationService({"custom": _provider()}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderRequestError) as error:
        service.generate(_request())

    assert str(error.value) == f"Image API request failed with status {status}{hint}"
    assert error.value.__context__ is None
    post.assert_called_once()


@pytest.mark.parametrize(
    "protocol,payload",
    [
        ("openai", None),
        ("xai", {"error": []}),
        ("openai", {"error": {"message": 123}}),
        ("xai", {"error": {"message": "fake-key"}}),
    ],
)
@pytest.mark.parametrize(
    "status,hint",
    [
        (502, "The remote image endpoint returned a gateway error. Check the relay and model availability."),
        (503, "The remote image endpoint is unavailable. Check the relay and model availability."),
    ],
)
def test_custom_gateway_errors_provide_safe_guidance_without_retry(
    protocol: str, status: int, hint: str, payload: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService

    provider = _provider()
    provider._app_config.update_config(
        {"external_custom_models": [{"id": "vendor/my-image", "name": "My model", "protocol": protocol}]}
    )
    response = Mock(
        ok=False,
        status_code=status,
        text="https://private.example/fake-key sensitive upstream body",
        json=Mock(return_value=payload),
    )
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    service = ExternalGenerationService({"custom": provider}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderRequestError) as error:
        service.generate(_request())

    assert str(error.value) == f"Image API request failed with status {status}. {hint}"
    post.assert_called_once()


@pytest.mark.parametrize("status", [400, 502, 503])
def test_openai_http_errors_keep_existing_error_handling(status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock(ok=False, status_code=status, text="OpenAI upstream error")
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    provider = OpenAIProvider(InvokeAIAppConfig(external_openai_api_key="fake-key"), logging.getLogger("test"))

    with pytest.raises(ExternalProviderRequestError) as error:
        provider.generate(_request())

    assert str(error.value) == f"OpenAI request failed with status {status}: OpenAI upstream error"
    response.json.assert_not_called()


def test_request_failure_has_sanitized_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "post", Mock(side_effect=requests.ConnectionError("sensitive endpoint details")))
    with pytest.raises(ExternalProviderRequestError, match="check the endpoint") as error:
        _provider().generate(_request())
    assert "sensitive" not in str(error.value)


@pytest.mark.parametrize("error_type", [requests.Timeout, requests.ReadTimeout, requests.ConnectTimeout])
def test_generation_timeout_is_sanitized_and_not_retried(
    error_type: type[requests.Timeout], monkeypatch: pytest.MonkeyPatch
) -> None:
    from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService

    post = Mock(side_effect=error_type("https://private.example/fake-key sensitive endpoint details"))
    monkeypatch.setattr(requests, "post", post)
    model_id = "gpt-image-2.5-sunburst"
    service = ExternalGenerationService({"custom": _provider(model_id=model_id)}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderRequestError) as error:
        service.generate(_request(model_id))

    assert str(error.value) == (
        "Image API request timed out; the remote endpoint may still be generating. "
        "Check its status before trying again."
    )
    assert error.value.__suppress_context__
    assert error.value.__cause__ is None
    post.assert_called_once()


@pytest.mark.parametrize("provider_type,prefix", [(OpenAIProvider, "openai"), (CustomOpenAIProvider, "custom")])
def test_credentials_snapshot_waits_for_complete_config_publication(
    provider_type: type[OpenAIProvider], prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_field = f"external_{prefix}_api_key"
    url_field = f"external_{prefix}_base_url"
    config = InvokeAIAppConfig(
        **{key_field: "old-key", url_field: "https://old.example/v1"},
        external_custom_models=[{"id": "vendor/my-image", "name": "My model"}],
    )
    provider = provider_type(config, logging.getLogger("test"))
    updates = {key_field: "new-key", url_field: "https://new.example/v1"}
    first_field_published = Event()
    finish_publication = Event()
    read_attempted = Event()
    immediate_acquisitions: list[bool] = []
    original_setattr = InvokeAIAppConfig.__setattr__

    def pause_after_first_field(self: InvokeAIAppConfig, name: str, value: object) -> None:
        original_setattr(self, name, value)
        if self is config and name in updates and not first_field_published.is_set():
            first_field_published.set()
            assert finish_publication.wait(timeout=5), "Config publication was not released"

    class ObservedReadLock:
        def __enter__(self) -> None:
            acquired = CONFIG_UPDATE_LOCK.acquire(blocking=False)
            immediate_acquisitions.append(acquired)
            read_attempted.set()
            if not acquired:
                CONFIG_UPDATE_LOCK.acquire()

        def __exit__(self, *args: object) -> None:
            CONFIG_UPDATE_LOCK.release()

    monkeypatch.setattr(InvokeAIAppConfig, "__setattr__", pause_after_first_field)
    monkeypatch.setattr(openai, "CONFIG_UPDATE_LOCK", ObservedReadLock())
    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(config.update_config, updates)
        try:
            assert first_field_published.wait(timeout=5)
            reader = executor.submit(provider._configuration)
            assert read_attempted.wait(timeout=5)
            assert immediate_acquisitions == [False]
            assert not reader.done()
        finally:
            finish_publication.set()
        writer.result(timeout=5)
        assert reader.result(timeout=5) == ("new-key", "https://new.example/v1")


@pytest.mark.parametrize("provider_type,prefix", [(OpenAIProvider, "openai"), (CustomOpenAIProvider, "custom")])
def test_config_updates_do_not_wait_for_an_inflight_http_request(
    provider_type: type[OpenAIProvider], prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_field = f"external_{prefix}_api_key"
    url_field = f"external_{prefix}_base_url"
    config = InvokeAIAppConfig(
        **{key_field: "old-key", url_field: "https://old.example/v1"},
        external_custom_models=[{"id": "vendor/my-image", "name": "My model"}],
    )
    provider = provider_type(config, logging.getLogger("test"))
    request_started = Event()
    finish_request = Event()
    captured: dict[str, object] = {}

    def blocked_post(url: str, **kwargs: object) -> Mock:
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        request_started.set()
        assert finish_request.wait(timeout=5), "HTTP request was not released"
        return _response({"data": [{"b64_json": encode_image_base64(Image.new("RGB", (2, 2)))}]})

    monkeypatch.setattr(requests, "post", blocked_post)
    with ThreadPoolExecutor(max_workers=2) as executor:
        generation = executor.submit(provider.generate, _request())
        try:
            assert request_started.wait(timeout=5)
            publication = executor.submit(
                config.update_config, {key_field: "new-key", url_field: "https://new.example/v1"}
            )
            publication.result(timeout=5)
            assert provider._configuration() == ("new-key", "https://new.example/v1")
            assert not generation.done()
        finally:
            finish_request.set()
        assert len(generation.result(timeout=5).images) == 1
    assert captured == {
        "url": "https://old.example/v1/images/generations",
        "headers": {"Authorization": "Bearer old-key"},
    }


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (1024, 1024, "1:1"),
        (1200, 900, "4:3"),
        (900, 1200, "3:4"),
        (1600, 900, "16:9"),
        (900, 1950, "9:19.5"),
        (1950, 900, "19.5:9"),
        (900, 2000, "9:20"),
        (1500, 1000, "3:2"),
        (2000, 1000, "2:1"),
        (2100, 1000, "19.5:9"),
        (500, 1000, "1:2"),
    ],
)
@pytest.mark.parametrize(
    "model_id,timeout", [("grok-imagine-image-2.0", 120), ("gpt-image-2.5-sunburst", 120), ("gpt-image-2", 300)]
)
def test_explicit_xai_protocol_uses_supported_aspect_ratio_once(
    model_id: str, timeout: int, width: int, height: int, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = InvokeAIAppConfig(
        external_custom_api_key="fake-key",
        external_custom_base_url="https://images.example/v1beta",
        external_custom_models=[{"id": model_id, "name": "Grok", "protocol": "xai"}],
    )
    provider = CustomOpenAIProvider(config, logging.getLogger("test"))
    post = Mock(return_value=_response({"data": [{"b64_json": encode_image_base64(Image.new("RGB", (2, 2)))}]}))
    monkeypatch.setattr(requests, "post", post)
    provider.generate(dataclasses.replace(_request(model_id), width=width, height=height))
    post.assert_called_once()
    assert post.call_args.args == ("https://images.example/v1/images/generations",)
    assert post.call_args.kwargs["json"] == {
        "model": model_id,
        "prompt": "A red cube",
        "n": 1,
        "aspect_ratio": expected,
        "response_format": "b64_json",
    }
    assert post.call_args.kwargs["timeout"] == timeout


@pytest.mark.parametrize("model_id", ["grok-imagine-image-2.0", "gemini-3.1-flash-image", "dall-e-3"])
def test_model_name_does_not_override_default_openai_protocol(model_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    post = Mock(return_value=_response({"data": [{"b64_json": encode_image_base64(Image.new("RGB", (2, 2)))}]}))
    monkeypatch.setattr(requests, "post", post)
    _provider("https://images.example/proxy/v1beta", model_id).generate(_request(model_id))
    post.assert_called_once()
    assert post.call_args.args == ("https://images.example/proxy/v1/images/generations",)
    assert post.call_args.kwargs["json"]["size"] == "1024x1024"
    assert "aspect_ratio" not in post.call_args.kwargs["json"]


@pytest.mark.parametrize("protocol", ["gemini", "xai"])
@pytest.mark.parametrize("replacement_id", [None, "renamed-model"])
def test_removed_or_renamed_model_is_rejected_before_network(
    protocol: str, replacement_id: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id = "previous-model"
    config = InvokeAIAppConfig(
        external_custom_api_key="fake-key",
        external_custom_base_url="https://images.example",
        external_custom_models=[
            {"id": model_id, "name": "Previous model", "protocol": protocol},
            {"id": "other-model", "name": "Other model"},
        ],
    )
    provider = CustomOpenAIProvider(config, logging.getLogger("test"))
    stale_request = _request(model_id)
    remaining_models = [{"id": "other-model", "name": "Other model"}]
    if replacement_id is not None:
        remaining_models.append({"id": replacement_id, "name": "Renamed model", "protocol": protocol})
    config.update_config({"external_custom_models": remaining_models})
    post = Mock()
    get = Mock()
    session = Mock()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(openai, "build_guarded_session", session)

    assert provider.is_configured()
    with pytest.raises(ExternalProviderRequestError, match="model is no longer configured"):
        provider.generate(stale_request)

    post.assert_not_called()
    get.assert_not_called()
    session.assert_not_called()


@pytest.mark.parametrize("status", [400, 429])
def test_explicit_xai_http_errors_do_not_trigger_service_retries(status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService

    model_id = "grok-imagine-image-2.0"
    config = InvokeAIAppConfig(
        external_custom_api_key="fake-key",
        external_custom_base_url="https://images.example",
        external_custom_models=[{"id": model_id, "name": "Grok", "protocol": "xai"}],
    )
    provider = CustomOpenAIProvider(config, logging.getLogger("test"))
    post = Mock(return_value=Mock(ok=False, status_code=status, text="fake-key sensitive upstream body"))
    monkeypatch.setattr(requests, "post", post)
    service = ExternalGenerationService({"custom": provider}, logging.getLogger("test"))
    with pytest.raises(ExternalProviderRequestError) as error:
        service.generate(_request(model_id))
    assert str(error.value) == f"Image API request failed with status {status}"
    post.assert_called_once()


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "img2img"},
        {"mode": "inpaint"},
        {"init_image": Image.new("RGB", (2, 2))},
        {"mask_image": Image.new("L", (2, 2))},
        {"reference_images": [object()]},
    ],
)
def test_xai_unsupported_edits_are_rejected_before_network(changes: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = "grok-imagine-image-2.0"
    config = InvokeAIAppConfig(
        external_custom_api_key="fake-key",
        external_custom_base_url="https://images.example",
        external_custom_models=[{"id": model_id, "name": "Grok", "protocol": "xai"}],
    )
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ExternalProviderRequestError, match="text-to-image only"):
        CustomOpenAIProvider(config, logging.getLogger("test")).generate(
            dataclasses.replace(_request(model_id), **changes)
        )
    post.assert_not_called()
