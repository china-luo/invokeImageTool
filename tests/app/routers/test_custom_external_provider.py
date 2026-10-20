"""Custom endpoint configuration is isolated from real credentials and model directories."""

import base64
import dataclasses
import io
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers import app_info
from invokeai.app.services.config.config_default import (
    InvokeAIAppConfig,
    load_and_migrate_config,
    load_external_api_keys,
)
from invokeai.app.services.external_generation.errors import ExternalProviderCapabilityError
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalReferenceImage,
)
from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService
from invokeai.app.services.external_generation.providers.openai import CustomOpenAIProvider
from invokeai.app.services.external_generation.startup import sync_custom_external_models
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_records.model_records_sql import ModelRecordServiceSQL
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

URL = "/api/v1/app/external_providers/config/custom"
PAYLOAD = {
    "api_key": "fake-custom-key",
    "base_url": "https://images.example/v1/",
    "models": [{"id": "vendor/image-v2", "name": "My model"}],
}
EXPECTED_MODELS = [{"id": "vendor/image-v2", "name": "My model", "protocol": "openai"}]


@pytest.fixture
def custom_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_invoker: Invoker) -> SimpleNamespace:
    config = InvokeAIAppConfig()
    config._root = tmp_path
    store = ModelRecordServiceSQL(mock_invoker.services.database, logging.getLogger("test"))
    monkeypatch.setattr(mock_invoker.services, "model_manager", SimpleNamespace(store=store))
    monkeypatch.setattr(mock_invoker.services.configuration, "multiuser", False)
    monkeypatch.setattr(ApiDependencies, "invoker", mock_invoker, raising=False)
    monkeypatch.setattr(app_info, "get_config", lambda: config)
    return SimpleNamespace(config=config, store=store)


def _custom_models(store: ModelRecordServiceSQL) -> list[ExternalApiModelConfig]:
    return [
        model
        for model in store.search_by_attr(
            base_model=BaseModelType.External, model_type=ModelType.ExternalImageGenerator
        )
        if model.provider_id == "custom"
    ]


def test_custom_configuration_registers_models_and_survives_restart(
    custom_setup: SimpleNamespace, client: TestClient
) -> None:
    response = client.post(URL, json=PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {
        "provider_id": "custom",
        "api_key_configured": True,
        "base_url": "https://images.example/v1",
        "models": EXPECTED_MODELS,
    }
    assert "fake-custom-key" not in response.text
    models = _custom_models(custom_setup.store)
    assert [(model.provider_model_id, model.name) for model in models] == [("vendor/image-v2", "My model")]
    assert models[0].capabilities.modes == ["txt2img"]
    assert models[0].capabilities.supports_negative_prompt is False
    assert models[0].path == "external://custom/vendor/image-v2"
    assert app_info._redact_config_secrets(custom_setup.config).external_custom_api_key == app_info.REDACTED_SECRET

    reloaded = load_and_migrate_config(custom_setup.config.config_file_path)
    assert reloaded.external_custom_api_key is None
    assert "fake-custom-key" not in custom_setup.config.config_file_path.read_text()
    reloaded.update_config(load_external_api_keys(custom_setup.config.api_keys_file_path))
    assert reloaded.external_custom_api_key == "fake-custom-key"
    # Startup re-creates missing metadata; repeated sync preserves the model's identity.
    custom_setup.store.del_model(models[0].key)
    sync_custom_external_models(reloaded, custom_setup.store)
    restored = _custom_models(custom_setup.store)[0]
    sync_custom_external_models(reloaded, custom_setup.store)
    assert _custom_models(custom_setup.store)[0].key == restored.key


def test_custom_model_edits_keep_keys_and_remove_only_custom_records(
    custom_setup: SimpleNamespace, client: TestClient
) -> None:
    other = custom_setup.store.add_model(
        ExternalApiModelConfig(
            name="Official",
            provider_id="openai",
            provider_model_id="gpt-image-1",
            capabilities=ExternalModelCapabilities(),
        )
    )
    assert client.post(URL, json=PAYLOAD).status_code == 200
    key = _custom_models(custom_setup.store)[0].key
    response = client.post(
        URL, json={"models": [{"id": "vendor/image-v2", "name": "Renamed"}, {"id": "image-3", "name": "Another"}]}
    )
    assert response.status_code == 200
    assert custom_setup.store.get_model(key).name == "Renamed"
    assert custom_setup.config.external_custom_api_key == "fake-custom-key"
    assert client.post(URL, json={"models": [{"id": "image-3", "name": "Another"}]}).status_code == 200
    assert not custom_setup.store.exists(key)
    assert custom_setup.store.exists(other.key)
    cleared = client.delete(URL)
    assert cleared.status_code == 200
    assert cleared.json() == {"provider_id": "custom", "api_key_configured": False, "base_url": None, "models": []}
    assert not _custom_models(custom_setup.store)
    assert custom_setup.store.exists(other.key)
    assert "external_custom_api_key" not in load_external_api_keys(custom_setup.config.api_keys_file_path)
    assert load_and_migrate_config(custom_setup.config.config_file_path).external_custom_models == []


def test_empty_key_disables_but_preserves_custom_model_definitions(
    custom_setup: SimpleNamespace, client: TestClient
) -> None:
    assert client.post(URL, json=PAYLOAD).status_code == 200
    response = client.post(URL, json={"api_key": " "})
    assert response.status_code == 200
    assert not _custom_models(custom_setup.store)
    assert response.json()["models"] == EXPECTED_MODELS
    assert client.post(URL, json={"api_key": "new-fake-key"}).status_code == 200
    assert len(_custom_models(custom_setup.store)) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"base_url": "file:///images"},
        {"base_url": "https://user:secret@images.example"},
        {"base_url": "https://images.example/?api_key=secret"},
        {"base_url": "https://images.example/#fragment"},
        {"base_url": "https://images.example:invalid"},
        {"base_url": ""},
        {"models": []},
        {"models": None},
        {"models": [{"id": " ", "name": "Name"}]},
        {"models": [{"id": "same", "name": "One"}, {"id": "same", "name": "Two"}]},
        {"models": [{"id": "a" * 257, "name": "Name"}]},
        {"models": [{"id": "gemini-image", "name": "Gemini", "protocol": "unknown"}]},
        {"models": [{"id": str(i), "name": str(i)} for i in range(51)]},
    ],
)
def test_invalid_custom_update_does_not_mutate_configuration(
    changes: dict, custom_setup: SimpleNamespace, client: TestClient
) -> None:
    assert client.post(URL, json=PAYLOAD).status_code == 200
    before = custom_setup.config.model_dump()
    before_files = [
        path.read_bytes() for path in (custom_setup.config.config_file_path, custom_setup.config.api_keys_file_path)
    ]
    assert client.post(URL, json=changes).status_code == 422
    assert custom_setup.config.model_dump() == before
    assert [
        path.read_bytes() for path in (custom_setup.config.config_file_path, custom_setup.config.api_keys_file_path)
    ] == before_files
    assert len(_custom_models(custom_setup.store)) == 1


def test_disk_save_failure_restores_runtime_and_persisted_settings(
    custom_setup: SimpleNamespace, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.post(URL, json=PAYLOAD).status_code == 200
    previous = [
        path.read_bytes() for path in (custom_setup.config.config_file_path, custom_setup.config.api_keys_file_path)
    ]

    def fail_write(self: InvokeAIAppConfig, path: Path, **kwargs: object) -> None:
        path.write_bytes(b"truncated")
        raise OSError("private filesystem detail")

    monkeypatch.setattr(type(custom_setup.config), "write_file", fail_write)
    response = client.post(URL, json={"api_key": "replacement-fake-key"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to save external provider configuration"
    assert custom_setup.config.external_custom_api_key == "fake-custom-key"
    assert [
        path.read_bytes() for path in (custom_setup.config.config_file_path, custom_setup.config.api_keys_file_path)
    ] == previous


def test_model_sync_failure_reports_retry_and_recovers(
    custom_setup: SimpleNamespace, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = custom_setup.store.sync_external_models

    def fail_add(*args: object, **kwargs: object) -> None:
        raise RuntimeError("sensitive database detail")

    monkeypatch.setattr(custom_setup.store, "sync_external_models", fail_add)
    response = client.post(URL, json=PAYLOAD)
    assert response.status_code == 500
    assert "retry saving" in response.json()["detail"]
    assert "sensitive" not in response.text
    monkeypatch.setattr(custom_setup.store, "sync_external_models", original)
    assert client.post(URL, json={"models": PAYLOAD["models"]}).status_code == 200
    assert len(_custom_models(custom_setup.store)) == 1


@pytest.mark.parametrize("method", ["post", "delete"])
def test_custom_settings_require_administrator(
    method: str, enable_multiuser: object, client: TestClient, user1_token: str
) -> None:
    kwargs = {"json": PAYLOAD} if method == "post" else {}
    assert getattr(client, method)(URL, **kwargs).status_code == 401
    assert getattr(client, method)(URL, headers={"Authorization": f"Bearer {user1_token}"}, **kwargs).status_code == 403


def test_null_models_validation_does_not_echo_submitted_key(custom_setup: SimpleNamespace, client: TestClient) -> None:
    response = client.post(URL, json=PAYLOAD | {"models": None})
    assert response.status_code == 422
    assert "fake-custom-key" not in response.text
    assert custom_setup.config.external_custom_api_key is None


def test_mixed_custom_protocols_persist_and_existing_model_keys_survive_protocol_changes(
    custom_setup: SimpleNamespace, client: TestClient
) -> None:
    assert client.post(URL, json=PAYLOAD).status_code == 200
    existing_key = _custom_models(custom_setup.store)[0].key
    models = [
        {"id": "vendor/image-v2", "name": "My model", "protocol": "xai"},
        {"id": "gemini-3.1-flash-image", "name": "Gemini", "protocol": "gemini"},
        {"id": "legacy-image", "name": "Legacy"},
    ]
    response = client.post(URL, json={"models": models})
    assert response.status_code == 200
    expected = [models[0], models[1], models[2] | {"protocol": "openai"}]
    assert response.json()["models"] == expected
    assert custom_setup.store.get_model(existing_key).provider_model_id == "vendor/image-v2"
    reloaded = load_and_migrate_config(custom_setup.config.config_file_path)
    assert [model.model_dump() for model in reloaded.external_custom_models] == expected
    reloaded.update_config(load_external_api_keys(custom_setup.config.api_keys_file_path))
    sync_custom_external_models(reloaded, custom_setup.store)
    assert custom_setup.store.get_model(existing_key).provider_model_id == "vendor/image-v2"
    assert len(_custom_models(custom_setup.store)) == 3


def test_legacy_custom_configuration_without_protocol_loads_as_openai(
    custom_setup: SimpleNamespace,
) -> None:
    path = custom_setup.config.config_file_path
    path.write_text(
        "schema_version: 4.0.3\nexternal_custom_models:\n  - id: grok-imagine-image-2.0\n    name: Existing Grok\n",
        encoding="utf-8",
    )
    reloaded = load_and_migrate_config(path)
    assert reloaded.external_custom_models[0].protocol == "openai"


@pytest.mark.parametrize(
    ("model_id", "protocol", "supports_references"),
    [
        ("gpt-image-2.5-sunburst", "openai", True),
        ("gpt-image-1", "openai", True),
        ("gpt-image-2.5-sunburst", "xai", False),
        ("gpt-image-2.5-sunburst", "gemini", False),
        ("dall-e-3", "openai", False),
        ("vendor/image-v2", "openai", False),
        ("other-gpt-image-2", "openai", False),
        ("gpt-image", "openai", False),
    ],
)
def test_custom_reference_controls_match_model_and_protocol(
    model_id: str, protocol: str, supports_references: bool, custom_setup: SimpleNamespace, client: TestClient
) -> None:
    response = client.post(
        URL, json=PAYLOAD | {"models": [{"id": model_id, "name": "Image model", "protocol": protocol}]}
    )
    assert response.status_code == 200
    model = _custom_models(custom_setup.store)[0]
    assert model.capabilities.supports_reference_images is supports_references
    assert model.capabilities.max_reference_images == (1 if supports_references else None)
    assert model.capabilities.modes == ["txt2img"]
    assert model.panel_schema is not None
    assert [control.name for control in model.panel_schema.prompts] == (
        ["reference_images"] if supports_references else []
    )
    assert [control.name for control in model.panel_schema.image] == ["dimensions"]


@pytest.mark.parametrize("other_protocol", ["xai", "gemini"])
def test_custom_reference_support_refreshes_existing_records_without_changing_keys(
    other_protocol: str, custom_setup: SimpleNamespace, client: TestClient
) -> None:
    model_id = "gpt-image-2.5-sunburst"
    old_model = custom_setup.store.add_model(
        ExternalApiModelConfig(
            name="Existing GPT Image",
            provider_id="custom",
            provider_model_id=model_id,
            capabilities=ExternalModelCapabilities(supports_reference_images=False),
        )
    )
    payload = PAYLOAD | {"models": [{"id": model_id, "name": "Existing GPT Image"}]}
    assert client.post(URL, json=payload).status_code == 200
    reloaded = load_and_migrate_config(custom_setup.config.config_file_path)
    reloaded.update_config(load_external_api_keys(custom_setup.config.api_keys_file_path))
    for _ in range(2):
        sync_custom_external_models(reloaded, custom_setup.store)
        models = _custom_models(custom_setup.store)
        assert len(models) == 1
        assert models[0].key == old_model.key
        assert models[0].capabilities.supports_reference_images is True
        assert models[0].capabilities.max_reference_images == 1

    for protocol, supports_references in [(other_protocol, False), ("openai", True)]:
        assert (
            client.post(
                URL, json={"models": [{"id": model_id, "name": "Existing GPT Image", "protocol": protocol}]}
            ).status_code
            == 200
        )
        models = _custom_models(custom_setup.store)
        assert len(models) == 1
        assert models[0].key == old_model.key
        assert models[0].capabilities.supports_reference_images is supports_references
        assert models[0].capabilities.max_reference_images == (1 if supports_references else None)
        assert models[0].panel_schema is not None
        assert [control.name for control in models[0].panel_schema.prompts] == (
            ["reference_images"] if supports_references else []
        )


def test_synced_custom_reference_generation_refreshes_stale_capabilities_and_limits_inputs(
    custom_setup: SimpleNamespace, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id = "gpt-image-2.5-sunburst"
    stale_model = custom_setup.store.add_model(
        ExternalApiModelConfig(
            name="Existing GPT Image",
            provider_id="custom",
            provider_model_id=model_id,
            capabilities=ExternalModelCapabilities(supports_reference_images=False),
        )
    )
    assert (
        client.post(URL, json=PAYLOAD | {"models": [{"id": model_id, "name": "Existing GPT Image"}]}).status_code == 200
    )
    output = io.BytesIO()
    Image.new("RGB", (3, 5), (4, 120, 200)).save(output, format="PNG")
    post = Mock(
        return_value=Mock(
            ok=True,
            status_code=200,
            headers={"x-request-id": "reference-request"},
            json=Mock(return_value={"data": [{"b64_json": base64.b64encode(output.getvalue()).decode("ascii")}]}),
        )
    )
    monkeypatch.setattr(requests, "post", post)
    logger = logging.getLogger("test")
    service = ExternalGenerationService(
        {"custom": CustomOpenAIProvider(custom_setup.config, logger)}, logger, record_store=custom_setup.store
    )
    reference = Image.new("RGBA", (7, 9), (20, 40, 60, 128))
    request = ExternalGenerationRequest(
        model=stale_model,
        mode="txt2img",
        prompt="Keep the subject and replace the background with a beach",
        seed=None,
        num_images=1,
        width=768,
        height=1368,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=[ExternalReferenceImage(image=reference)],
        metadata=None,
        provider_options={"quality": "high", "background": "opaque", "input_fidelity": "high"},
    )
    with pytest.raises(ExternalProviderCapabilityError, match="at most 1 reference images"):
        service.generate(dataclasses.replace(request, reference_images=request.reference_images * 2))
    post.assert_not_called()

    result = service.generate(request)

    post.assert_called_once()
    assert post.call_args.args == ("https://images.example/v1/images/edits",)
    assert post.call_args.kwargs["data"] == {
        "model": model_id,
        "prompt": "Keep the subject and replace the background with a beach",
        "n": 1,
        "size": "768x1368",
        "output_format": "png",
        "quality": "high",
        "background": "opaque",
        "input_fidelity": "high",
    }
    assert post.call_args.kwargs["timeout"] == 300
    assert len(post.call_args.kwargs["files"]) == 1
    field, (filename, image_buffer, content_type) = post.call_args.kwargs["files"][0]
    assert (field, filename, content_type) == ("image", "image_0.png", "image/png")
    with Image.open(image_buffer) as sent_reference:
        assert sent_reference.format == "PNG"
        assert sent_reference.size == (7, 9)
        assert sent_reference.getpixel((0, 0)) == (20, 40, 60, 128)
    assert len(result.images) == 1
    assert result.images[0].image.size == (3, 5)
    assert result.images[0].image.getpixel((0, 0)) == (4, 120, 200)
    assert result.provider_request_id == "reference-request"
