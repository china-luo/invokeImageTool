from logging import Logger
from typing import TYPE_CHECKING

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges, ModelRecordServiceBase
from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelConfig,
    ExternalApiModelDefaultSettings,
    ExternalModelCapabilities,
    ExternalModelPanelControl,
    ExternalModelPanelSchema,
)
from invokeai.backend.model_manager.starter_models import STARTER_MODELS
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

if TYPE_CHECKING:
    from invokeai.app.services.model_manager.model_manager_base import ModelManagerServiceBase


def sync_configured_external_starter_models(
    configured_provider_ids: set[str],
    model_manager: "ModelManagerServiceBase",
    logger: Logger,
) -> list[str]:
    """Queue missing external starter models for configured providers."""

    if not configured_provider_ids:
        return []

    installed_sources = {
        model.source
        for model in model_manager.store.search_by_attr(
            base_model=BaseModelType.External,
            model_type=ModelType.ExternalImageGenerator,
        )
        if isinstance(model, ExternalApiModelConfig) and model.source
    }

    queued_sources: list[str] = []
    for starter_model in STARTER_MODELS:
        if not starter_model.source.startswith("external://"):
            continue

        provider_id = starter_model.source.removeprefix("external://").split("/", 1)[0]
        if provider_id not in configured_provider_ids:
            continue

        if starter_model.source in installed_sources:
            continue

        model_manager.install.heuristic_import(
            starter_model.source,
            config=ModelRecordChanges(
                name=starter_model.name,
                base=starter_model.base,
                type=starter_model.type,
                description=starter_model.description,
                format=starter_model.format,
                capabilities=starter_model.capabilities,
                default_settings=starter_model.default_settings,
            ),
        )
        queued_sources.append(starter_model.source)
        logger.info("Queued external starter model sync for %s", starter_model.source)

    return queued_sources


def sync_custom_external_models(config: InvokeAIAppConfig, store: ModelRecordServiceBase) -> None:
    """Reconcile metadata-only models synchronously; the persisted config is the restart recovery source."""
    desired = (
        config.external_custom_models if config.external_custom_api_key and config.external_custom_base_url else []
    )
    store.sync_external_models(
        "custom",
        [
            ExternalApiModelConfig(
                name=model.name,
                provider_id="custom",
                provider_model_id=model.id,
                capabilities=ExternalModelCapabilities(
                    modes=["txt2img"],
                    supports_reference_images=False,
                    supports_negative_prompt=False,
                    max_images_per_request=1,
                ),
                default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
                panel_schema=ExternalModelPanelSchema(image=[ExternalModelPanelControl(name="dimensions")]),
            )
            for model in desired
        ],
    )
