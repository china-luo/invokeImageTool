/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ExternalProviderConfig, ExternalProviderModel } from '@features/models/data/api';

import { createListCollection, HStack, Input, Stack, Text } from '@chakra-ui/react';
import { clearExternalProviderConfig, saveExternalProviderConfig } from '@features/models/data/externalProvidersStore';
import { getModelsSnapshot, refreshModels } from '@features/models/data/modelsStore';
import { KeyCardShell } from '@features/models/ui/credentials/KeyCardShell';
import { clearHighlightedProvider } from '@features/models/ui/uiStore';
import { createUuid } from '@platform/browser/randomUuid';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { Field } from '@platform/ui/Field';
import { Select } from '@platform/ui/Select';
import { PlugIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

type ModelDraft = ExternalProviderModel & { protocol: 'openai' | 'xai' | 'gemini'; rowKey: string };
const PROTOCOL_VALUES = { openai: ['openai'], xai: ['xai'], gemini: ['gemini'] };
const draftRows = (models: ExternalProviderModel[] = []): ModelDraft[] =>
  (models.length ? models : [{ id: '', name: '', protocol: 'openai' as const }]).map((model) => ({
    ...model,
    protocol: model.protocol ?? 'openai',
    rowKey: createUuid(),
  }));

export const CustomProviderKeyCard = ({
  config,
  isHighlighted,
  onError,
}: {
  config: ExternalProviderConfig;
  isHighlighted: boolean;
  onError: (message: string) => void;
}) => {
  const { t } = useTranslation();
  const baseUrlRef = useRef<HTMLInputElement>(null);
  const addModelRef = useRef<HTMLButtonElement>(null);
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState(config.base_url ?? '');
  const [models, setModels] = useState(() => draftRows(config.models));
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const { isBusy, run } = useScopedAction();
  const revealRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (!node || !isHighlighted) {
        return;
      }
      node.scrollIntoView({ block: 'nearest' });
      setRevealed(true);
      clearHighlightedProvider();
    },
    [isHighlighted]
  );
  const protocols = useMemo(
    () =>
      createListCollection({
        items: [
          { value: 'openai', label: t('models.customProvider.openaiProtocol') },
          { value: 'xai', label: t('models.customProvider.xaiProtocol') },
          { value: 'gemini', label: t('models.customProvider.geminiProtocol') },
        ],
      }),
    [t]
  );
  const normalizedModels = models.map(({ id, name, protocol }) => ({ id: id.trim(), name: name.trim(), protocol }));
  const savedModels = (config.models ?? []).map(({ id, name, protocol }) => ({
    id,
    name,
    protocol: protocol ?? 'openai',
  }));
  const comparableModels =
    normalizedModels.length === 1 && !normalizedModels[0]?.id && !normalizedModels[0]?.name ? [] : normalizedModels;
  const hasChanges =
    apiKey.trim().length > 0 ||
    baseUrl.trim() !== (config.base_url ?? '') ||
    JSON.stringify(comparableModels) !== JSON.stringify(savedModels);
  let validation: string | null = null;

  try {
    const url = new URL(baseUrl.trim());
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
      validation = t('models.customProvider.invalidUrl');
    }
  } catch {
    validation = t('models.customProvider.invalidUrl');
  }
  if (!config.api_key_configured && !apiKey.trim()) {
    validation ??= t('models.customProvider.keyRequired');
  }
  if (normalizedModels.some((model) => !model.id || !model.name)) {
    validation ??= t('models.customProvider.modelRequired');
  }
  if (new Set(normalizedModels.map((model) => model.id)).size !== normalizedModels.length) {
    validation ??= t('models.customProvider.duplicateModel');
  }

  const changeDraft = () => {
    setSaved(false);
    setError(null);
  };
  const perform = (clear: boolean) => {
    if (!clear && (validation || !hasChanges)) {
      return;
    }
    setError(null);
    setSaved(false);
    void run(
      async (owner) => {
        const next = clear
          ? await clearExternalProviderConfig('custom')
          : await saveExternalProviderConfig('custom', {
              ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
              base_url: baseUrl.trim(),
              models: normalizedModels,
            });
        assertAccountScopeCurrent(owner);
        setApiKey('');
        setBaseUrl(next.base_url ?? '');
        setModels((rows) =>
          draftRows(next.models).map((row, index) => ({ ...row, rowKey: rows[index]?.rowKey ?? row.rowKey }))
        );
        setSaved(true);
        await refreshModels(owner);
        assertAccountScopeCurrent(owner);
        if (getModelsSnapshot().error) {
          setError(t('models.customProvider.refreshFailed'));
        }
      },
      (_message, cause) => {
        const message = getApiErrorMessage(cause, t('common.somethingWentWrong'));
        setError(message);
        onError(message);
      }
    ).then((success) => {
      if (clear && success) {
        requestAnimationFrame(() => baseUrlRef.current?.focus());
      }
    });
  };

  return (
    <KeyCardShell
      ref={revealRef}
      description={t('models.customProvider.description')}
      icon={PlugIcon}
      isHighlighted={revealed}
      status={{
        label: config.api_key_configured ? t('models.keyStatus.configured') : t('models.keyStatus.unknown'),
        palette: config.api_key_configured ? 'green' : 'gray',
      }}
      title={t('models.customProvider.title')}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          perform(false);
        }}
      >
        <Stack gap="2.5">
          <Text color="fg.muted" fontSize="sm">
            {t('models.customProvider.protocol')}
          </Text>
          <Field label={t('models.customProvider.baseUrl')}>
            <Input
              ref={baseUrlRef}
              disabled={isBusy}
              placeholder="https://api.example.com/v1"
              value={baseUrl}
              onChange={(event) => {
                changeDraft();
                setBaseUrl(event.currentTarget.value);
              }}
            />
          </Field>
          <Field label={t('models.apiKey')}>
            <Input
              autoComplete="off"
              disabled={isBusy}
              placeholder={config.api_key_configured ? t('models.apiKeyConfigured') : t('models.apiKey')}
              type="password"
              value={apiKey}
              onChange={(event) => {
                changeDraft();
                setApiKey(event.currentTarget.value);
              }}
            />
          </Field>
          {models.map((model, index) => (
            <Stack key={model.rowKey} borderColor="border.subtle" borderWidth="1px" gap="1.5" p="2" rounded="md">
              <Field label={t('models.customProvider.modelProtocol', { index: index + 1 })}>
                <Select
                  collection={protocols}
                  disabled={isBusy}
                  value={PROTOCOL_VALUES[model.protocol]}
                  onValueChange={({ value }) => {
                    const protocol = value[0];
                    if (protocol !== 'openai' && protocol !== 'xai' && protocol !== 'gemini') {
                      return;
                    }
                    changeDraft();
                    setModels(models.map((row) => (row.rowKey === model.rowKey ? { ...row, protocol } : row)));
                  }}
                />
              </Field>
              <Field label={t('models.customProvider.modelId', { index: index + 1 })}>
                <Input
                  disabled={isBusy}
                  maxLength={256}
                  placeholder="gpt-image-1"
                  value={model.id}
                  onChange={(event) => {
                    changeDraft();
                    setModels(
                      models.map((row) =>
                        row.rowKey === model.rowKey ? { ...row, id: event.currentTarget.value } : row
                      )
                    );
                  }}
                />
              </Field>
              <Field label={t('models.customProvider.modelName', { index: index + 1 })}>
                <Input
                  disabled={isBusy}
                  maxLength={200}
                  value={model.name}
                  onChange={(event) => {
                    changeDraft();
                    setModels(
                      models.map((row) =>
                        row.rowKey === model.rowKey ? { ...row, name: event.currentTarget.value } : row
                      )
                    );
                  }}
                />
              </Field>
              <Button
                alignSelf="start"
                aria-label={t('models.customProvider.removeModel', { index: index + 1 })}
                disabled={isBusy || models.length === 1}
                type="button"
                variant="ghost"
                onClick={() => {
                  changeDraft();
                  setModels(models.filter((row) => row.rowKey !== model.rowKey));
                  requestAnimationFrame(() => addModelRef.current?.focus());
                }}
              >
                {t('models.customProvider.removeModel', { index: index + 1 })}
              </Button>
            </Stack>
          ))}
          <Button
            ref={addModelRef}
            alignSelf="start"
            disabled={isBusy || models.length >= 50}
            type="button"
            variant="outline"
            onClick={() => {
              changeDraft();
              setModels([...models, { id: '', name: '', protocol: 'openai', rowKey: createUuid() }]);
            }}
          >
            {t('models.customProvider.addModel')}
          </Button>
          {validation && hasChanges ? (
            <Text color="fg.muted" fontSize="sm">
              {validation}
            </Text>
          ) : null}
          {error ? (
            <Text color="fg.error" fontSize="sm" role="alert">
              {error}
            </Text>
          ) : null}
          {saved ? (
            <Text color="fg.muted" fontSize="sm" role="status">
              {t('models.customProvider.saved')}
            </Text>
          ) : null}
          <HStack gap="1.5">
            <Button disabled={isBusy || !hasChanges || !!validation} loading={isBusy} type="submit" variant="solid">
              {t('common.save')}
            </Button>
            {config.api_key_configured || config.base_url || config.models?.length ? (
              <Button disabled={isBusy} type="button" variant="ghost" onClick={() => perform(true)}>
                {t('common.clear')}
              </Button>
            ) : null}
          </HStack>
          <Text color="fg.muted" fontSize="sm">
            {t('models.customProvider.clearHint')}
          </Text>
        </Stack>
      </form>
    </KeyCardShell>
  );
};
