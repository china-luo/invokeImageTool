import type * as ModelsApi from '@features/models/data/api';
import type { ExternalProviderConfig, ExternalProviderConfigUpdate } from '@features/models/data/api';

import { ChakraProvider } from '@chakra-ui/react';
import {
  getExternalProvidersSnapshot,
  setExternalProvidersSnapshotForTests,
} from '@features/models/data/externalProvidersStore';
import { getModelsSnapshot } from '@features/models/data/modelsStore';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { ExternalProviderKeyCards } from './ExternalProviderKeyCards';

const api = vi.hoisted(() => ({
  getModelsDir: vi.fn(() => Promise.resolve('/models')),
  listMissingModels: vi.fn(() => Promise.resolve([])),
  listModels: vi.fn(() => Promise.resolve([])),
  resetExternalProviderConfig: vi.fn(),
  setExternalProviderConfig: vi.fn(),
}));
vi.mock('@features/models/data/api', async (original) => ({ ...(await original<typeof ModelsApi>()), ...api }));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: { index?: number }) => (values?.index ? `${key} ${String(values.index)}` : key),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const EMPTY: ExternalProviderConfig = { api_key_configured: false, base_url: null, models: [], provider_id: 'custom' };
const CONFIGURED: ExternalProviderConfig = {
  ...EMPTY,
  api_key_configured: true,
  base_url: 'https://relay.example/v1',
  models: [{ id: 'custom-image', name: 'My image model' }],
};
const input = (label: string) => page.getByLabelText(label, { exact: true });
const save = () => page.getByRole('button', { name: 'common.save', exact: true });
const fill = (label: string, value: string) => act(() => input(label).fill(value));
const clickSave = () => act(() => save().click());
const protocolControl = (index: number) =>
  page.getByRole('combobox', { name: `models.customProvider.modelProtocol ${String(index)}`, exact: true });
const selectProtocol = async (index: number, protocol: 'openai' | 'xai' | 'gemini') => {
  await act(() => protocolControl(index).click());
  await act(() => page.getByRole('option', { name: `models.customProvider.${protocol}Protocol`, exact: true }).click());
};

describe('custom image provider configuration', () => {
  let root: Root;
  let host: HTMLDivElement;
  const onError = vi.fn();

  const mount = async (config = EMPTY) => {
    setExternalProvidersSnapshotForTests({ configs: [config], error: null, status: 'loaded' });
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <ExternalProviderKeyCards onError={onError} />
        </ChakraProvider>
      )
    );
  };
  const fillNew = async () => {
    await fill('models.customProvider.baseUrl', 'https://relay.example/v1');
    await fill('models.apiKey', 'test-api-key');
    await fill('models.customProvider.modelId 1', 'custom-image');
    await fill('models.customProvider.modelName 1', 'My image model');
  };

  beforeEach(() => {
    applyThemeToRoot(DEFAULT_THEME_ID);
    accountLifecycle.activate('custom-api-browser', ':user:custom-api-browser');
    onError.mockReset();
    api.listModels.mockClear();
    api.resetExternalProviderConfig.mockReset().mockResolvedValue(EMPTY);
    api.setExternalProviderConfig
      .mockReset()
      .mockImplementation((_provider: string, changes: ExternalProviderConfigUpdate) =>
        Promise.resolve({ ...CONFIGURED, base_url: changes.base_url, models: changes.models })
      );
    host = document.createElement('div');
    host.style.width = '420px';
    document.body.append(host);
    root = createRoot(host);
  });
  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('validates, saves multiple arbitrary models, and refreshes the model catalog', async () => {
    await mount();
    await expect.element(save()).toBeDisabled();
    await fillNew();
    await fill('models.customProvider.baseUrl', 'https://user:password@relay.example/v1');
    await expect.element(save()).toBeDisabled();
    await expect.element(page.getByText('models.customProvider.invalidUrl')).toBeVisible();
    await fill('models.customProvider.baseUrl', 'https://relay.example/v1');
    await act(() => page.getByRole('button', { name: 'models.customProvider.addModel' }).click());
    await fill('models.customProvider.modelId 2', 'custom-image');
    await fill('models.customProvider.modelName 2', 'Second image model');
    await expect.element(save()).toBeDisabled();
    await expect.element(page.getByText('models.customProvider.duplicateModel')).toBeVisible();
    await fill('models.customProvider.modelId 2', 'another/model');
    await selectProtocol(2, 'gemini');
    expect(await auditAccessibility(host)).toEqual([]);

    await clickSave();

    await expect.poll(() => api.setExternalProviderConfig.mock.calls.length).toBe(1);
    expect(api.setExternalProviderConfig).toHaveBeenCalledWith(
      'custom',
      {
        api_key: 'test-api-key',
        base_url: 'https://relay.example/v1',
        models: [
          { id: 'custom-image', name: 'My image model', protocol: 'openai' },
          { id: 'another/model', name: 'Second image model', protocol: 'gemini' },
        ],
      },
      expect.any(AbortSignal)
    );
    await expect.poll(() => api.listModels.mock.calls.length).toBe(1);
    expect(getModelsSnapshot().status).toBe('loaded');
    await expect.element(input('models.apiKey')).toHaveValue('');
    await expect.element(save()).toBeDisabled();
    await expect.element(page.getByRole('status')).toHaveTextContent('models.customProvider.saved');
  });

  it('edits and removes model rows without resending a saved key, and clears the whole configuration', async () => {
    await mount(CONFIGURED);
    await expect.element(input('models.apiKey')).toHaveValue('');
    await expect.element(protocolControl(1)).toHaveTextContent('models.customProvider.openaiProtocol');
    await expect.element(save()).toBeDisabled();
    await act(() => page.getByRole('button', { name: 'models.customProvider.addModel' }).click());
    await act(() => page.getByRole('button', { name: 'models.customProvider.removeModel 2' }).click());
    await expect.element(page.getByRole('button', { name: 'models.customProvider.addModel' })).toHaveFocus();
    await fill('models.customProvider.modelName 1', 'Renamed image model');
    await clickSave();
    await expect.element(save()).toBeDisabled();
    expect(api.setExternalProviderConfig).toHaveBeenCalledWith(
      'custom',
      {
        base_url: CONFIGURED.base_url,
        models: [{ id: 'custom-image', name: 'Renamed image model', protocol: 'openai' }],
      },
      expect.any(AbortSignal)
    );

    await act(() => page.getByRole('button', { name: 'common.clear', exact: true }).click());

    expect(api.resetExternalProviderConfig).toHaveBeenCalledWith('custom', expect.any(AbortSignal));
    await expect.poll(() => api.listModels.mock.calls.length).toBe(2);
    await expect.element(input('models.customProvider.baseUrl')).toHaveValue('');
    await expect.element(input('models.customProvider.baseUrl')).toHaveFocus();
    await expect.element(input('models.customProvider.modelId 1')).toHaveValue('');
    expect(getExternalProvidersSnapshot().configs).toEqual([EMPTY]);
  });

  it('preserves a legacy Grok row and lets its protocol change independently', async () => {
    const grok = { ...CONFIGURED, models: [{ id: 'grok-imagine-image', name: 'Existing Grok' }] };
    await mount(grok);
    await expect.element(save()).toBeDisabled();
    await expect.element(protocolControl(1)).toHaveTextContent('models.customProvider.openaiProtocol');
    await selectProtocol(1, 'xai');
    await expect.element(save()).toBeEnabled();
    await selectProtocol(1, 'openai');
    await expect.element(save()).toBeDisabled();
    await selectProtocol(1, 'xai');
    await clickSave();

    expect(api.setExternalProviderConfig).toHaveBeenCalledWith(
      'custom',
      {
        base_url: CONFIGURED.base_url,
        models: [{ id: 'grok-imagine-image', name: 'Existing Grok', protocol: 'xai' }],
      },
      expect.any(AbortSignal)
    );
    await expect.element(input('models.customProvider.modelId 1')).toHaveValue('grok-imagine-image');
    await expect.element(input('models.customProvider.modelName 1')).toHaveValue('Existing Grok');
    await expect.element(protocolControl(1)).toHaveTextContent('models.customProvider.xaiProtocol');
    await expect.element(save()).toBeDisabled();
  });

  it('keeps the drafts after a failed save and supports retry', async () => {
    api.setExternalProviderConfig.mockRejectedValueOnce(new Error('Configuration could not be saved.'));
    await mount();
    await fillNew();
    await selectProtocol(1, 'gemini');
    await clickSave();

    await expect.element(page.getByRole('alert')).toHaveTextContent('Configuration could not be saved.');
    await expect.element(input('models.apiKey')).toHaveValue('test-api-key');
    await expect.element(input('models.customProvider.modelName 1')).toHaveValue('My image model');
    expect(api.listModels).not.toHaveBeenCalled();
    await expect.element(protocolControl(1)).toHaveTextContent('models.customProvider.geminiProtocol');
    await clickSave();
    await expect.poll(() => api.listModels.mock.calls.length).toBe(1);
    await expect.element(page.getByRole('alert')).not.toBeInTheDocument();
  });

  it('disables the form while saving and ignores a response from an ended account lifetime', async () => {
    let resolve!: (config: ExternalProviderConfig) => void;
    api.setExternalProviderConfig.mockReturnValueOnce(
      new Promise<ExternalProviderConfig>((done) => {
        resolve = done;
      })
    );
    await mount();
    await fillNew();
    await clickSave();
    await expect.element(input('models.apiKey')).toBeDisabled();
    await expect.element(protocolControl(1)).toBeDisabled();
    await expect.element(page.getByRole('button', { name: 'models.customProvider.addModel' })).toBeDisabled();
    await act(() => {
      accountLifecycle.activate('custom-api-another-account', ':user:custom-api-another-account');
      resolve(CONFIGURED);
    });
    await vi.waitFor(() => expect(getExternalProvidersSnapshot().configs).toBeNull());
    expect(api.listModels).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });
});
