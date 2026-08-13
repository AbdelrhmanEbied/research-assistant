import { api, showToast } from './utils.js';

const settingsModal = document.getElementById('settingsModal');
const settingsModalClose = document.getElementById('settingsModalClose');
const settingsBtn = document.getElementById('settingsBtn');
const setProvider = document.getElementById('setProvider');
const setModel = document.getElementById('setModel');
const setApiKey = document.getElementById('setApiKey');
const setLlmSave = document.getElementById('setLlmSave');
const setSearchType = document.getElementById('setSearchType');
const setSearchDepth = document.getElementById('setSearchDepth');
const setRetrieveLimit = document.getElementById('setRetrieveLimit');
const setRerank = document.getElementById('setRerank');
const setRetrievalSave = document.getElementById('setRetrievalSave');
const settingsStatus = document.getElementById('settingsStatus');

async function loadSettings() {
  try {
    const res = await api('/settings/');
    const s = await res.json();
    setProvider.value = s.llm.default_provider;
    setModel.value = s.llm.default_model;
    const hasKey = s.providers && s.providers[s.llm.default_provider] && s.providers[s.llm.default_provider].has_api_key;
    setApiKey.placeholder = hasKey ? 'Stored — leave blank to keep' : 'Uses .env';
    setSearchType.value = s.retrieval.search_type;
    setSearchDepth.value = s.retrieval.search_depth || 'basic';
    setRetrieveLimit.value = s.retrieval.limit;
    setRerank.checked = !!s.retrieval.rerank;
    settingsStatus.textContent = `Default: ${s.llm.default_provider} · ${s.llm.default_model}`;
  } catch (_) {
    settingsStatus.textContent = 'Could not load settings';
  }
}

export function openSettingsModal() { settingsModal.hidden = false; loadSettings(); }
export function closeSettingsModal() { settingsModal.hidden = true; }

setLlmSave.addEventListener('click', async () => {
  const model = setModel.value.trim();
  const provider = setProvider.value;
  if (!model) { showToast('Enter a model name first'); setModel.focus(); return; }
  try {
    await api('/settings/llm', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, model_provider: provider }),
    });
    const key = setApiKey.value.trim();
    if (key) {
      await api('/settings/api-keys', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, api_key: key }),
      });
      setApiKey.value = '';
    }
    showToast('Model settings saved');
    loadSettings();
  } catch (_) {
    showToast('Failed to save model settings');
  }
});

setRetrievalSave.addEventListener('click', async () => {
  const limit = Number(setRetrieveLimit.value);
  if (Number.isNaN(limit) || limit < 1 || limit > 50) { showToast('Limit must be 1–50'); return; }
  try {
    await api('/settings/retrieval', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        search_type: setSearchType.value,
        search_depth: setSearchDepth.value,
        limit,
        rerank: setRerank.checked,
      }),
    });
    showToast('Retrieval settings saved');
    loadSettings();
  } catch (_) {
    showToast('Failed to save retrieval settings');
  }
});

settingsBtn.addEventListener('click', openSettingsModal);
settingsModalClose.addEventListener('click', closeSettingsModal);
settingsModal.addEventListener('click', (e) => { if (e.target === settingsModal) closeSettingsModal(); });