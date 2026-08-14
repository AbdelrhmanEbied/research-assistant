import { dom, state } from './state.js';
import { api, showToast } from './utils.js';
import { ensureConversation, loadConversations } from './conversations.js';
import { addRow, showTypingIndicator, createTypewriter, parseTail, renderMessageActions, renderSources, createThinkingPanel, setThinkingText } from './render.js';
import { typesetMath } from './markdown.js';

const inputEl = dom.input;
const sendBtn = dom.sendBtn;
const sourceSelectEl = dom.sourceSelect;
const advancedBtnEl = document.getElementById('advancedBtn');
const advancedPopoverEl = document.getElementById('advancedPopover');
const reqSearchTypeEl = document.getElementById('reqSearchType');
const reqSearchDepthEl = document.getElementById('reqSearchDepth');
const reqRetrieveLimitEl = document.getElementById('reqRetrieveLimit');
const reqRerankEl = document.getElementById('reqRerank');

export function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 170) + 'px';
}

export function updateSendState() {
  if (state.isStreaming) {
    sendBtn.classList.add('stop');
    sendBtn.disabled = false;
    sendBtn.title = 'Stop generating';
    sendBtn.setAttribute('aria-label', 'Stop generating');
  } else {
    sendBtn.classList.remove('stop');
    sendBtn.disabled = inputEl.value.trim().length === 0;
    sendBtn.title = 'Send';
    sendBtn.setAttribute('aria-label', 'Send message');
  }
}

function selectedSource() {
  const v = sourceSelectEl.value;
  return v === 'auto' ? undefined : v;
}

function retrievalConfig() {
  const cfg = {};
  if (reqSearchTypeEl.value) cfg.search_type = reqSearchTypeEl.value;
  if (reqSearchDepthEl.value) cfg.search_depth = reqSearchDepthEl.value;
  if (reqRetrieveLimitEl.value) {
    const n = Number(reqRetrieveLimitEl.value);
    if (n >= 1 && n <= 50) cfg.limit = n;
  }
  if (reqRerankEl.checked) cfg.rerank = true;
  return Object.keys(cfg).length ? cfg : undefined;
}

// streams a chat or regenerate response into an assistant bubble and wires
// up the sources / details / regenerate affordances on completion
async function streamInto(contentEl, row, { path, body }) {
  const controller = new AbortController();
  state.currentController = controller;

  state.isStreaming = true;
  state.userStopped = false;
  updateSendState();

  const thinkingMode = body.agent_mode === 'thinking';
  let thinkingPanel = null;

  let res;
  try {
    res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => '');
      throw new Error(`${res.status} ${res.statusText}: ${errText}`);
    }
  } catch (err) {
    state.isStreaming = false;
    state.currentController = null;
    updateSendState();
    if (err.name === 'AbortError' || controller.signal.aborted) {
      finishStream(contentEl, '', null, null, null, state.userStopped, false, null, thinkingPanel);
      return;
    }
    finishStream(contentEl, '', null, null, err, false, false, null, thinkingPanel);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const typer = createTypewriter(contentEl);
  let full = '';
  let networkBroken = false;

  while (true) {
    let done = false;
    let piece = null;
    try {
      const r = await reader.read();
      done = r.done;
      piece = r.done ? null : decoder.decode(r.value, { stream: true });
    } catch (_) {
      networkBroken = true;
      break;
    }
    if (done) break;
    if (!piece) continue;
    full += piece;
    const parsed = parseTail(full);
    typer.push(parsed.text);
    // create the thinking panel lazily on the first real content so a model
    // that exposes no thoughts doesn't show an empty/fake reasoning section
    if (thinkingMode && parsed.thinking) {
      if (!thinkingPanel) thinkingPanel = createThinkingPanel(row);
      setThinkingText(thinkingPanel, parsed.thinking);
    }
  }

  try { await typer.finish(); } catch (_) {}

  state.isStreaming = false;
  state.currentController = null;
  updateSendState();

  const parsed = parseTail(full);
  finishStream(
    contentEl,
    parsed.text,
    parsed.sources,
    parsed.details,
    parsed.error,
    state.userStopped,
    networkBroken,
    parsed.thinking,
    thinkingPanel
  );
}

function finishStream(contentEl, text, sources, details, error, stopped, networkBroken, thinking, thinkingPanel) {
  const row = contentEl.closest('.row');
  const bubble = contentEl;

  if (error) {
    bubble.innerHTML = '';
    const errSpan = document.createElement('span');
    errSpan.style.color = 'var(--danger)';
    errSpan.textContent = `Error: ${error.message || 'Generation failed'}`;
    bubble.appendChild(errSpan);
    if (thinkingPanel) thinkingPanel.remove();
    renderMessageActions(row, { details: null, stopped: false });
  } else if (stopped && !text.trim()) {
    bubble.innerHTML = '<em style="color:var(--text-dim)">Generation stopped.</em>';
    if (thinkingPanel) thinkingPanel.remove();
    renderMessageActions(row, { details: null, stopped: true });
  } else if (networkBroken && !text.trim()) {
    bubble.innerHTML = '<em style="color:var(--danger)">Stream interrupted — the response did not complete.</em>';
    if (thinkingPanel) thinkingPanel.remove();
    renderMessageActions(row, { details: null, stopped: false });
  } else if (!text.trim()) {
    bubble.innerHTML = '<em style="color:var(--text-dim)">No response.</em>';
    if (thinkingPanel) thinkingPanel.remove();
    renderMessageActions(row, { details: null, stopped: false });
  } else {
    typesetMath(bubble);
    if (sources) renderSources(bubble, sources);
    renderMessageActions(row, { details, stopped });
    scrollToBottom();
  }

  loadConversations();
}

function scrollToBottom() {
  dom.messages.scrollTop = dom.messages.scrollHeight;
}

export async function regenerateLast() {
  if (!state.currentConversationId || state.isStreaming) return;
  const rows = [...dom.messages.querySelectorAll('.row.assistant')];
  const row = rows[rows.length - 1];
  if (!row) return;
  const bubble = row.querySelector('.bubble-content');

  row.querySelectorAll('.msg-actions, .details-panel, .source-list, .sources-count, .thinking-panel').forEach(n => n.remove());
  bubble.innerHTML = '';
  showTypingIndicator(bubble);

  const body = { conversation_id: state.currentConversationId, agent_mode: state.agentMode };
  const src = selectedSource();
  if (src) body.source = src;
  const retr = retrievalConfig();
  if (retr) body.retrieval = retr;

  await streamInto(bubble, row, { path: '/chat/regenerate', body });
}

export async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || state.isStreaming) return;

  const conversationId = await ensureConversation();

  inputEl.value = '';
  autoResize();
  addRow('user', text);
  const assistantContent = addRow('assistant', '');
  const row = assistantContent.closest('.row');
  showTypingIndicator(assistantContent);

  const body = { query: text, conversation_id: conversationId, agent_mode: state.agentMode };
  if (state.pendingMode) { body.mode = state.pendingMode; state.pendingMode = null; }
  const src = selectedSource();
  if (src) body.source = src;
  const retr = retrievalConfig();
  if (retr) body.retrieval = retr;

  await streamInto(assistantContent, row, { path: '/chat/', body });
}

/* ---- wiring ---- */

sendBtn.addEventListener('click', () => {
  if (state.isStreaming) {
    state.userStopped = true;
    if (state.currentController) state.currentController.abort();
  } else {
    sendMessage();
  }
});

inputEl.addEventListener('input', () => { autoResize(); updateSendState(); });
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

advancedBtnEl.addEventListener('click', () => {
  const open = advancedPopoverEl.classList.toggle('open');
  advancedPopoverEl.hidden = !open;
  advancedBtnEl.classList.toggle('active', open);
});

const modeToggleEl = document.querySelector('.mode-toggle');
modeToggleEl.addEventListener('click', (e) => {
  const btn = e.target.closest('.mode-toggle-btn');
  if (!btn) return;
  modeToggleEl.querySelectorAll('.mode-toggle-btn').forEach(b => b.classList.toggle('active', b === btn));
  state.agentMode = btn.dataset.mode;
});

// delegated, the empty state gets replaced wholesale on every remount
dom.messages.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  inputEl.value = chip.dataset.prompt;
  autoResize();
  updateSendState();
  inputEl.focus();
});
